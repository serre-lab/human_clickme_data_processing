import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import glob
import numpy as np
import torch
import timm
import faiss
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from joblib import Parallel, delayed

# Path configurations
CLICKME_PATHS = [
    "clickme_test_images_v1",
    "clickme_train_images_v1",
    "clickme_val_images_v1"
]
IMAGENET_TRAIN = "/media/data_cifs/projects/prj_video_imagenet/imagenet/ILSVRC/Data/CLS-LOC/train"
IMAGENET_VAL = "/media/data_cifs/projects/prj_video_imagenet/imagenet/ILSVRC/Data/CLS-LOC/val2"

# Database cache paths
FAISS_INDEX_PATH = "clickme_faiss.index"
REFERENCE_PATHS_CACHE = "clickme_reference_paths.npy"

# Configuration
FORCE_BUILD = False  # Set to True to force rebuild the database
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

def setup_model():
    """Setup the DINO ViT model and transform."""
    model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    model.head = torch.nn.Identity()  # Remove classification head to get embeddings
    model = model.to(DEVICE)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform

def get_embedding(model, transform, image_paths, batch_size=32):
    """Get embeddings for a batch of images using parallel loading."""
    embeddings = []
    valid_paths = []
    
    def load_single_image(img_path):
        """Helper function to load and transform a single image."""
        try:
            if ".npy" in img_path:
                image = Image.fromarray(np.load(img_path))
            else:
                image = Image.open(img_path).convert('RGB')
            return transform(image), img_path
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Parallel load and transform images
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(load_single_image)(path) for path in batch_paths
        )
        
        # Filter out None results and separate images and paths
        batch_images = []
        batch_valid_paths = []
        for img, path in results:
            if img is not None:
                batch_images.append(img)
                batch_valid_paths.append(path)
        
        if not batch_images:
            continue
            
        # Process batch on GPU
        batch_tensor = torch.stack(batch_images).to(DEVICE)
        with torch.no_grad():
            batch_embeddings = model(batch_tensor).cpu().numpy()
        
        embeddings.extend(batch_embeddings)
        valid_paths.extend(batch_valid_paths)
    
    return np.array(embeddings), valid_paths

def process_batch(batch_paths, model, transform):
    """Process a batch of images in parallel."""
    embeddings, valid_paths = get_embedding(model, transform, batch_paths)
    return embeddings, valid_paths

def build_clickme_database(model, transform, rebuild=False):
    """Build FAISS database from ClickMe images using efficient batch processing.
    
    Args:
        model: The DINO ViT model
        transform: Image transformation pipeline
        rebuild: If True, delete existing database before building
    """
    # Check if we should delete existing database
    if rebuild:
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(REFERENCE_PATHS_CACHE):
            os.remove(REFERENCE_PATHS_CACHE)
    
    # Collect all image paths
    all_image_paths = []
    for path in CLICKME_PATHS:
        all_image_paths.extend(glob.glob(os.path.join(path, "*.npy")))
    
    # Initialize FAISS index
    dimension = 384  # ViT-Small DINO embedding dimension
    # Create GPU index
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(dimension)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    valid_paths = []
    
    def load_single_image(img_path):
        """Helper function to load and transform a single image."""
        try:
            image = Image.fromarray(np.load(img_path))
            return transform(image), img_path
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None

    # Process in batches
    batch_size = 128
    for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Processing images"):
        batch_paths = all_image_paths[i:i + batch_size]
        
        # Parallel load and transform images on CPU
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(load_single_image)(path) for path in batch_paths
        )
        
        # Filter out None results and prepare batch
        batch_images = []
        batch_valid_paths = []
        for img, path in results:
            if img is not None:
                batch_images.append(img)
                batch_valid_paths.append(path)
        
        if not batch_images:
            continue
        
        # Process batch on GPU
        batch_tensor = torch.stack(batch_images).to(DEVICE)
        with torch.no_grad():
            batch_embeddings = model(batch_tensor).cpu().numpy().astype('float32')
        
        # Add to FAISS index immediately
        gpu_index.add(batch_embeddings)
        valid_paths.extend(batch_valid_paths)
    
    # Convert back to CPU index for saving
    index = faiss.index_gpu_to_cpu(gpu_index)
    return index, valid_paths

def find_similar_images(model, transform, index, reference_paths, query_paths, batch_size=2048):
    """Find similar images between query images and reference database."""
    # Convert to GPU index for searching
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    similarity_dict = {}
    
    def load_single_image(img_path):
        """Helper function to load and transform a single image."""
        try:
            if ".npy" in img_path:
                image = Image.fromarray(np.load(img_path))
            else:
                image = Image.open(img_path).convert('RGB')
            return transform(image), img_path
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None
    
    # Process in batches
    for i in tqdm(range(0, len(query_paths), batch_size), desc="Finding similar images"):
        batch_paths = query_paths[i:i + batch_size]
        
        # Parallel load and transform images on CPU
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(load_single_image)(path) for path in batch_paths
        )
        
        # Filter out None results and prepare batch
        batch_images = []
        batch_valid_paths = []
        for img, path in results:
            if img is not None:
                batch_images.append(img)
                batch_valid_paths.append(path)
        
        if not batch_images:
            continue
        
        # Process batch on GPU
        batch_tensor = torch.stack(batch_images).to(DEVICE)
        with torch.no_grad():
            batch_embeddings = model(batch_tensor).cpu().numpy().astype('float32')
        
        # Batch search in FAISS
        D, I = gpu_index.search(batch_embeddings, 1)  # embeddings, nearest neighbor
        
        # Update similarity dictionary
        for query_path, idx in zip(batch_valid_paths, I):
            similarity_dict[query_path] = reference_paths[idx[0]]
    
    return similarity_dict

def load_or_build_database(model, transform, force_rebuild=False):
    """Load existing database or build new one if necessary."""
    if not force_rebuild and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(REFERENCE_PATHS_CACHE):
        print("Loading existing database...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        reference_paths = np.load(REFERENCE_PATHS_CACHE, allow_pickle=True).tolist()
        print(f"Loaded database with {len(reference_paths)} images")
        return index, reference_paths
    
    print("Building new database...")
    index, reference_paths = build_clickme_database(model, transform, rebuild=force_rebuild)
    
    # Save the database
    print("Saving database...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(REFERENCE_PATHS_CACHE, reference_paths)
    print(f"Saved database with {len(reference_paths)} images")
    
    return index, reference_paths

def main():
    # Setup model
    model, transform = setup_model()
    
    # Load or build database
    index, reference_paths = load_or_build_database(model, transform, force_rebuild=FORCE_BUILD)
    
    # Get ImageNet paths
    imagenet_paths = (
        glob.glob(os.path.join(IMAGENET_TRAIN, "**/*.JPEG"), recursive=True) +
        glob.glob(os.path.join(IMAGENET_VAL, "**/*.JPEG"), recursive=True)
    )
    
    # Find similar images
    similarity_dict = find_similar_images(model, transform, index, reference_paths, imagenet_paths)
    
    # Save results
    print("Saving results...")
    np.save('image_similarity_dict.npy', similarity_dict)
    print(f"Processed {len(similarity_dict)} images. Results saved to image_similarity_dict.npy")

if __name__ == "__main__":
    main() 