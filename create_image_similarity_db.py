import os
import glob
import numpy as np
import torch
import timm
import faiss
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

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

def get_embedding(model, transform, image_path):
    """Get embedding for a single image."""
    import pdb; pdb.set_trace()
    try:
        if ".npy" in image_path:
            image = np.load(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(image).cpu().numpy()
        return embedding.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def build_clickme_database(model, transform):
    """Build FAISS database from ClickMe images."""
    embeddings = []
    image_paths = []
    
    # Process all ClickMe images
    for path in CLICKME_PATHS:
        import pdb; pdb.set_trace()
        for img_file in tqdm(glob.glob(os.path.join(path, "*.npy")), desc=f"Processing {path}"):
            embedding = get_embedding(model, transform, img_file)
            if embedding is not None:
                embeddings.append(embedding)
                image_paths.append(img_file)
    
    # Create FAISS index
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, image_paths

def find_similar_images(model, transform, index, reference_paths, query_paths):
    """Find similar images between query images and reference database."""
    similarity_dict = {}
    
    for img_path in tqdm(query_paths, desc="Finding similar images"):
        embedding = get_embedding(model, transform, img_path)
        if embedding is not None:
            # Search in the index
            D, I = index.search(embedding.reshape(1, -1), 1)
            similar_image = reference_paths[I[0][0]]
            similarity_dict[img_path] = similar_image
    
    return similarity_dict

def load_or_build_database(model, transform, force_build=False):
    """Load existing database or build new one if necessary."""
    if not force_build and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(REFERENCE_PATHS_CACHE):
        print("Loading existing database...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        reference_paths = np.load(REFERENCE_PATHS_CACHE, allow_pickle=True).tolist()
        print(f"Loaded database with {len(reference_paths)} images")
        return index, reference_paths
    
    print("Building new database...")
    index, reference_paths = build_clickme_database(model, transform)
    
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
    index, reference_paths = load_or_build_database(model, transform, force_build=FORCE_BUILD)
    
    # Get ImageNet paths
    imagenet_paths = (
        glob.glob(os.path.join(IMAGENET_TRAIN, "**/*.JPEG"), recursive=True) +
        glob.glob(os.path.join(IMAGENET_VAL, "**/*.JPEG"), recursive=True)
    )
    
    # Find similar images
    print("Finding similar images...")
    similarity_dict = find_similar_images(model, transform, index, reference_paths, imagenet_paths)
    
    # Save results
    print("Saving results...")
    np.save('image_similarity_dict.npy', similarity_dict)
    print(f"Processed {len(similarity_dict)} images. Results saved to image_similarity_dict.npy")

if __name__ == "__main__":
    main() 