"""
Pure Python implementations of the same functions as cython_utils.
These are used as fallback when Cython compilation fails.
"""

import numpy as np
from scipy.spatial.distance import cdist

def create_clickmap_fast(point_lists, image_shape, exponential_decay=False, tau=0.5):
    """
    Python implementation of the clickmap creation function.
    
    Args:
        point_lists (list): List of lists containing (x, y) tuples representing clicks
        image_shape (tuple): Shape of the image (height, width)
        exponential_decay (bool): Whether to apply exponential decay based on click order
        tau (float): Decay rate for exponential decay
    
    Returns:
        np.ndarray: A 2D array representing the clickmap
    """
    height, width = image_shape
    heatmap = np.zeros(image_shape, dtype=np.uint8)
    
    for click_points in point_lists:
        if exponential_decay:
            for idx, point in enumerate(click_points):
                x, y = point
                if 0 <= y < height and 0 <= x < width:
                    weight = np.exp(-idx / tau)
                    if weight > 1.0:
                        weight = 1.0
                    heatmap[y, x] += int(weight)
        else:
            for point in click_points:
                x, y = point
                if 0 <= y < height and 0 <= x < width:
                    if heatmap[y, x] < 255:
                        heatmap[y, x] += 1
    
    return heatmap


def fast_duplicate_detection(clickmaps_vec, duplicate_thresh):
    """
    Python implementation of the duplicate detection function.
    
    Args:
        clickmaps_vec (np.ndarray): Matrix of flattened clickmaps, shape (n_maps, n_pixels)
        duplicate_thresh (float): Threshold for considering two maps as duplicates
    
    Returns:
        np.ndarray: Indices of non-duplicate maps
    """
    n_maps = clickmaps_vec.shape[0]
    is_duplicate = np.zeros(n_maps, dtype=np.int32)
    
    for i in range(n_maps):
        if is_duplicate[i]:
            continue
        
        for j in range(i+1, n_maps):
            if is_duplicate[j]:
                continue
            
            # Compute distance for this pair
            diff = clickmaps_vec[i] - clickmaps_vec[j]
            distance = np.sum(diff * diff)
            
            # If distance is below threshold, mark as duplicate
            if distance < duplicate_thresh * duplicate_thresh:
                is_duplicate[j] = 1
    
    # Return indices of non-duplicate maps
    return np.where(is_duplicate == 0)[0]


def fast_ious_binary(v1, v2):
    """
    Python implementation of the IoU computation function.
    
    Args:
        v1 (np.ndarray): First binary array
        v2 (np.ndarray): Second binary array
    
    Returns:
        float: IoU score
    """
    intersection = np.logical_and(v1, v2).sum()
    union = np.logical_or(v1, v2).sum()
    
    if union == 0:
        return 0.0
    else:
        return intersection / float(union) 