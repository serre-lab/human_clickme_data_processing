"""
Cython-optimized versions of performance-critical functions for the ClickMe data processing pipeline.

To compile:
python setup.py build_ext --inplace
"""

import numpy as np
from libc.math cimport exp
cimport numpy as np
cimport cython

# Define C types for NumPy arrays
ctypedef np.int_t DTYPE_INT
ctypedef np.float_t DTYPE_FLOAT
ctypedef np.uint8_t DTYPE_UINT8

@cython.boundscheck(False)  # Disable bounds checking
@cython.wraparound(False)   # Disable negative indexing
def create_clickmap_fast(list point_lists, tuple image_shape, bint exponential_decay=False, float tau=0.5):
    """
    Cython-optimized function to create a clickmap from click points.
    This is much faster than the Python version, especially for large datasets.

    Args:
        point_lists (list): List of lists containing (x, y) tuples representing clicks
        image_shape (tuple): Shape of the image (height, width)
        exponential_decay (bool): Whether to apply exponential decay based on click order
        tau (float): Decay rate for exponential decay

    Returns:
        np.ndarray: A 2D array representing the clickmap
    """
    cdef int height = image_shape[0]
    cdef int width = image_shape[1]
    cdef np.ndarray[DTYPE_UINT8, ndim=2] heatmap = np.zeros(image_shape, dtype=np.uint8)
    
    cdef int x, y, idx
    cdef float weight
    cdef list click_points
    
    # Loop through all point lists (click sessions)
    for click_points in point_lists:
        if exponential_decay:
            # Apply exponential decay based on click order
            for idx, point in enumerate(click_points):
                x, y = point
                if 0 <= y < height and 0 <= x < width:
                    weight = exp(-idx / tau)
                    if weight > 1.0:  # Ensure we don't overflow uint8
                        weight = 1.0
                    heatmap[y, x] += <DTYPE_UINT8>(weight)
        else:
            # Simple counter - add 1 for each click
            for point in click_points:
                x, y = point
                if 0 <= y < height and 0 <= x < width:
                    # Check if incrementing would overflow uint8
                    if heatmap[y, x] < 255:
                        heatmap[y, x] += 1
    
    return heatmap


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_duplicate_detection(np.ndarray[DTYPE_FLOAT, ndim=2] clickmaps_vec, float duplicate_thresh):
    """
    Cython-optimized function to detect duplicate clickmaps.
    
    Args:
        clickmaps_vec (np.ndarray): Matrix of flattened clickmaps, shape (n_maps, n_pixels)
        duplicate_thresh (float): Threshold for considering two maps as duplicates
        
    Returns:
        np.ndarray: Indices of non-duplicate maps
    """
    cdef int n_maps = clickmaps_vec.shape[0]
    cdef int n_pixels = clickmaps_vec.shape[1]
    cdef np.ndarray[np.int_t, ndim=1] is_duplicate = np.zeros(n_maps, dtype=np.int)
    cdef int i, j
    cdef float distance, diff
    cdef float sq_duplicate_thresh = duplicate_thresh * duplicate_thresh
    
    # Process pairs of maps (only lower triangular part of the distance matrix)
    for i in range(n_maps):
        if is_duplicate[i]:
            continue
        
        for j in range(i+1, n_maps):
            if is_duplicate[j]:
                continue
                
            # Compute Euclidean distance squared (faster than cdist)
            distance = 0.0
            for k in range(n_pixels):
                diff = clickmaps_vec[i, k] - clickmaps_vec[j, k]
                distance += diff * diff
                
                # Early stopping - if we exceed threshold, no need to continue
                if distance > sq_duplicate_thresh:
                    break
            
            # If distance is below threshold, mark as duplicate
            if distance < sq_duplicate_thresh:
                is_duplicate[j] = 1
    
    # Create array of indices for non-duplicate maps
    cdef np.ndarray[np.int_t, ndim=1] non_duplicate_indices = np.where(is_duplicate == 0)[0]
    return non_duplicate_indices


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_ious_binary(np.ndarray[DTYPE_UINT8, ndim=1] v1, np.ndarray[DTYPE_UINT8, ndim=1] v2):
    """
    Cython-optimized computation of IoU between two binary arrays.
    
    Args:
        v1 (np.ndarray): First binary array
        v2 (np.ndarray): Second binary array
        
    Returns:
        float: IoU score
    """
    cdef int intersection = 0
    cdef int union = 0
    cdef int i
    cdef int n = v1.shape[0]
    
    for i in range(n):
        if v1[i] and v2[i]:
            intersection += 1
            union += 1
        elif v1[i] or v2[i]:
            union += 1
    
    if union == 0:
        return 0.0
    else:
        return intersection / <float>union 