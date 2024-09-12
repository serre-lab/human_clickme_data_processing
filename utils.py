import torch
import numpy as np


def create_clickmap(point_lists, image_shape):
    """
    Create a clickmap from click points.

    Args:
        click_points (list of tuples): List of (x, y) coordinates where clicks occurred.
        image_shape (tuple): Shape of the image (height, width).
        blur_kernel (torch.Tensor, optional): Gaussian kernel for blurring. Default is None.

    Returns:
        np.ndarray: A 2D array representing the clickmap, blurred if kernel provided.
    """
    heatmap = np.zeros(image_shape, dtype=np.uint8)
    for click_points in point_lists:
        for point in click_points:
            if 0 <= point[1] < image_shape[0] and 0 <= point[0] < image_shape[1]:
                heatmap[point[1], point[0]] += 1
    return heatmap


def gaussian_kernel(size=10, sigma=10):
    """
    Generates a 2D Gaussian kernel.

    Parameters
    ----------
    size : int, optional
        Kernel size, by default 10
    sigma : int, optional
        Kernel sigma, by default 10

    Returns
    -------
    kernel : torch.Tensor
        A Gaussian kernel.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return kernel

def gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.

    Parameters
    ----------
    heatmap : torch.Tensor
        The heatmap to blur.
    kernel : torch.Tensor
        The Gaussian kernel.

    Returns
    -------
    blurred_heatmap : torch.Tensor
        The blurred heatmap.
    """
    # Ensure heatmap and kernel have the correct dimensions
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = torch.nn.functional.conv2d(heatmap, kernel, padding='same')

    return blurred_heatmap[0]