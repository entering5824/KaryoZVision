"""
Utility functions for image preprocessing and segmentation.
Part 1: Preprocessing & Segmentation for Karyotype Image Processing
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure
from typing import Tuple, Optional, List
import json
from datetime import datetime


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.
    
    Args:
        image: Input grayscale image (uint8)
        clip_limit: CLAHE clip limit (default: 2.0)
        tile_size: Grid size for CLAHE (default: (8, 8))
    
    Returns:
        CLAHE-enhanced grayscale image (uint8)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(image)
    return enhanced


def apply_threshold(
    image: np.ndarray,
    method: str = "otsu",
    threshold_value: Optional[int] = None,
    use_inverse: bool = True,
    adaptive: bool = False,
    adaptive_block_size: int = 11,
    adaptive_c: float = 2.0,
    otsu_offset: int = 10
) -> Tuple[np.ndarray, float]:
    """
    Apply thresholding to create binary mask.
    
    For karyotype images:
    - Background = white (high intensity ~255)
    - Chromosomes = dark gray (low intensity ~60-140)
    - Use THRESH_BINARY_INV to get: chromosomes = 1 (white), background = 0 (black)
    
    Args:
        image: Input grayscale image (uint8)
        method: Thresholding method - "otsu", "global", or "adaptive" (default: "otsu")
        threshold_value: Threshold value T for global method (required if method="global")
        use_inverse: Use inverse thresholding (True for dark objects on bright background)
        adaptive: Use adaptive thresholding (fallback if Otsu fails)
        adaptive_block_size: Block size for adaptive threshold (must be odd)
        adaptive_c: Constant subtracted from mean in adaptive threshold
    
    Returns:
        Tuple of (binary_mask, threshold_value_used)
        mask(x,y) = 1 for chromosomes (white), 0 for background (black)
    """
    if method == "otsu":
        # Use THRESH_BINARY_INV because chromosomes are darker than background
        threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if use_inverse else cv2.THRESH_BINARY + cv2.THRESH_OTSU
        otsu_value, _ = cv2.threshold(
            image, 0, 255, threshold_type
        )
        # Apply a slightly stricter threshold to avoid merged chromosomes
        adjusted_thresh = max(0, min(255, otsu_value + otsu_offset))
        _, binary_mask = cv2.threshold(
            image, adjusted_thresh, 255, cv2.THRESH_BINARY_INV if use_inverse else cv2.THRESH_BINARY
        )
        threshold_value_used = float(adjusted_thresh)
    elif method == "global":
        if threshold_value is None:
            raise ValueError("threshold_value must be provided when method='global'")
        threshold_value_used = float(threshold_value)
        threshold_type = cv2.THRESH_BINARY_INV if use_inverse else cv2.THRESH_BINARY
        _, binary_mask = cv2.threshold(
            image, threshold_value, 255, threshold_type
        )
    elif method == "adaptive":
        # Adaptive thresholding as fallback
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1  # Ensure odd number
        binary_mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if use_inverse else cv2.THRESH_BINARY,
            adaptive_block_size,
            adaptive_c
        )
        threshold_value_used = -1.0  # Adaptive doesn't have single threshold value
    else:
        raise ValueError(f"Unknown threshold method: {method}. Use 'otsu', 'global', or 'adaptive'")
    
    # Convert to binary (0 or 1)
    binary_mask = (binary_mask > 0).astype(np.uint8)
    
    return binary_mask, threshold_value_used


def apply_morphology(mask: np.ndarray, kernel_size: int = 5, operations: Optional[List[str]] = None) -> np.ndarray:
    """
    Apply morphological operations to refine binary mask.
    
    Args:
        mask: Binary mask (0 or 1)
        kernel_size: Size of morphological kernel (default: 5)
        operations: List of operations to apply: "opening", "closing", "dilation", "erosion"
                   If None, applies ["opening", "closing"] by default
    
    Returns:
        Refined binary mask
    """
    if operations is None:
        operations = ["opening", "closing"]
    
    # Create elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    result = mask.copy()
    
    for op in operations:
        if op == "opening":
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op == "closing":
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif op == "dilation":
            result = cv2.dilate(result, kernel, iterations=1)
        elif op == "erosion":
            result = cv2.erode(result, kernel, iterations=1)
        else:
            raise ValueError(f"Unknown morphological operation: {op}")
    
    return result


def fill_small_holes(mask: np.ndarray, hole_area_threshold: int = 500) -> np.ndarray:
    """
    Fill only small holes in binary mask, avoiding filling large background regions.
    
    Args:
        mask: Binary mask (0 or 1)
        hole_area_threshold: Maximum area of a hole to fill (pixels, default: 500)
    
    Returns:
        Binary mask with small holes filled
    """
    # Invert mask to find holes (background regions inside objects)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    inverted = cv2.bitwise_not(mask_uint8)
    
    # Find connected components in inverted mask (these are potential holes)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    result = mask_uint8.copy()
    
    # Fill holes that are small enough
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area <= hole_area_threshold:
            # This is a small hole - fill it
            hole_mask = (labels == label_id).astype(np.uint8) * 255
            result = cv2.bitwise_or(result, hole_mask)
    
    return (result > 0).astype(np.uint8)


def fill_holes(mask: np.ndarray, max_blob_area: int = 10000, hole_area_threshold: int = 500) -> np.ndarray:
    """
    Fill holes in binary mask with improved per-blob processing.
    Uses smart hole filling strategy: fills small holes in all blobs, 
    and fills all holes in small blobs.
    
    Args:
        mask: Binary mask (0 or 1)
        max_blob_area: Maximum blob area to fill all holes for (default: 10000)
        hole_area_threshold: Maximum area of individual holes to fill (default: 500)
    
    Returns:
        Binary mask with holes filled selectively
    """
    # Ensure uint8 0/255 for labeling
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    result = np.zeros_like(mask_uint8)
    
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        blob_mask = (labels == label_id).astype(np.uint8) * 255
        
        if area < max_blob_area:
            # Small blob: fill all holes
            filled = ndimage.binary_fill_holes((blob_mask > 0).astype(bool)).astype(np.uint8) * 255
        else:
            # Large blob: only fill small holes
            filled = fill_small_holes((blob_mask > 0).astype(np.uint8), hole_area_threshold) * 255
        
        result = cv2.bitwise_or(result, filled)
    
    return (result > 0).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Remove small connected components from binary mask.
    
    Args:
        mask: Binary mask (0 or 1)
        min_area: Minimum area (in pixels) to keep a component (default: 100)
    
    Returns:
        Binary mask with small components removed
    """
    # Label connected components
    labeled_mask = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    
    # Create output mask
    cleaned_mask = np.zeros_like(mask)
    
    # Keep only components with area >= min_area
    for region in regions:
        if region.area >= min_area:
            cleaned_mask[labeled_mask == region.label] = 1
    
    return cleaned_mask


def smooth_edges(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Smooth edges of binary mask using morphological operations.
    
    Args:
        mask: Binary mask (0 or 1)
        kernel_size: Size of morphological kernel for smoothing (default: 3)
        iterations: Number of times to apply smoothing (default: 1)
    
    Returns:
        Binary mask with smoothed edges
    """
    # Ensure uint8 0/255
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Create small elliptical kernel for gentle smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    result = mask_uint8.copy()
    
    # Apply gentle closing to smooth edges
    for _ in range(iterations):
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        # Optional: light opening to remove small protrusions
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    return (result > 0).astype(np.uint8)


def reduce_fragmentation(mask: np.ndarray, merge_distance: int = 15, kernel_size: int = 5) -> np.ndarray:
    """
    Reduce fragmentation by merging nearby fragments using morphological operations.
    
    Args:
        mask: Binary mask (0 or 1)
        merge_distance: Maximum distance between fragments to merge (pixels, default: 15)
        kernel_size: Kernel size for morphological closing (default: 5)
    
    Returns:
        Binary mask with reduced fragmentation
    """
    # Ensure uint8 0/255
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Use morphological closing to merge nearby fragments
    # Kernel size should be roughly 2 * merge_distance
    effective_kernel_size = max(3, min(kernel_size, merge_distance * 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (effective_kernel_size, effective_kernel_size))
    
    # Apply closing to merge fragments
    result = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Optional: light opening to remove small artifacts created by closing
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, small_kernel, iterations=1)
    
    return (result > 0).astype(np.uint8)


def remove_border_artifacts(mask: np.ndarray, border_size: int = 50) -> np.ndarray:
    """
    Remove artifacts near image borders (tray edges, labels).
    
    Args:
        mask: Binary mask (0 or 1)
        border_size: Border width to clear (pixels)
    
    Returns:
        Binary mask with borders cleared
    """
    result = mask.copy()
    h, w = result.shape
    result[:border_size, :] = 0
    result[h - border_size:, :] = 0
    result[:, :border_size] = 0
    result[:, w - border_size:] = 0
    return result


def apply_row_specific_morphology(mask: np.ndarray) -> np.ndarray:
    """
    Apply row-specific morphology tuned for karyotype rows with improved hole filling and edge smoothing.
    
    Row boundaries (Y):
      Row 1: Y < 344
      Row 2: 344 ≤ Y < 500
      Row 3: 500 ≤ Y < 626
      Row 4: Y ≥ 626
    """
    # Work on uint8 0/255
    cleaned = (mask > 0).astype(np.uint8) * 255
    cleaned = remove_border_artifacts(cleaned, border_size=50)

    row_boundaries = [
        (120, 344),   # Row 1
        (344, 500),   # Row 2
        (500, 626),   # Row 3
        (626, 760)    # Row 4
    ]

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    result = cleaned.copy()

    # Row 1
    y1, y2 = row_boundaries[0]
    row = cleaned[y1:y2, :]
    row_closed = cv2.morphologyEx(row, cv2.MORPH_CLOSE, k3, iterations=3)  # Increased iterations
    row_binary = (row_closed > 0).astype(np.uint8)
    row_filled = fill_small_holes(row_binary, hole_area_threshold=500) * 255
    contours, _ = cv2.findContours(row_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_contour = row_filled.copy()
    cv2.drawContours(row_contour, contours, -1, 255, cv2.FILLED)
    row_opened = cv2.morphologyEx(row_contour, cv2.MORPH_OPEN, k3, iterations=2)
    row_smoothed = smooth_edges((row_opened > 0).astype(np.uint8), kernel_size=3, iterations=1)
    result[y1:y2, :] = row_smoothed * 255

    # Row 2
    y1, y2 = row_boundaries[1]
    row = cleaned[y1:y2, :]
    row_closed = cv2.morphologyEx(row, cv2.MORPH_CLOSE, k4, iterations=4)  # Increased iterations
    row_binary = (row_closed > 0).astype(np.uint8)
    row_filled = fill_small_holes(row_binary, hole_area_threshold=500) * 255
    contours, _ = cv2.findContours(row_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_contour = row_filled.copy()
    cv2.drawContours(row_contour, contours, -1, 255, cv2.FILLED)
    row_opened = cv2.morphologyEx(row_contour, cv2.MORPH_OPEN, k3, iterations=2)
    row_smoothed = smooth_edges((row_opened > 0).astype(np.uint8), kernel_size=3, iterations=1)
    result[y1:y2, :] = row_smoothed * 255

    # Row 3
    y1, y2 = row_boundaries[2]
    row = cleaned[y1:y2, :]
    row_closed = cv2.morphologyEx(row, cv2.MORPH_CLOSE, k4, iterations=3)  # Increased iterations
    row_binary = (row_closed > 0).astype(np.uint8)
    row_filled = fill_small_holes(row_binary, hole_area_threshold=500) * 255
    row_opened = cv2.morphologyEx(row_filled, cv2.MORPH_OPEN, k3, iterations=2)
    row_smoothed = smooth_edges((row_opened > 0).astype(np.uint8), kernel_size=3, iterations=1)
    result[y1:y2, :] = row_smoothed * 255

    # Row 4: apply basic processing
    y1, y2 = row_boundaries[3]
    if y2 > y1:
        row = cleaned[y1:y2, :]
        row_closed = cv2.morphologyEx(row, cv2.MORPH_CLOSE, k3, iterations=2)
        row_binary = (row_closed > 0).astype(np.uint8)
        row_filled = fill_small_holes(row_binary, hole_area_threshold=500) * 255
        row_smoothed = smooth_edges((row_filled > 0).astype(np.uint8), kernel_size=3, iterations=1)
        result[y1:y2, :] = row_smoothed * 255

    # Global flood fill to close remaining holes touching borders
    filled = result.copy()
    h, w = filled.shape
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, mask_ff, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    result = cv2.bitwise_or(result, filled_inv)

    return (result > 0).astype(np.uint8)


def save_metadata(output_path: str, metadata_dict: dict) -> None:
    """
    Save processing metadata to JSON file.
    
    Args:
        output_path: Path to output JSON file
        metadata_dict: Dictionary containing metadata
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

