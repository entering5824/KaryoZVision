#!/usr/bin/env python3
"""
Part 2: Blob Extraction & Feature Calculation

Extracts connected components (blobs) from binary masks and calculates features:
- Area, Centroid, Bounding Box
- Optional: Aspect ratio, Eccentricity

Reads masks from Part 1 output and generates blob features.
"""

import argparse
import json
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np
from .blob_processing_utils import (
    apply_nms,
    split_large_blobs,
    filter_blobs_by_aspect_ratio,
    filter_blobs_by_eccentricity,
    filter_blobs_by_area,
    merge_close_blobs,
    smart_split_rows,
    fill_blob_holes,
    refine_mask_edges,
    ROW_BOUNDARIES,
    EXPECTED_BLOBS_PER_ROW
)


def read_mask(mask_path: str) -> np.ndarray:
    """
    Read binary mask from file.
    
    Args:
        mask_path: Path to mask image
    
    Returns:
        Binary mask (0 and 255) as uint8
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Could not read mask image: {mask_path}')
    
    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Ensure binary: threshold at 127
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Ensure output is uint8 (0 or 255)
    mask = mask.astype(np.uint8)
    
    return mask


def compute_eccentricity(component_mask: np.ndarray) -> float:
    """
    Compute eccentricity of a blob using ellipse fitting or moments.
    
    Args:
        component_mask: Binary mask of single component (0/255)
    
    Returns:
        Eccentricity value (0.0 to 1.0)
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    
    if len(cnt) < 5:
        # Not enough points to fit ellipse; use moments
        M = cv2.moments(cnt)
        if M['mu20'] + M['mu02'] == 0:
            return 0.0
        # Covariance matrix elements
        a = M['mu20'] / M['m00']
        b = M['mu11'] / M['m00']
        c = M['mu02'] / M['m00']
        # Eigenvalues of [[a, b],[b, c]]
        trace = a + c
        det = a * c - b * b
        discriminant = trace * trace - 4 * det
        if discriminant <= 0:
            return 0.0
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2
        if lambda1 <= 0:
            return 0.0
        # Eccentricity = sqrt(1 - lambda2/lambda1)
        return float(np.sqrt(max(0.0, 1.0 - (lambda2 / lambda1))))
    else:
        # Fit ellipse
        ellipse = cv2.fitEllipse(cnt)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        if major_axis == 0:
            return 0.0
        e = np.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2))
        return float(e)


def sort_blobs_by_position(blobs: List[Dict[str, Any]], row_tol: float = 25.0) -> List[Dict[str, Any]]:
    """
    Sort blobs first by row (top to bottom) then by column (left to right).
    
    This ensures IDs reflect the physical order (row → column) on the karyotype image,
    preventing pairs from being reversed (e.g., 1↔2, 3↔4).
    
    Args:
        blobs: List of blob dictionaries with 'centroid' key
        row_tol: Maximum y-distance (pixels) to consider blobs in the same row
    
    Returns:
        Sorted list of blobs
    """
    if not blobs:
        return blobs
    
    # Sort by y-coordinate of centroid
    blobs = sorted(blobs, key=lambda b: b['centroid'][1])
    
    sorted_blobs = []
    current_row = []
    last_y = None
    
    for b in blobs:
        cy = b['centroid'][1]
        if last_y is None or abs(cy - last_y) <= row_tol:
            current_row.append(b)
        else:
            # Sort previous row by x-coordinate
            current_row = sorted(current_row, key=lambda b: b['centroid'][0])
            sorted_blobs.extend(current_row)
            current_row = [b]
        last_y = cy
    
    # Sort and add last row
    if current_row:
        current_row = sorted(current_row, key=lambda b: b['centroid'][0])
        sorted_blobs.extend(current_row)
    
    return sorted_blobs


def extract_blobs(
    mask: np.ndarray,
    min_area: int = 50,
    connectivity: int = 8,
    include_optional: bool = True,
    apply_nms_flag: bool = True,
    nms_iou_threshold: float = 0.4,
    split_large_blobs_flag: bool = True,
    area_threshold_ratio: float = 1.5,
    filter_aspect_ratio: bool = True,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0,
    filter_eccentricity: bool = False,
    min_eccentricity: float = 0.3,
    max_eccentricity: float = 0.99,
    use_row_thresholds: bool = True,
    use_row_splitting: bool = True,
    refine_blob_masks: bool = True,
    fill_blob_holes_flag: bool = True,
    hole_area_threshold: int = 500,
    edge_smoothing_kernel: int = 3,
    edge_smoothing_iterations: int = 1
) -> List[Dict[str, Any]]:
    """
    Extract blobs from binary mask using connected components with improved filtering.
    
    Args:
        mask: Binary mask (0 and 255)
        min_area: Minimum area to keep a blob (pixels)
        connectivity: Connectivity for connected components (4 or 8)
        include_optional: Whether to include aspect_ratio and eccentricity
        apply_nms_flag: Whether to apply Non-Maximum Suppression (default: True)
        nms_iou_threshold: IoU threshold for NMS (default: 0.4)
        split_large_blobs_flag: Whether to split blobs larger than average (default: True)
        area_threshold_ratio: Ratio threshold for splitting (default: 1.5 = 150% of average)
        filter_aspect_ratio: Whether to filter by aspect ratio (default: True)
        min_aspect_ratio: Minimum aspect ratio (width/height)
        max_aspect_ratio: Maximum aspect ratio (width/height)
        filter_eccentricity: Whether to filter by eccentricity (default: False)
        min_eccentricity: Minimum eccentricity
        max_eccentricity: Maximum eccentricity
    
    Returns:
        List of blob dictionaries with features
    """
    # Ensure mask is uint8 with 0 and 255 values (not 0/1)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Normalize to 0/255 if mask is 0/1
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # connectedComponentsWithStats expects 0 and non-zero (255)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    
    blobs = []
    # Row-specific min area thresholds (Y-based)
    row_min_areas = [
        (ROW_BOUNDARIES[0], 800),
        (ROW_BOUNDARIES[1], 600),
        (ROW_BOUNDARIES[2], 400),
        (float("inf"), 300)
    ]
    
    for i in range(1, num_labels):  # Skip background label 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        cy = float(centroids[i, 1])

        # Determine row-specific area threshold
        min_area_row = min_area
        if use_row_thresholds:
            for boundary, min_a in row_min_areas:
                if cy < boundary:
                    min_area_row = max(min_area, min_a)
                    break

        if area < min_area_row:
            continue
        
        left = int(stats[i, cv2.CC_STAT_LEFT])
        top = int(stats[i, cv2.CC_STAT_TOP])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        
        # Create blob dictionary with required features
        # Note: Using 'bbox' to match JSON sample format in requirements
        blob = {
            'id': int(i),
            'original_label': int(i),  # Store original connected component label index for mask extraction
            'area': area,
            'centroid': [float(cx), float(cy)],
            'bbox': [left, top, width, height]
        }
        
        # Refine mask for this blob before computing features
        comp_mask = (labels == i).astype('uint8') * 255
        
        if refine_blob_masks:
            # Extract blob region with padding to avoid edge issues
            pad = 5
            x1 = max(0, left - pad)
            y1 = max(0, top - pad)
            x2 = min(mask.shape[1], left + width + pad)
            y2 = min(mask.shape[0], top + height + pad)
            
            # Extract blob region
            blob_region = comp_mask[y1:y2, x1:x2]
            
            # Fill holes in blob mask
            if fill_blob_holes_flag:
                blob_region = fill_blob_holes(blob_region, hole_area_threshold=hole_area_threshold)
            
            # Smooth edges of blob mask
            blob_region = refine_mask_edges(blob_region, kernel_size=edge_smoothing_kernel, iterations=edge_smoothing_iterations)
            
            # Recompute stats after refinement
            num_refined, labels_refined, stats_refined, centroids_refined = cv2.connectedComponentsWithStats(
                blob_region, connectivity=connectivity
            )
            
            if num_refined > 1:
                # Use the largest component (should be the refined blob)
                largest_idx = 1
                if num_refined > 2:
                    areas = [stats_refined[j, cv2.CC_STAT_AREA] for j in range(1, num_refined)]
                    largest_idx = 1 + np.argmax(areas)
                
                # Update blob stats from refined mask (adjust coordinates back to full image)
                area = int(stats_refined[largest_idx, cv2.CC_STAT_AREA])
                left_refined = int(stats_refined[largest_idx, cv2.CC_STAT_LEFT]) + x1
                top_refined = int(stats_refined[largest_idx, cv2.CC_STAT_TOP]) + y1
                width_refined = int(stats_refined[largest_idx, cv2.CC_STAT_WIDTH])
                height_refined = int(stats_refined[largest_idx, cv2.CC_STAT_HEIGHT])
                cx_refined = float(centroids_refined[largest_idx, 0]) + x1
                cy_refined = float(centroids_refined[largest_idx, 1]) + y1
                
                blob['area'] = area
                blob['bbox'] = [left_refined, top_refined, width_refined, height_refined]
                blob['centroid'] = [float(cx_refined), float(cy_refined)]
                
                # Update comp_mask for feature computation
                refined_full = np.zeros_like(mask)
                refined_region = (labels_refined == largest_idx).astype('uint8') * 255
                refined_full[y1:y2, x1:x2] = refined_region
                comp_mask = refined_full
                
                # Update local variables for optional features
                left, top, width, height = left_refined, top_refined, width_refined, height_refined
                cx, cy = cx_refined, cy_refined
        
        # Add optional features if requested
        if include_optional:
            eccentricity = compute_eccentricity(comp_mask)
            aspect_ratio = float(width) / float(height) if height != 0 else 0.0
            
            blob['aspect_ratio'] = aspect_ratio
            blob['eccentricity'] = eccentricity
            
            # Sanity check: ensure centroid is inside bbox
            if not (left <= cx <= left + width and top <= cy <= top + height):
                # Recompute centroid from contour moments
                cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    M = cv2.moments(cnts[0])
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        blob['centroid'] = [float(cx), float(cy)]
        
        blobs.append(blob)
    
    # Post-processing: Merge close blobs FIRST (fix over-segmentation before splitting)
    # This helps reduce over-segmentation by merging nearby blobs early
    if len(blobs) > 1:
        blobs = merge_close_blobs(blobs, distance_threshold=25.0, max_area_ratio=3.0)
    
    # Post-processing: Filter by area (remove outliers - too large or too small)
    if len(blobs) > 0:
        blobs = filter_blobs_by_area(blobs, min_area, max_area_ratio=3.0)
    
    # Post-processing: Split large blobs (fix merged objects) - only if still too large after merging
    if split_large_blobs_flag and len(blobs) > 0:
        blobs = split_large_blobs(blobs, mask, labels, area_threshold_ratio, min_area)
    
    # Post-processing: Merge close blobs again after splitting (in case splitting created fragments)
    if len(blobs) > 1:
        blobs = merge_close_blobs(blobs, distance_threshold=20.0, max_area_ratio=2.5)
    
    # Post-processing: Filter by aspect ratio
    if filter_aspect_ratio:
        blobs = filter_blobs_by_aspect_ratio(blobs, min_aspect_ratio, max_aspect_ratio)
    
    # Post-processing: Filter by eccentricity
    if filter_eccentricity and include_optional:
        blobs = filter_blobs_by_eccentricity(blobs, min_eccentricity, max_eccentricity)
    
    # Post-processing: Row-aware smart splitting to reach expected counts
    if use_row_splitting and len(blobs) > 0:
        blobs = smart_split_rows(blobs)
        if include_optional:
            # Recompute lightweight optional features for split blobs
            for b in blobs:
                left, top, width, height = b["bbox"]
                b["aspect_ratio"] = float(width) / float(height) if height != 0 else 0.0

    # Post-processing: Apply NMS to remove duplicates (improved)
    if apply_nms_flag:
        blobs = apply_nms(blobs, nms_iou_threshold)
    
    # --- Sort blobs by position (row → column) before reassigning IDs ---
    # This ensures IDs reflect the physical order on the karyotype image
    blobs = sort_blobs_by_position(blobs, row_tol=25.0)
    
    # Reassign IDs sequentially
    for idx, blob in enumerate(blobs, start=1):
        blob['id'] = idx
    
    return blobs


def save_blobs_json(blobs: List[Dict[str, Any]], image_name: str, output_path: str) -> None:
    """
    Save blobs to JSON file with format: {"image": "...", "blobs": [...]}
    
    Args:
        blobs: List of blob dictionaries
        image_name: Name of input image
        output_path: Path to output JSON file
    """
    data = {
        "image": image_name,
        "blobs": blobs
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_blobs_csv(blobs: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save blobs to CSV file with flattened structure.
    
    Args:
        blobs: List of blob dictionaries
        output_path: Path to output CSV file
    """
    if not blobs:
        # Write header only
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'area', 'centroid_x', 'centroid_y',
                'bbox_left', 'bbox_top', 'bbox_w', 'bbox_h',
                'aspect_ratio', 'eccentricity'
            ])
        return
    
    rows = []
    for b in blobs:
        row = [
            b['id'],
            b['area'],
            b['centroid'][0],
            b['centroid'][1],
            b['bbox'][0],
            b['bbox'][1],
            b['bbox'][2],
            b['bbox'][3]
        ]
        # Add optional features if present
        if 'aspect_ratio' in b:
            row.append(b['aspect_ratio'])
        else:
            row.append(None)
        if 'eccentricity' in b:
            row.append(b['eccentricity'])
        else:
            row.append(None)
        rows.append(row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id', 'area', 'centroid_x', 'centroid_y',
            'bbox_left', 'bbox_top', 'bbox_w', 'bbox_h',
            'aspect_ratio', 'eccentricity'
        ])
        writer.writerows(rows)


def create_overlay(mask: np.ndarray, blobs: List[Dict[str, Any]], output_path: str) -> None:
    """
    Create overlay image with bounding boxes, centroids, and blob IDs.
    
    Args:
        mask: Binary mask
        blobs: List of blob dictionaries
        output_path: Path to output overlay image
    """
    # Convert mask to BGR for colored overlay
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    for b in blobs:
        left, top, width, height = b['bbox']
        cx, cy = int(round(b['centroid'][0])), int(round(b['centroid'][1]))
        
        # Draw bounding box (green)
        cv2.rectangle(overlay, (left, top), (left + width, top + height), (0, 255, 0), 2)
        
        # Draw centroid (red circle)
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)
        
        # Draw blob ID (blue text)
        cv2.putText(
            overlay,
            str(b['id']),
            (left, top - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    cv2.imwrite(output_path, overlay)


def get_mask_files(input_path: str) -> List[str]:
    """
    Get list of mask files from input path (file, directory, or masks subdirectory).
    
    Args:
        input_path: Path to mask file, directory, or Part 1 output directory
    
    Returns:
        List of mask file paths
    """
    path = Path(input_path)
    
    if path.is_file():
        # Single mask file
        return [str(path)]
    elif path.is_dir():
        # Check if it's a Part 1 output directory (has masks/ subdirectory)
        masks_dir = path / "masks"
        if masks_dir.exists() and masks_dir.is_dir():
            # Look for *_mask.png files in masks subdirectory
            mask_files = list(masks_dir.glob("*_mask.png"))
            if mask_files:
                return sorted([str(f) for f in mask_files])
        
        # Otherwise, look for mask files directly in directory
        mask_files = list(path.glob("*_mask.png"))
        if mask_files:
            return sorted([str(f) for f in mask_files])
        
        # If no mask files found, return empty list
        return []
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def process_mask_file(
    mask_path: str,
    output_dir: str,
    min_area: int = 50,
    connectivity: int = 8,
    include_optional: bool = True,
    save_csv: bool = True,
    save_overlay: bool = True,
    apply_nms_flag: bool = True,
    nms_iou_threshold: float = 0.4,
    split_large_blobs_flag: bool = True,
    area_threshold_ratio: float = 1.5,
    filter_aspect_ratio: bool = True,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0,
    filter_eccentricity: bool = False,
    min_eccentricity: float = 0.3,
    max_eccentricity: float = 0.99,
    use_row_thresholds: bool = True,
    use_row_splitting: bool = True,
    refine_blob_masks: bool = True,
    fill_blob_holes_flag: bool = True,
    hole_area_threshold: int = 500,
    edge_smoothing_kernel: int = 3,
    edge_smoothing_iterations: int = 1
) -> bool:
    """
    Process a single mask file and extract blobs with improved filtering.
    
    Args:
        mask_path: Path to mask image
        output_dir: Output directory
        min_area: Minimum area to keep a blob
        connectivity: Connectivity for connected components
        include_optional: Include aspect_ratio and eccentricity
        save_csv: Save CSV file
        save_overlay: Save overlay image
        apply_nms_flag: Whether to apply NMS (default: True)
        nms_iou_threshold: IoU threshold for NMS (default: 0.4)
        split_large_blobs_flag: Whether to split large blobs (default: True)
        area_threshold_ratio: Ratio threshold for splitting (default: 1.5)
        filter_aspect_ratio: Whether to filter by aspect ratio (default: True)
        min_aspect_ratio: Minimum aspect ratio
        max_aspect_ratio: Maximum aspect ratio
        filter_eccentricity: Whether to filter by eccentricity (default: False)
        min_eccentricity: Minimum eccentricity
        max_eccentricity: Maximum eccentricity
        use_row_thresholds: Enable row-based min_area thresholds
        use_row_splitting: Enable row-aware smart splitting
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read mask
        mask = read_mask(mask_path)
        
        # Get base name from mask file (remove _mask.png suffix)
        base_name = Path(mask_path).stem.replace('_mask', '')
        
        # Extract blobs with improved filtering
        blobs = extract_blobs(
            mask,
            min_area=min_area,
            connectivity=connectivity,
            include_optional=include_optional,
            apply_nms_flag=apply_nms_flag,
            nms_iou_threshold=nms_iou_threshold,
            split_large_blobs_flag=split_large_blobs_flag,
            area_threshold_ratio=area_threshold_ratio,
            filter_aspect_ratio=filter_aspect_ratio,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            filter_eccentricity=filter_eccentricity,
            min_eccentricity=min_eccentricity,
            max_eccentricity=max_eccentricity,
            use_row_thresholds=use_row_thresholds,
            use_row_splitting=use_row_splitting,
            refine_blob_masks=refine_blob_masks,
            fill_blob_holes_flag=fill_blob_holes_flag,
            hole_area_threshold=hole_area_threshold,
            edge_smoothing_kernel=edge_smoothing_kernel,
            edge_smoothing_iterations=edge_smoothing_iterations
        )
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON (required format with image name)
        json_path = output_path / f"{base_name}_blobs.json"
        save_blobs_json(blobs, base_name, str(json_path))
        
        # Save CSV (optional)
        if save_csv:
            csv_path = output_path / f"{base_name}_blobs.csv"
            save_blobs_csv(blobs, str(csv_path))
        
        # Save overlay (required)
        if save_overlay:
            overlay_path = output_path / f"{base_name}_overlay.png"
            create_overlay(mask, blobs, str(overlay_path))
        
        print(f"Processed: {mask_path} -> Found {len(blobs)} blobs")
        return True
        
    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Part 2: Blob Extraction & Feature Calculation for Karyotype Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process masks from Part 1 output directory
  python code/part2_blobs.py --input out/ --out out/
  
  # Process single mask file
  python code/part2_blobs.py --input out/masks/image_mask.png --out out/
  
  # Custom parameters
  python code/part2_blobs.py --input out/ --out out/ --min-area 100 --connectivity 4
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to mask file, directory with masks, or Part 1 output directory (with masks/ subdirectory)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for blob files"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum area (pixels) to keep a blob (default: 50)"
    )
    
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Connectivity for connected components: 4 or 8 (default: 8)"
    )
    
    parser.add_argument(
        "--no-optional",
        dest="include_optional",
        action="store_false",
        default=True,
        help="Exclude optional features (aspect_ratio, eccentricity)"
    )
    
    parser.add_argument(
        "--no-csv",
        dest="save_csv",
        action="store_false",
        default=True,
        help="Do not save CSV file"
    )
    
    parser.add_argument(
        "--no-overlay",
        dest="save_overlay",
        action="store_false",
        default=True,
        help="Do not save overlay image"
    )
    
    # NMS options
    parser.add_argument(
        "--no-nms",
        action="store_true",
        help="Disable Non-Maximum Suppression (NMS)"
    )
    
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.4,
        help="IoU threshold for NMS (default: 0.4)"
    )
    
    # Blob splitting options
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable splitting of large blobs"
    )
    
    parser.add_argument(
        "--split-threshold",
        type=float,
        default=1.5,
        help="Area ratio threshold for splitting blobs (default: 1.5 = 150%% of average)"
    )
    
    # Filtering options
    parser.add_argument(
        "--no-aspect-filter",
        action="store_true",
        help="Disable aspect ratio filtering"
    )
    
    parser.add_argument(
        "--min-aspect",
        type=float,
        default=0.1,
        help="Minimum aspect ratio (width/height) (default: 0.1)"
    )
    
    parser.add_argument(
        "--max-aspect",
        type=float,
        default=10.0,
        help="Maximum aspect ratio (width/height) (default: 10.0)"
    )
    
    parser.add_argument(
        "--filter-eccentricity",
        action="store_true",
        help="Enable eccentricity filtering"
    )
    
    parser.add_argument(
        "--min-eccentricity",
        type=float,
        default=0.3,
        help="Minimum eccentricity (default: 0.3)"
    )
    
    parser.add_argument(
        "--max-eccentricity",
        type=float,
        default=0.99,
        help="Maximum eccentricity (default: 0.99)"
    )
    parser.add_argument(
        "--no-row-thresholds",
        action="store_true",
        help="Disable row-specific min_area thresholds"
    )
    parser.add_argument(
        "--no-row-splitting",
        action="store_true",
        help="Disable row-aware smart splitting"
    )
    parser.add_argument(
        "--no-refine-blob-masks",
        dest="refine_blob_masks",
        action="store_false",
        default=True,
        help="Disable blob mask refinement (hole filling and edge smoothing)"
    )
    parser.add_argument(
        "--no-fill-blob-holes",
        dest="fill_blob_holes",
        action="store_false",
        default=True,
        help="Disable filling holes in blob masks"
    )
    parser.add_argument(
        "--blob-hole-threshold",
        type=int,
        default=500,
        help="Maximum area of holes to fill in blob masks (pixels, default: 500)"
    )
    parser.add_argument(
        "--blob-edge-kernel",
        type=int,
        default=3,
        help="Kernel size for blob edge smoothing (default: 3)"
    )
    parser.add_argument(
        "--blob-edge-iterations",
        type=int,
        default=1,
        help="Number of iterations for blob edge smoothing (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Get list of mask files
    try:
        mask_files = get_mask_files(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not mask_files:
        print(f"Warning: No mask files found in {args.input}", file=sys.stderr)
        print("Looking for files matching pattern: *_mask.png", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(mask_files)} mask file(s) to process")
    print(f"Output directory: {args.out}")
    print(f"Min area: {args.min_area}")
    print(f"Connectivity: {args.connectivity}")
    print()
    
    # Process each mask file
    success_count = 0
    for mask_path in mask_files:
        success = process_mask_file(
            mask_path,
            args.out,
            min_area=args.min_area,
            connectivity=args.connectivity,
            include_optional=args.include_optional,
            save_csv=args.save_csv,
            save_overlay=args.save_overlay,
            apply_nms_flag=not args.no_nms,
            nms_iou_threshold=args.nms_iou,
            split_large_blobs_flag=not args.no_split,
            area_threshold_ratio=args.split_threshold,
            filter_aspect_ratio=not args.no_aspect_filter,
            min_aspect_ratio=args.min_aspect,
            max_aspect_ratio=args.max_aspect,
            filter_eccentricity=args.filter_eccentricity,
            min_eccentricity=args.min_eccentricity,
            max_eccentricity=args.max_eccentricity,
            use_row_thresholds=not args.no_row_thresholds,
            use_row_splitting=not args.no_row_splitting,
            refine_blob_masks=args.refine_blob_masks,
            fill_blob_holes_flag=args.fill_blob_holes,
            hole_area_threshold=args.blob_hole_threshold,
            edge_smoothing_kernel=args.blob_edge_kernel,
            edge_smoothing_iterations=args.blob_edge_iterations
        )
        if success:
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(mask_files)} mask(s) processed successfully")
    
    if success_count < len(mask_files):
        sys.exit(1)


if __name__ == "__main__":
    main()
