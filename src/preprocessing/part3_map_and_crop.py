#!/usr/bin/env python3
"""
Part 3: Position Mapping → Chromosome Labels & Image Cropping

Maps blob centroids to karyotype grid positions and crops individual chromosomes.
- Automatically detects grid layout from centroids
- Maps centroids to chromosome labels (1-22, X, Y)
- Crops images using bounding boxes
- Saves cropped images to organized directory structure
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from .config import load_layout_config

ROW_BOUNDARIES = [344, 500, 626]  # Fixed Y boundaries for 4 karyotype rows
ROW_TOLERANCE = 25  # pixels tolerance for near-boundary adjustment


def assign_row_index(y: float, row_boundaries: List[float]) -> int:
    """Map Y coordinate to row index based on fixed boundaries."""
    for idx, boundary in enumerate(row_boundaries):
        if y < boundary:
            return idx
    return len(row_boundaries)


def build_row_sequences(layout: dict) -> List[List[str]]:
    """Expand chromosome names into per-blob sequence for each row."""
    sequences = []
    for row in layout.get("row_config", []):
        indiv = set(row.get("individual_chromosomes", []))
        seq = []
        for chrom in row.get("chromosomes", []):
            if chrom in indiv:
                seq.append(chrom)
            else:
                seq.extend([chrom, chrom])
        sequences.append(seq)
    return sequences


def sort_blobs_by_row_col(blobs: List[Dict[str, Any]], row_tol: float = 25.0) -> List[Dict[str, Any]]:
    """
    Sort blobs first by row (top->bottom), then by column (left->right).
    
    This ensures proper ordering before label assignment, preventing pairs from being reversed
    (e.g., 1↔2, 3↔4).
    
    Args:
        blobs: List of blob dictionaries with 'centroid' key
        row_tol: Maximum y-distance (pixels) to consider blobs in the same row
    
    Returns:
        Sorted list of blobs
    """
    if not blobs:
        return blobs
    
    # Step 1: Sort by Y (top to bottom)
    blobs = sorted(blobs, key=lambda b: b['centroid'][1])
    
    sorted_blobs = []
    current_row = []
    last_y = None
    
    for b in blobs:
        cy = b['centroid'][1]
        if last_y is None or abs(cy - last_y) <= row_tol:
            current_row.append(b)
        else:
            # Sort current row by X (left to right)
            current_row = sorted(current_row, key=lambda b: b['centroid'][0])
            sorted_blobs.extend(current_row)
            current_row = [b]
        last_y = cy
    
    # Sort and add last row
    if current_row:
        current_row = sorted(current_row, key=lambda b: b['centroid'][0])
        sorted_blobs.extend(current_row)
    
    return sorted_blobs


def assign_chromosomes_by_layout(
    blobs: List[Dict[str, Any]],
    layout: dict,
    row_boundaries: List[float] = ROW_BOUNDARIES,
    row_tolerance: float = 25.0
) -> Dict[int, Tuple[Optional[str], bool]]:
    """
    Assign chromosome labels using improved sorting algorithm:
    1. Sort blobs by row (top->bottom) then column (left->right)
    2. Group into rows using tolerance
    3. Assign labels based on ISCN layout in correct order
    
    Returns mapping: blob_id -> (label, needs_review)
    """
    sequences = build_row_sequences(layout)
    
    # Step 1: Sort blobs by row → column before assigning labels
    blobs_sorted = sort_blobs_by_row_col(blobs, row_tolerance)
    
    # Step 2: Group sorted blobs into rows
    rows: List[List[Dict[str, Any]]] = []
    current_row: List[Dict[str, Any]] = []
    last_y = None
    
    for blob in blobs_sorted:
        cy = blob["centroid"][1]
        if last_y is None or abs(cy - last_y) <= row_tolerance:
            current_row.append(blob)
        else:
            # Start new row
            rows.append(current_row)
            current_row = [blob]
        last_y = cy
    
    # Add last row
    if current_row:
        rows.append(current_row)
    
    # Ensure we have at least as many rows as expected (pad if needed)
    while len(rows) < len(sequences):
        rows.append([])
    
    # Step 3: Assign labels based on position in row and ISCN layout
    # Blobs are already sorted by X within each row from sort_blobs_by_row_col()
    assignments: Dict[int, Tuple[Optional[str], bool]] = {}
    
    for idx, row_blobs in enumerate(rows):
        seq = sequences[idx] if idx < len(sequences) else []
        
        for j, blob in enumerate(row_blobs):
            if j < len(seq):
                label = seq[j]
                needs_review = False
            else:
                # More blobs than expected in sequence
                label = seq[-1] if seq else None
                needs_review = True
            
            assignments[blob["id"]] = (label, needs_review)
    
    return assignments


def assign_chromosome_to_centroid(
    centroid: Tuple[float, float],
    grid: List[Dict[str, Any]]
) -> Tuple[Optional[str], bool]:
    """
    Assign chromosome label to a centroid based on grid position.
    Improved to handle cases where centroid is not inside any cell.
    
    Args:
        centroid: (x, y) centroid coordinates
        grid: List of grid cells with chromosome labels and bounding boxes
    
    Returns:
        Tuple of (chromosome_label, needs_manual_review)
        - chromosome_label: "1", "2", ..., "22", "X", "Y", or None
        - needs_manual_review: True if assignment is uncertain
    """
    cx, cy = centroid
    
    # First, try to find cell that contains the centroid
    best_match = None
    min_distance = float('inf')
    inside_cell = False
    
    for cell in grid:
        x1, y1, x2, y2 = cell["bbox"]
        
        # Check if centroid is inside cell
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            inside_cell = True
            # Calculate distance to center of cell
            cell_center_x = (x1 + x2) / 2
            cell_center_y = (y1 + y2) / 2
            distance = np.sqrt((cx - cell_center_x)**2 + (cy - cell_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_match = cell
    
    # If found inside a cell, return it
    if best_match and inside_cell:
        # If distance to center is large, flag for review
        needs_review = min_distance > 50  # Threshold for uncertainty
        return best_match["chrom"], needs_review
    
    # If not inside any cell, find nearest cell by distance
    if not best_match:
        for cell in grid:
            x1, y1, x2, y2 = cell["bbox"]
            cell_center_x = (x1 + x2) / 2
            cell_center_y = (y1 + y2) / 2
            distance = np.sqrt((cx - cell_center_x)**2 + (cy - cell_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_match = cell
    
    if best_match:
        # If distance is very large, flag for review
        needs_review = min_distance > 100  # Higher threshold for outside cells
        return best_match["chrom"], needs_review
    
    # No match found - needs manual review
    return None, True


def convert_bbox_format(bbox: List[int]) -> Tuple[int, int, int, int]:
    """
    Convert bbox from [left, top, width, height] to [x1, y1, x2, y2].
    
    Args:
        bbox: Bounding box as [left, top, width, height]
    
    Returns:
        Tuple of (x1, y1, x2, y2)
    """
    left, top, width, height = bbox
    return left, top, left + width, top + height


def process_blobs_file(
    blob_json_path: str,
    input_image_dir: str,
    output_dir: str,
    layout_config: dict,
    row_boundaries: List[float] = ROW_BOUNDARIES
) -> List[Dict[str, Any]]:
    """
    Process a single blobs JSON file: map centroids to chromosomes and crop images.
    
    Args:
        blob_json_path: Path to blobs JSON file from Part 2
        input_image_dir: Directory containing original input images
        output_dir: Output directory for cropped images
        layout_config: Layout configuration loaded from JSON
        row_boundaries: Fixed Y boundaries to split rows
    
    Returns:
        List of mapping results
    """
    # Read blobs JSON
    with open(blob_json_path, 'r', encoding='utf-8') as f:
        blob_data = json.load(f)
    
    # Get image name and blobs
    image_name = blob_data.get("image", "")
    blobs = blob_data.get("blobs", [])
    
    if not image_name or not blobs:
        print(f"Warning: Invalid blob data in {blob_json_path}", file=sys.stderr)
        return []
    
    # Find original image file
    image_path = None
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
        candidate = Path(input_image_dir) / f"{image_name}{ext}"
        if candidate.exists():
            image_path = str(candidate)
            break
    
    if not image_path:
        print(f"Warning: Could not find image file for {image_name} in {input_image_dir}", file=sys.stderr)
        return []
    
    # Read original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}", file=sys.stderr)
        return []
    
    # Assign chromosomes using layout and fixed row boundaries
    assignments = assign_chromosomes_by_layout(blobs, layout_config, row_boundaries)
    
    # Get individual chromosomes from layout config
    individual_chromosomes = set()
    for row_config in layout_config.get("row_config", []):
        if "individual_chromosomes" in row_config:
            for chrom in row_config["individual_chromosomes"]:
                individual_chromosomes.add(chrom)
    
    # Process each blob
    results = []
    base_name = Path(image_name).stem
    chromosome_counts: Dict[str, int] = {}  # Track count per chromosome for suffix
    
    for blob in blobs:
        blob_id = blob['id']
        centroid = blob['centroid']
        bbox = blob['bbox']  # [left, top, width, height]
        
        # Assign chromosome label
        chrom_label, needs_review = assignments.get(blob_id, (None, True))
        
        # Convert bbox format
        x1, y1, x2, y2 = convert_bbox_format(bbox)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Crop image
        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
            
            # Determine chromosome label (use "UNK" if not assigned)
            final_label = chrom_label if chrom_label else "UNK"
            
            # Create chromosome directory
            chr_dir = Path(output_dir) / final_label
            chr_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with suffix for pairs (like karyotype_extractor.py)
            if final_label in individual_chromosomes:
                # Individual chromosomes (X, Y): no suffix
                output_filename = f"{base_name}_{final_label}.jpg"
            else:
                # Pairs: add suffix 'a', 'b', etc.
                count = chromosome_counts.get(final_label, 0)
                chromosome_counts[final_label] = count + 1
                suffix = chr(ord('a') + count) if count < 26 else f"_{count}"
                output_filename = f"{base_name}_{final_label}{suffix}.jpg"
            
            output_path = chr_dir / output_filename
            
            cv2.imwrite(str(output_path), crop)
            
            # Record mapping result
            # Ensure all values are JSON-serializable (convert numpy types to Python types)
            results.append({
                "original_file": str(image_name),
                "blob_id": int(blob_id),
                "assigned_chromosome": str(chrom_label) if chrom_label else None,
                "chromosome_label_used": str(final_label),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_original": [int(b) for b in bbox],
                "centroid": [float(c) for c in centroid],
                "flag_manual_review": bool(needs_review),  # Ensure Python bool, not numpy bool
                "output_path": str(output_path)
            })
        else:
            print(f"Warning: Invalid bbox for blob {blob_id}: {bbox}", file=sys.stderr)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Part 3: Map centroids to chromosome labels and crop images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process blobs from Part 2 output
  python code/part3_map_and_crop.py --blobs out/ --input data_in/ --out out/
  
  # Use custom grid dimensions
  python code/part3_map_and_crop.py --blobs out/ --input data_in/ --out out/ --rows 6 --cols 4
        """
    )
    
    parser.add_argument(
        "--blobs",
        type=str,
        required=True,
        help="Path to blobs JSON files (directory or single file) or Part 2 output directory"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing original input images"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for cropped chromosome images"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="karyotype_layout.json",
        help="Path to layout configuration JSON"
    )
    # Backward compatibility (ignored)
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="(Deprecated) Grid rows - ignored; layout comes from config"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="(Deprecated) Grid cols - ignored; layout comes from config"
    )
    
    args = parser.parse_args()
    
    # Load layout configuration
    layout_config = load_layout_config(args.config)

    # Find blob JSON files
    blob_path = Path(args.blobs)
    blob_files = []
    
    if blob_path.is_file() and blob_path.suffix == '.json':
        blob_files = [str(blob_path)]
    elif blob_path.is_dir():
        # Look for *_blobs.json files in the directory
        blob_files = [str(f) for f in blob_path.glob("*_blobs.json")]
        if not blob_files:
            # Check subdirectories (e.g., if blobs are in a subdirectory)
            blob_files = [str(f) for f in blob_path.glob("**/*_blobs.json")]
    else:
        print(f"Error: Invalid blobs path: {args.blobs}", file=sys.stderr)
        sys.exit(1)
    
    if not blob_files:
        print(f"Error: No blob JSON files found in {args.blobs}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each blob file
    all_results = []
    
    print("Part 3: Mapping & Cropping")
    print("=" * 50)
    print(f"Found {len(blob_files)} blob file(s)")
    print(f"Output directory: {output_dir}")
    print(f"Layout config: {args.config}")
    print()
    
    for blob_file in blob_files:
        print(f"Processing: {Path(blob_file).name}")
        results = process_blobs_file(
            blob_file,
            args.input,
            str(output_dir),
            layout_config,
            ROW_BOUNDARIES
        )
        all_results.extend(results)
        print(f"  → Processed {len(results)} blobs")
    
    # Save mapping JSON
    # Ensure all values are JSON-serializable
    def make_json_serializable(obj):
        """Recursively convert numpy types and other non-serializable types to Python types."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None:
            return None
        else:
            return obj
    
    # Convert all results to JSON-serializable format
    serializable_results = make_json_serializable(all_results)
    
    mapping_path = output_dir / "mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 50)
    print(f"Completed: {len(all_results)} chromosomes cropped")
    print(f"Mapping file: {mapping_path}")
    
    # Count chromosomes by label
    from collections import Counter
    label_counts = Counter(r["chromosome_label_used"] for r in all_results)
    print("\nChromosome distribution:")
    for label in sorted(label_counts.keys(), key=lambda x: (x.isdigit() and int(x) or 999, x)):
        print(f"  {label}: {label_counts[label]}")


if __name__ == "__main__":
    main()
