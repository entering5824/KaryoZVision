#!/usr/bin/env python3
"""
Part 1: Preprocessing & Segmentation for Karyotype Image Processing

Main CLI script for preprocessing chromosome images:
- Applies CLAHE for contrast enhancement
- Creates binary masks using thresholding (Otsu or global)
- Applies morphological operations for refinement
- Outputs: mask PNG, preprocessed PNG, metadata JSON
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from .preprocessing_utils import (
    apply_clahe,
    apply_threshold,
    apply_morphology,
    fill_holes,
    fill_small_holes,
    remove_small_components,
    remove_border_artifacts,
    apply_row_specific_morphology,
    smooth_edges,
    reduce_fragmentation,
    save_metadata
)
from datetime import datetime


def get_image_files(input_path: str) -> List[str]:
    """
    Get list of image files from input path (file or directory).
    
    Args:
        input_path: Path to image file or directory
    
    Returns:
        List of image file paths
    """
    path = Path(input_path)
    
    if path.is_file():
        # Single file
        return [str(path)]
    elif path.is_dir():
        # Directory - find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(path.glob(f'*{ext}'))
        return sorted([str(f) for f in image_files])
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def process_image(
    image_path: str,
    output_dir: str,
    method: str = "otsu",
    threshold_value: Optional[int] = None,
    kernel_size: int = 5,
    fill_holes_flag: bool = True,
    min_area: int = 100,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    use_inverse: bool = True,
    adaptive_block_size: int = 11,
    adaptive_c: float = 2.0,
    border_size: int = 50,
    use_row_morphology: bool = True,
    otsu_offset: int = 10,
    hole_area_threshold: int = 500,
    edge_smoothing_kernel: int = 3,
    edge_smoothing_iterations: int = 1,
    fragmentation_merge_distance: int = 15,
    reduce_fragmentation_flag: bool = True,
    smooth_edges_flag: bool = True
) -> bool:
    """
    Process a single image: CLAHE → threshold → morphology → post-process → save outputs.
    
    For karyotype images (dark chromosomes on bright background):
    - Uses THRESH_BINARY_INV by default (use_inverse=True)
    - Mask result: chromosomes = 1 (white), background = 0 (black)
    
    Args:
        image_path: Path to input image
        output_dir: Output directory (will create masks subdirectory)
        method: Thresholding method ("otsu", "global", or "adaptive")
        threshold_value: Threshold value for global method
        kernel_size: Morphological kernel size
        fill_holes_flag: Whether to fill holes
        min_area: Minimum component area
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_size: CLAHE tile size
        use_inverse: Use inverse thresholding (True for dark objects on bright background)
        adaptive_block_size: Block size for adaptive thresholding (must be odd)
        adaptive_c: Constant for adaptive thresholding
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read image {image_path}", file=sys.stderr)
            return False
        
        # Get base filename without extension
        base_name = Path(image_path).stem
        
        # Ensure output directory exists (create masks subdirectory)
        masks_dir = Path(output_dir) / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Apply CLAHE preprocessing
        clahe_image = apply_clahe(
            image,
            clip_limit=clahe_clip_limit,
            tile_size=(clahe_tile_size, clahe_tile_size)
        )
        
        # Save preprocessed image
        preproc_path = masks_dir / f"{base_name}_preproc.png"
        cv2.imwrite(str(preproc_path), clahe_image)
        
        # Step 2: Apply thresholding
        # Use inverse thresholding by default (chromosomes are darker than background)
        binary_mask, threshold_used = apply_threshold(
            clahe_image,
            method=method,
            threshold_value=threshold_value,
            use_inverse=use_inverse,
            adaptive=(method == "adaptive"),
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            otsu_offset=otsu_offset
        )
        # Remove border artifacts early
        binary_mask = remove_border_artifacts(binary_mask, border_size=border_size)
        
        # Step 3: Apply morphological operations
        morphology_ops = ["opening", "closing"]
        refined_mask = apply_morphology(
            binary_mask,
            kernel_size=kernel_size,
            operations=morphology_ops
        )

        # Row-specific morphology to better separate chromosomes by row
        if use_row_morphology:
            refined_mask = apply_row_specific_morphology(refined_mask)
        
        # Step 4: Reduce fragmentation (merge nearby fragments)
        if reduce_fragmentation_flag:
            refined_mask = reduce_fragmentation(refined_mask, merge_distance=fragmentation_merge_distance)
        
        # Step 5: Post-processing
        final_mask = refined_mask.copy()
        
        if fill_holes_flag:
            final_mask = fill_holes(final_mask, hole_area_threshold=hole_area_threshold)
        
        if smooth_edges_flag:
            final_mask = smooth_edges(final_mask, kernel_size=edge_smoothing_kernel, iterations=edge_smoothing_iterations)
        
        if min_area > 0:
            final_mask = remove_small_components(final_mask, min_area=min_area)
        
        # Save binary mask (as 0-255 PNG)
        mask_path = masks_dir / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), final_mask * 255)
        
        # Step 5: Create and save metadata
        metadata = {
            "input_image": os.path.basename(image_path),
            "threshold_method": method,
            "threshold_value": float(threshold_used),
            "use_inverse": use_inverse,
            "kernel_size": kernel_size,
            "morphology_operations": morphology_ops,
            "fill_holes": fill_holes_flag,
            "min_component_area": min_area,
            "border_size": border_size,
            "use_row_morphology": use_row_morphology,
            "clahe_params": {
                "clip_limit": clahe_clip_limit,
                "tile_size": [clahe_tile_size, clahe_tile_size]
            },
            "adaptive_params": {
                "block_size": adaptive_block_size,
                "c": adaptive_c
            } if method == "adaptive" else None,
            "threshold_params": {
                "method": method,
                "otsu_offset": otsu_offset,
                "threshold_used": float(threshold_used)
            },
            "hole_area_threshold": hole_area_threshold,
            "edge_smoothing_kernel": edge_smoothing_kernel,
            "edge_smoothing_iterations": edge_smoothing_iterations,
            "fragmentation_merge_distance": fragmentation_merge_distance,
            "reduce_fragmentation": reduce_fragmentation_flag,
            "smooth_edges": smooth_edges_flag,
            "processing_timestamp": datetime.now().isoformat(),
            "image_shape": list(image.shape)
        }
        
        meta_path = masks_dir / f"{base_name}_mask_meta.json"
        save_metadata(str(meta_path), metadata)
        
        print(f"Processed: {image_path} -> {base_name}_mask.png")
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Part 1: Preprocessing & Segmentation for Karyotype Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with Otsu thresholding (inverse by default)
  python code/part1_preproc.py --input data_in/image.jpg --out out/
  
  # Process directory with global thresholding
  python code/part1_preproc.py --input data_in/ --out out/ --method global --threshold 127
  
  # Use adaptive thresholding if Otsu fails
  python code/part1_preproc.py --input data_in/ --out out/ --method adaptive
  
  # Custom parameters
  python code/part1_preproc.py --input data_in/image.jpg --out out/ --kernel-size 7 --min-area 200
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image file or directory"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory (masks will be saved to <out>/masks/)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["otsu", "global", "adaptive"],
        default="otsu",
        help="Thresholding method: 'otsu' (automatic), 'global' (fixed T), or 'adaptive' (default: otsu)"
    )
    
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Threshold value T for global thresholding (required if --method=global)"
    )
    
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Morphological kernel size (default: 5)"
    )
    
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        default=True,
        help="Fill holes in binary mask (default: True)"
    )
    
    parser.add_argument(
        "--no-fill-holes",
        dest="fill_holes",
        action="store_false",
        help="Disable filling holes"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum area (pixels) to keep a connected component (default: 100)"
    )
    
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (default: 2.0)"
    )
    
    parser.add_argument(
        "--clahe-tile-size",
        type=int,
        default=8,
        help="CLAHE tile size (default: 8)"
    )
    
    parser.add_argument(
        "--no-inverse",
        dest="use_inverse",
        action="store_false",
        default=True,
        help="Disable inverse thresholding (use normal thresholding). Default: inverse (for dark chromosomes on bright background)"
    )
    
    parser.add_argument(
        "--adaptive-block-size",
        type=int,
        default=11,
        help="Block size for adaptive thresholding (must be odd, default: 11)"
    )
    
    parser.add_argument(
        "--adaptive-c",
        type=float,
        default=2.0,
        help="Constant subtracted from mean in adaptive thresholding (default: 2.0)"
    )
    parser.add_argument(
        "--border-size",
        type=int,
        default=50,
        help="Border size (pixels) to clear artifacts (default: 50)"
    )
    parser.add_argument(
        "--no-row-morphology",
        dest="use_row_morphology",
        action="store_false",
        default=True,
        help="Disable row-specific morphology"
    )
    parser.add_argument(
        "--otsu-offset",
        type=int,
        default=10,
        help="Offset added to Otsu threshold for stricter separation (default: 10)"
    )
    parser.add_argument(
        "--hole-area-threshold",
        type=int,
        default=500,
        help="Maximum area of holes to fill (pixels, default: 500)"
    )
    parser.add_argument(
        "--edge-smoothing-kernel",
        type=int,
        default=3,
        help="Kernel size for edge smoothing (default: 3)"
    )
    parser.add_argument(
        "--edge-smoothing-iterations",
        type=int,
        default=1,
        help="Number of iterations for edge smoothing (default: 1)"
    )
    parser.add_argument(
        "--fragmentation-merge-distance",
        type=int,
        default=15,
        help="Distance threshold for merging fragments (pixels, default: 15)"
    )
    parser.add_argument(
        "--no-reduce-fragmentation",
        dest="reduce_fragmentation",
        action="store_false",
        default=True,
        help="Disable fragmentation reduction"
    )
    parser.add_argument(
        "--no-smooth-edges",
        dest="smooth_edges",
        action="store_false",
        default=True,
        help="Disable edge smoothing"
    )
    
    args = parser.parse_args()
    
    # Validate threshold for global method
    if args.method == "global" and args.threshold is None:
        parser.error("--threshold is required when --method=global")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images to process
    try:
        image_files = get_image_files(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not image_files:
        print(f"Warning: No image files found in {args.input}", file=sys.stderr)
        sys.exit(0)
    
    print(f"Found {len(image_files)} image(s) to process")
    print(f"Output directory: {output_dir / 'masks'}")
    print(f"Method: {args.method}")
    if args.method == "global":
        print(f"Threshold: {args.threshold}")
    print()
    
    # Process each image
    success_count = 0
    for image_path in image_files:
        success = process_image(
            image_path,
            str(output_dir),
            method=args.method,
            threshold_value=args.threshold,
            kernel_size=args.kernel_size,
            fill_holes_flag=args.fill_holes,
            min_area=args.min_area,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_size=args.clahe_tile_size,
            use_inverse=args.use_inverse,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
            border_size=args.border_size,
            use_row_morphology=args.use_row_morphology,
            otsu_offset=args.otsu_offset,
            hole_area_threshold=args.hole_area_threshold,
            edge_smoothing_kernel=args.edge_smoothing_kernel,
            edge_smoothing_iterations=args.edge_smoothing_iterations,
            fragmentation_merge_distance=args.fragmentation_merge_distance,
            reduce_fragmentation_flag=args.reduce_fragmentation,
            smooth_edges_flag=args.smooth_edges
        )
        if success:
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(image_files)} images processed successfully")
    
    if success_count < len(image_files):
        sys.exit(1)


if __name__ == "__main__":
    main()

