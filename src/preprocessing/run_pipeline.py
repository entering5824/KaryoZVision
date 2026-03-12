#!/usr/bin/env python3
"""
End-to-end pipeline for karyotype image processing.

Runs all parts sequentially:
1. Preprocessing & Segmentation (Part 1)
2. Blob Extraction (Part 2)
3. Grid Mapping & Cropping (Part 3)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_part1(input_path: str, output_dir: str, **kwargs) -> bool:
    """
    Run Part 1: Preprocessing & Segmentation.
    
    Args:
        input_path: Path to input image(s)
        output_dir: Output directory
        **kwargs: Additional arguments for part1_preproc.py
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("PART 1: Preprocessing & Segmentation")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        "-m",
        "src.preprocessing.part1_preproc",
        "--input", input_path,
        "--out", output_dir
    ]
    
    # Add optional arguments
    if "method" in kwargs:
        cmd.extend(["--method", kwargs["method"]])
    if "threshold" in kwargs and kwargs["threshold"] is not None:
        cmd.extend(["--threshold", str(kwargs["threshold"])])
    if "kernel_size" in kwargs:
        cmd.extend(["--kernel-size", str(kwargs["kernel_size"])])
    if "min_area" in kwargs:
        cmd.extend(["--min-area", str(kwargs["min_area"])])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error in Part 1: {e}", file=sys.stderr)
        return False


def run_part2(mask_input: str, output_dir: str, **kwargs) -> bool:
    """
    Run Part 2: Blob Extraction & Feature Calculation.
    
    Args:
        mask_input: Path to mask files or Part 1 output directory
        output_dir: Output directory
        **kwargs: Additional arguments for part2_blobs.py
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("PART 2: Blob Extraction & Feature Calculation")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        "-m",
        "src.preprocessing.part2_blobs",
        "--input", mask_input,
        "--out", output_dir
    ]
    
    # Add optional arguments
    if "min_area" in kwargs:
        cmd.extend(["--min-area", str(kwargs["min_area"])])
    if "connectivity" in kwargs:
        cmd.extend(["--connectivity", str(kwargs["connectivity"])])
    if "no_csv" in kwargs and kwargs["no_csv"]:
        cmd.append("--no-csv")
    if "no_overlay" in kwargs and kwargs["no_overlay"]:
        cmd.append("--no-overlay")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error in Part 2: {e}", file=sys.stderr)
        return False


def run_part3(blobs_input: str, input_image_dir: str, output_dir: str, **kwargs) -> bool:
    """
    Run Part 3: Position Mapping → Chromosome Labels & Image Cropping.
    
    Args:
        blobs_input: Path to blobs JSON files or Part 2 output directory
        input_image_dir: Directory containing original input images
        output_dir: Output directory for cropped images
        **kwargs: Additional arguments for part3_map_and_crop.py
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("PART 3: Position Mapping → Chromosome Labels & Cropping")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        "-m",
        "src.preprocessing.part3_map_and_crop",
        "--blobs", blobs_input,
        "--input", input_image_dir,
        "--out", output_dir
    ]
    
    # Add optional arguments
    if "rows" in kwargs:
        cmd.extend(["--rows", str(kwargs["rows"])])
    if "cols" in kwargs:
        cmd.extend(["--cols", str(kwargs["cols"])])
    if "no_auto_grid" in kwargs and kwargs["no_auto_grid"]:
        cmd.append("--no-auto-grid")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error in Part 3: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end karyotype image processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on a directory of images
  python code/run_pipeline.py --input data_in/ --out out/
  
  # Run pipeline on a single image
  python code/run_pipeline.py --input data_in/image.jpg --out out/
  
  # Skip Part 3 (only preprocessing and blob extraction)
  python code/run_pipeline.py --input data_in/ --out out/ --skip-part3
  
  # Custom parameters
  python code/run_pipeline.py --input data_in/ --out out/ --method otsu --min-area 100
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
        help="Output directory (all parts will write to subdirectories)"
    )
    
    # Part 1 options
    parser.add_argument(
        "--method",
        type=str,
        choices=["otsu", "global", "adaptive"],
        default="otsu",
        help="Thresholding method for Part 1 (default: otsu)"
    )
    
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Threshold value for global thresholding (Part 1)"
    )
    
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Morphological kernel size (Part 1, default: 5)"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum blob area for filtering (Part 1 & 2, default: 100)"
    )
    
    # Part 2 options
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Connectivity for connected components (Part 2, default: 8)"
    )
    
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output in Part 2"
    )
    
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip overlay image output in Part 2"
    )
    
    # Part 3 options
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of rows in karyotype grid (Part 3, default: 6)"
    )
    
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in karyotype grid (Part 3, default: 4)"
    )
    
    parser.add_argument(
        "--no-auto-grid",
        action="store_true",
        help="Disable automatic grid detection in Part 3"
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip-part1",
        action="store_true",
        help="Skip Part 1 (assume masks already exist)"
    )
    
    parser.add_argument(
        "--skip-part2",
        action="store_true",
        help="Skip Part 2 (assume blobs already exist)"
    )
    
    parser.add_argument(
        "--skip-part3",
        action="store_true",
        help="Skip Part 3 (only run preprocessing and blob extraction)"
    )
    
    parser.add_argument(
        "--part1-only",
        action="store_true",
        help="Only run Part 1"
    )
    
    parser.add_argument(
        "--part2-only",
        action="store_true",
        help="Only run Part 2 (requires Part 1 output)"
    )
    
    parser.add_argument(
        "--part3-only",
        action="store_true",
        help="Only run Part 3 (requires Part 2 output)"
    )
    
    args = parser.parse_args()
    
    # Determine which parts to run
    run_p1 = not args.skip_part1 and not args.part2_only and not args.part3_only
    run_p2 = not args.skip_part2 and not args.part1_only and not args.part3_only
    run_p3 = not args.skip_part3 and not args.part1_only and not args.part2_only
    
    # Prepare output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for each part
    part1_kwargs = {
        "method": args.method,
        "threshold": args.threshold,
        "kernel_size": args.kernel_size,
        "min_area": args.min_area
    }
    
    part2_kwargs = {
        "min_area": args.min_area,
        "connectivity": args.connectivity,
        "no_csv": args.no_csv,
        "no_overlay": args.no_overlay
    }
    
    part3_kwargs = {
        "rows": args.rows,
        "cols": args.cols,
        "no_auto_grid": args.no_auto_grid
    }
    
    print("=" * 60)
    print("KARYOTYPE IMAGE PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print()
    print("Parts to run:")
    print(f"  Part 1 (Preprocessing): {'[OK]' if run_p1 else '[SKIP]'}")
    print(f"  Part 2 (Blob Extraction): {'[OK]' if run_p2 else '[SKIP]'}")
    print(f"  Part 3 (Mapping & Cropping): {'[OK]' if run_p3 else '[SKIP]'}")
    print()
    
    # Run Part 1
    if run_p1:
        success = run_part1(args.input, str(output_dir), **part1_kwargs)
        if not success:
            print("\n[ERROR] Part 1 failed. Stopping pipeline.", file=sys.stderr)
            sys.exit(1)
    
    # Run Part 2
    if run_p2:
        # Part 2 input is Part 1 output (masks directory)
        mask_input = str(output_dir / "masks") if run_p1 else args.input
        success = run_part2(mask_input, str(output_dir), **part2_kwargs)
        if not success:
            print("\n[ERROR] Part 2 failed. Stopping pipeline.", file=sys.stderr)
            sys.exit(1)
    
    # Run Part 3
    if run_p3:
        # Part 3 needs: blobs JSON files and original images
        blobs_input = str(output_dir)  # Part 2 writes blobs JSON here
        # Original images directory
        input_image_dir = args.input if Path(args.input).is_dir() else str(Path(args.input).parent)
        
        success = run_part3(blobs_input, input_image_dir, str(output_dir), **part3_kwargs)
        if not success:
            print("\n[ERROR] Part 3 failed. Stopping pipeline.", file=sys.stderr)
            sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    if run_p1:
        print(f"  - Masks: {output_dir / 'masks'}")
    if run_p2:
        print(f"  - Blobs JSON/CSV: {output_dir}")
    if run_p3:
        print(f"  - Cropped chromosomes: {output_dir / '1'}, {output_dir / '2'}, ..., {output_dir / 'X'}, {output_dir / 'Y'}")
        print(f"  - Mapping file: {output_dir / 'mapping.json'}")
    print()


if __name__ == "__main__":
    main()
