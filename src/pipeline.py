"""
Unified pipeline entrypoint for chromosome classification.

This module wraps the karyotype preprocessing steps (ported from the
`cv-karyotype-preprocessing` project) and connects them with the
chromosome classification data directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import config as clf_config


def run_preprocessing(
    input_path: str,
    output_dir: Optional[str] = None,
    *,
    method: str = "otsu",
    threshold: Optional[int] = None,
    kernel_size: int = 5,
    min_area: int = 100,
    connectivity: int = 8,
    no_csv: bool = False,
    no_overlay: bool = False,
    rows: int = 4,
    cols: int = 10,
    no_auto_grid: bool = False,
    skip_part1: bool = False,
    skip_part2: bool = False,
    skip_part3: bool = False,
    part1_only: bool = False,
    part2_only: bool = False,
    part3_only: bool = False,
) -> Path:
    """
    Run the full (or partial) karyotype preprocessing pipeline.

    Returns:
        Path to the preprocessing output directory.
    """
    if output_dir is None:
        output_dir = str(clf_config.DATA_DIR / "preprocessed")

    cmd = [
        sys.executable,
        "-m",
        "src.preprocessing.run_pipeline",
        "--input",
        str(input_path),
        "--out",
        str(output_dir),
        "--method",
        method,
        "--kernel-size",
        str(kernel_size),
        "--min-area",
        str(min_area),
        "--connectivity",
        str(connectivity),
        "--rows",
        str(rows),
        "--cols",
        str(cols),
    ]

    if threshold is not None:
        cmd += ["--threshold", str(threshold)]
    if no_csv:
        cmd.append("--no-csv")
    if no_overlay:
        cmd.append("--no-overlay")
    if no_auto_grid:
        cmd.append("--no-auto-grid")

    if skip_part1:
        cmd.append("--skip-part1")
    if skip_part2:
        cmd.append("--skip-part2")
    if skip_part3:
        cmd.append("--skip-part3")
    if part1_only:
        cmd.append("--part1-only")
    if part2_only:
        cmd.append("--part2-only")
    if part3_only:
        cmd.append("--part3-only")

    subprocess.run(cmd, check=True)

    return Path(output_dir)


def preprocess_to_classification_data(
    input_path: str,
    *,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Convenience helper that runs preprocessing and prepares data
    for the classification pipeline.
    """
    out_dir = run_preprocessing(input_path=input_path, output_dir=output_dir)
    # At this point, cropped chromosomes are expected to live under out_dir
    # subfolders 1-22, X, Y. Users can then move/link these into DATA_DIR
    # if needed for training.
    return out_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified pipeline wrapper for chromosome preprocessing.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input karyotype sheet image file or directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for preprocessed data (default: data/preprocessed).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_preprocessing(input_path=args.input, output_dir=args.out)


if __name__ == "__main__":
    main()

