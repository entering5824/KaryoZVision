"""
Configuration file for karyotype image processing pipeline.
Contains default parameters and settings.
"""

import json
from pathlib import Path

# Default thresholding parameters
DEFAULT_THRESHOLD_METHOD = "otsu"
DEFAULT_THRESHOLD_VALUE = 127  # Only used for global method

# Default morphological parameters
DEFAULT_KERNEL_SIZE = 5
DEFAULT_MORPHOLOGY_OPS = ["opening", "closing"]

# Default post-processing parameters
DEFAULT_FILL_HOLES = True
DEFAULT_MIN_AREA = 100

# Default CLAHE parameters
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_SIZE = 8

# Default layout configuration (ISCN inspired)
DEFAULT_LAYOUT = {
    "row_config": [
        {"row": 1, "chromosomes": ["1", "2", "3", "4", "5"], "blobs_expected": 10},
        {"row": 2, "chromosomes": ["6", "7", "8", "9", "10", "11", "12"], "blobs_expected": 14},
        {"row": 3, "chromosomes": ["13", "14", "15", "16", "17", "18"], "blobs_expected": 12},
        {"row": 4, "chromosomes": ["19", "20", "21", "22", "X", "Y"], "blobs_expected": 10, "individual_chromosomes": ["X", "Y"]}
    ]
}

# Grid mapping parameters (for Part 3)
DEFAULT_GRID_ROWS = 4
DEFAULT_GRID_COLS = 10


def load_layout_config(config_path: str = "karyotype_layout.json") -> dict:
    """
    Load ISCN-like layout configuration from JSON.
    Falls back to DEFAULT_LAYOUT if file is missing or invalid.
    """
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "row_config" in data:
                    return data
        except Exception:
            pass
    return DEFAULT_LAYOUT


def derive_chromosome_order(layout: dict) -> list:
    """
    Convert layout row_config into CHROMOSOME_ORDER for mapping utilities.
    Each chromosome name appears once; pairs are represented as single labels.
    """
    order = []
    for row in layout.get("row_config", []):
        order.append(row.get("chromosomes", []))
    return order


# Derived default order from DEFAULT_LAYOUT
CHROMOSOME_ORDER = derive_chromosome_order(DEFAULT_LAYOUT)

