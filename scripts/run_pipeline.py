"""
Simple CLI entrypoint to run the preprocessing pipeline
from raw karyotype sheets to cropped chromosomes.
"""

import sys
from pathlib import Path


def main() -> None:
    # Ensure project root is on sys.path so `import src` works
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.pipeline import run_preprocessing

    raw_dir = Path("data") / "raw"
    out_dir = Path("data") / "preprocessed"

    run_preprocessing(input_path=str(raw_dir), output_dir=str(out_dir))


if __name__ == "__main__":
    main()


