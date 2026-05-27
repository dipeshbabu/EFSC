from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def safe_copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--tables_dir", required=True)
    parser.add_argument("--plots_dir", required=True)
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_copy(Path(args.results_csv), output_dir / "tables" / "merged_results.csv")
    for path in Path(args.tables_dir).glob("*.tex"):
        safe_copy(path, output_dir / "tables" / path.name)
    for path in Path(args.plots_dir).glob("*.png"):
        safe_copy(path, output_dir / "figures" / path.name)
    for path in Path(args.analysis_dir).glob("*.json"):
        safe_copy(path, output_dir / "analysis" / path.name)
    print(f"Wrote final paper bundle to {output_dir}")


if __name__ == "__main__":
    main()
