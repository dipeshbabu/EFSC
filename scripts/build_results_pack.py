from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_all(src_dir: Path, dst_dir: Path, pattern: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        for path in src_dir.glob(pattern):
            shutil.copy2(path, dst_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--subsets_dir", required=True)
    parser.add_argument("--efficiency_dir", required=True)
    parser.add_argument("--tables_dir", required=True)
    parser.add_argument("--plots_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    copy_all(Path(args.metrics_dir), output / "metrics", "*.json")
    copy_all(Path(args.subsets_dir), output / "subsets", "*.json")
    copy_all(Path(args.efficiency_dir), output / "efficiency", "*.json")
    copy_all(Path(args.tables_dir), output / "tables", "*.tex")
    copy_all(Path(args.plots_dir), output / "figures", "*.png")
    print(f"Built results pack at {output}")


if __name__ == "__main__":
    main()
