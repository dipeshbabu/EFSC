from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows: List[Dict] = []
    for path in sorted(metrics_dir.glob("*.json")):
        row = {"file": path.name}
        row.update(load_json(path))
        rows.append(row)
    if not rows:
        raise ValueError(f"No metric files found in {metrics_dir}")

    fieldnames = sorted({key for row in rows for key in row})
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote aggregate results to {output}")


if __name__ == "__main__":
    main()
