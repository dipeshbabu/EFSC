from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_meta(path: Path):
    parts = path.stem.split("__")
    return {
        "raw_name": path.stem,
        "model": parts[0] if len(parts) > 0 else "",
        "method": parts[1] if len(parts) > 1 else "",
        "seed": parts[2] if len(parts) > 2 else "",
        "precision": parts[3] if len(parts) > 3 else "",
        "dataset": "__".join(parts[4:]) if len(parts) > 4 else "",
    }


def flatten_value(value):
    if isinstance(value, dict) and "mean" in value:
        return value["mean"]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.metrics_dir).glob("*.json")):
        row = parse_meta(path)
        for key, value in load_json(path).items():
            value = flatten_value(value)
            if isinstance(value, (int, float, str, bool)):
                row[key] = value
        rows.append(row)

    fieldnames = sorted(set().union(*(set(row.keys()) for row in rows))) if rows else []
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
