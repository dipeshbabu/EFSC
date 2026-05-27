from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry_csv", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--status", required=True, choices=["planned", "running", "done", "failed"])
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    path = Path(args.registry_csv)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty registry: {path}")

    found = False
    for row in rows:
        if row["run_id"] == args.run_id:
            row["status"] = args.status
            if args.notes:
                row["notes"] = args.notes
            found = True
            break

    if not found:
        raise ValueError(f"run_id not found: {args.run_id}")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {args.run_id} to status={args.status}")


if __name__ == "__main__":
    main()
