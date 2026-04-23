from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(vals):
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [load_json(Path(path)) for path in args.metric_files]
    keys = sorted(set().union(*(set(row.keys()) for row in rows)))
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            m, s = mean_std(vals)
            out[key] = {"mean": m, "std": s, "n": len(vals)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
