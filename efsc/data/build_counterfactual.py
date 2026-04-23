from __future__ import annotations
import argparse
from efsc.utils import load_jsonl, save_jsonl

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    by_family = {}
    for row in rows: by_family.setdefault(row["family_id"], []).append(row)
    pairs = []
    for family, fam_rows in by_family.items():
        harmful = [x for x in fam_rows if x["variant_type"] in {"harmful_malicious", "jailbreak", "toxic_harmful"}]
        benign = [x for x in fam_rows if x["variant_type"] in {"educational_benign", "defensive_benign", "authorized_benign", "hard_benign", "ambiguous_legitimate", "translation_benign", "paraphrase_benign"}]
        for h in harmful:
            for b in benign:
                pairs.append({"family_id": family, "harmful_id": h["id"], "benign_id": b["id"], "harmful_prompt": h["prompt"], "benign_prompt": b["prompt"]})
    save_jsonl(pairs, args.output)

if __name__ == "__main__":
    main()
