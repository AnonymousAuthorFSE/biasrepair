"""
Generate a rating sheet CSV from predictions.jsonl for manual evaluation.
Samples 20 per category (100 total) or uses provided limit. Annotators fill bias_removal, semantic_preservation, fluency, pedagogical_fit (1–5).
"""
import argparse
import csv
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl", help="Path to predictions.jsonl (e.g. runs/.../predictions.jsonl)")
    parser.add_argument("--output", default="manual_eval/rating_sheet.csv", help="Output CSV path")
    parser.add_argument("--per_category", type=int, default=20, help="Number of sentences per category (default 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    with open(args.predictions_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_cat = {}
    for r in rows:
        cat = r.get("category", "Unknown")
        by_cat.setdefault(cat, []).append(r)

    random.seed(args.seed)
    sampled = []
    for cat, items in by_cat.items():
        n = min(args.per_category, len(items))
        sampled.extend(random.sample(items, n))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "category", "original", "rewrite", "annotator_id", "bias_removal", "semantic_preservation", "fluency", "pedagogical_fit", "notes"])
        for r in sampled:
            w.writerow([
                r.get("id", ""),
                r.get("category", ""),
                r.get("original", ""),
                r.get("rewrite", ""),
                "",  # annotator_id
                "",  # bias_removal 1-5
                "",  # semantic_preservation 1-5
                "",  # fluency 1-5
                "",  # pedagogical_fit 1-5
                "",
            ])
    print(f"Wrote {out_path} with {len(sampled)} rows.")


if __name__ == "__main__":
    main()
