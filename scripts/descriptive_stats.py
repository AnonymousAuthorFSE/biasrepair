"""
Descriptive statistics: counts per category; optionally per source if 'source' field exists.
Writes output to docs/descriptive_stats.md.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

from biasrepair.io import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/dataset", help="Path to dataset dir")
    parser.add_argument("--output", default="docs/descriptive_stats.md", help="Output markdown path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    labels_path = dataset_dir / "labels_multilabel.jsonl"
    if not labels_path.exists():
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("# Descriptive statistics\n\nNo `labels_multilabel.jsonl` found. Add dataset and re-run.\n")
        return

    rows = load_jsonl(str(labels_path))
    by_cat = defaultdict(int)
    by_source_cat = defaultdict(lambda: defaultdict(int))
    for row in rows:
        labels = row.get("labels", row.get("categories", []))
        if not isinstance(labels, list):
            labels = [labels]
        source = row.get("source", "")
        for L in labels:
            by_cat[L] += 1
            if source:
                by_source_cat[source][L] += 1

    lines = ["# Descriptive statistics\n", "## Counts per category\n", "| Category | Count |", "|----------|-------|"]
    for cat in sorted(by_cat.keys()):
        lines.append(f"| {cat} | {by_cat[cat]} |")
    total_labels = sum(by_cat.values())
    lines.append(f"\n**Total labels:** {total_labels}")
    lines.append(f"**Unique sentences (rows):** {len(rows)}\n")

    if by_source_cat:
        lines.append("## Counts per category per source\n")
        for src in sorted(by_source_cat.keys()):
            lines.append(f"### {src}\n")
            lines.append("| Category | Count |")
            lines.append("|----------|-------|")
            for cat in sorted(by_source_cat[src].keys()):
                lines.append(f"| {cat} | {by_source_cat[src][cat]} |")
            lines.append("")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
