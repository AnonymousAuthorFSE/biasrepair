"""
Compute Cohen's kappa from two rating CSVs (annotator 1 and annotator 2).
Expects CSV with columns: sentence_id, annotator_id (or separate files), and criterion columns (e.g. bias_removal, semantic_preservation, fluency, pedagogical_fit) with values 1-5.
"""
import argparse
import csv
from pathlib import Path


def load_ratings(path: Path, annotator_col: str = "annotator_id", criteria: list[str] | None = None) -> dict[str, dict[str, int]]:
    """Returns dict: sentence_id -> {criterion: score}."""
    if criteria is None:
        criteria = ["bias_removal", "semantic_preservation", "fluency", "pedagogical_fit"]
    out = {}
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row.get("sentence_id", "").strip()
            if not sid:
                continue
            out[sid] = {}
            for c in criteria:
                if c in row and row[c].strip().isdigit():
                    out[sid][c] = int(row[c].strip())
    return out


def cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
    """Cohen's kappa for two lists of integer ratings (same length)."""
    n = len(rater1)
    if n != len(rater2) or n == 0:
        return 0.0
    min_val = min(min(rater1), min(rater2))
    max_val = max(max(rater1), max(rater2))
    categories = list(range(min_val, max_val + 1))
    # Observed agreement
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n
    # Expected agreement by chance
    from collections import Counter
    c1, c2 = Counter(rater1), Counter(rater2)
    pe = 0.0
    for k in categories:
        pe += (c1.get(k, 0) / n) * (c2.get(k, 0) / n)
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv1", help="First annotator rating sheet (with annotator_id=1 or single annotator)")
    parser.add_argument("csv2", help="Second annotator rating sheet")
    parser.add_argument("--criteria", nargs="+", default=["bias_removal", "semantic_preservation", "fluency", "pedagogical_fit"])
    args = parser.parse_args()

    ratings1 = load_ratings(Path(args.csv1), criteria=args.criteria)
    ratings2 = load_ratings(Path(args.csv2), criteria=args.criteria)
    common_ids = sorted(set(ratings1.keys()) & set(ratings2.keys()))

    print("Cohen's kappa per criterion (two annotators):")
    for c in args.criteria:
        a1 = [ratings1[sid].get(c) for sid in common_ids if ratings1[sid].get(c) is not None]
        a2 = [ratings2[sid].get(c) for sid in common_ids if ratings2[sid].get(c) is not None]
        # Align by index
        a1 = [ratings1[sid][c] for sid in common_ids if c in ratings1[sid] and c in ratings2[sid]]
        a2 = [ratings2[sid][c] for sid in common_ids if c in ratings1[sid] and c in ratings2[sid]]
        if len(a1) > 0 and len(a1) == len(a2):
            k = cohens_kappa(a1, a2)
            print(f"  {c}: {k:.3f}")
    # Overall (average across criteria or concatenate)
    all_a1, all_a2 = [], []
    for c in args.criteria:
        for sid in common_ids:
            if c in ratings1[sid] and c in ratings2[sid]:
                all_a1.append(ratings1[sid][c])
                all_a2.append(ratings2[sid][c])
    if all_a1 and all_a2:
        k_overall = cohens_kappa(all_a1, all_a2)
        print(f"  overall (pooled): {k_overall:.3f}")


if __name__ == "__main__":
    main()
