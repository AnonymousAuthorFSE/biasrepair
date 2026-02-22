"""
Load JSONL datasets, split IDs, and validate schema.
Schema: consolidated_sentences (id, sentence), labels_multilabel (id, labels[]), ground_truth_rewrites (id, rewrite).
"""
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split_ids(splits_dir: str | Path, split_name: str) -> list[str]:
    path = Path(splits_dir) / f"{split_name}_ids.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_consolidated_sentences(dataset_dir: str | Path) -> dict[str, str]:
    path = Path(dataset_dir) / "consolidated_sentences.jsonl"
    out = {}
    for row in load_jsonl(path):
        sid = row.get("id")
        sent = row.get("sentence", row.get("text", ""))
        if sid is not None:
            out[str(sid)] = sent
    return out


def load_labels_multilabel(dataset_dir: str | Path) -> dict[str, list[str]]:
    path = Path(dataset_dir) / "labels_multilabel.jsonl"
    out = {}
    for row in load_jsonl(path):
        sid = row.get("id")
        labels = row.get("labels", row.get("categories", []))
        if sid is not None:
            out[str(sid)] = labels if isinstance(labels, list) else [labels]
    return out


def load_ground_truth_rewrites(dataset_dir: str | Path) -> dict[str, str]:
    path = Path(dataset_dir) / "ground_truth_rewrites.jsonl"
    out = {}
    for row in load_jsonl(path):
        sid = row.get("id")
        rew = row.get("rewrite", row.get("reference", ""))
        if sid is not None:
            out[str(sid)] = rew
    return out


# Repair categories (Neutral excluded from repair evaluation; used only as exemplars)
REPAIR_CATEGORIES = [
    "GenericPronouns",
    "ExclusionaryTerms",
    "StereotypingBias",
    "Sexism",
    "SemanticBias",
]


def get_eval_instances(
    dataset_dir: str | Path,
    splits_dir: str | Path,
    split_name: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return list of {id, sentence, category, reference} for evaluation. Uses first non-Neutral label per instance."""
    ids = load_split_ids(splits_dir, split_name)
    sentences = load_consolidated_sentences(dataset_dir)
    labels = load_labels_multilabel(dataset_dir)
    rewrites = load_ground_truth_rewrites(dataset_dir)
    out = []
    for sid in ids:
        if sid not in sentences or sid not in labels:
            continue
        cats = labels[sid]
        # Pick first repair category (exclude Neutral for evaluation)
        category = None
        for c in cats:
            if c in REPAIR_CATEGORIES:
                category = c
                break
        if category is None:
            continue
        ref = rewrites.get(sid, "")
        out.append({
            "id": sid,
            "sentence": sentences[sid],
            "category": category,
            "reference": ref,
        })
        if limit is not None and len(out) >= limit:
            break
    return out


def load_exemplars(exemplars_path: str | Path) -> list[dict[str, Any]]:
    """Load bias-free exemplars: expect {sentence, rewrite} or {original, rewrite}."""
    rows = load_jsonl(exemplars_path)
    out = []
    for r in rows:
        orig = r.get("sentence") or r.get("original", "")
        rew = r.get("rewrite", "")
        if orig or rew:
            out.append({"original": orig, "rewrite": rew, **r})
    return out
