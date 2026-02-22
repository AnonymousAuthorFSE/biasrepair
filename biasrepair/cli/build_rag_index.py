"""
Build persistent ChromaDB index from training split bias-free exemplars only (leakage prevention).
Index contains: bias-free exemplars file + bias-free rewrites from train split. No dev/test.
"""
import argparse
from pathlib import Path

import yaml

from biasrepair.io import load_exemplars, load_split_ids, load_consolidated_sentences, load_labels_multilabel, load_ground_truth_rewrites, REPAIR_CATEGORIES
from biasrepair.rag import build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. full_system)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = config.get("data", {})
    dataset_dir = Path(data.get("dataset_dir", "data/dataset"))
    splits_dir = Path(data.get("splits_dir", "data/splits"))
    exemplars_path = Path(data.get("exemplars_path", "data/exemplars/bias_free_exemplars.jsonl"))
    index_path = Path(config.get("rag", {}).get("index_path", "data/rag_index"))
    model_name = config.get("rag", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

    exemplars = load_exemplars(exemplars_path) if exemplars_path.exists() else []
    train_ids = load_split_ids(splits_dir, "train")
    sentences = load_consolidated_sentences(dataset_dir)
    labels = load_labels_multilabel(dataset_dir)
    rewrites = load_ground_truth_rewrites(dataset_dir)

    # Add bias-free rewrites from train (only inclusive/neutral; we use rewrites as bias-free)
    for sid in train_ids:
        if sid not in rewrites or sid not in labels:
            continue
        cats = labels[sid]
        if "Neutral" in cats:
            orig = sentences.get(sid, "")
            exemplars.append({"original": orig, "rewrite": rewrites[sid]})
        elif any(c in REPAIR_CATEGORIES for c in cats):
            # Use ground-truth rewrite as bias-free exemplar for retrieval
            exemplars.append({"original": sentences.get(sid, ""), "rewrite": rewrites[sid]})

    build_index(exemplars, index_path, embedding_model_name=model_name)
    print(f"Index built at {index_path} with {len(exemplars)} exemplars.")


if __name__ == "__main__":
    main()
