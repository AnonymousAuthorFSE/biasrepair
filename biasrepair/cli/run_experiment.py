"""
Run one experiment from a config; write predictions, metrics, config copy, and provenance to runs/<timestamp>/<config_name>/.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def get_provenance() -> dict:
    out = {}
    try:
        out["git_hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        out["git_hash"] = None
    for pkg in ["torch", "transformers", "sentence_transformers", "chromadb", "sacrebleu"]:
        try:
            m = __import__(pkg)
            out[pkg] = getattr(m, "__version__", "?")
        except Exception:
            out[pkg] = None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", default="runs", help="Output directory for runs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test instances")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        config.setdefault("data", {})["limit"] = args.limit

    config_name = config_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp / config_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths
    data_cfg = config.get("data", {})
    dataset_dir = Path(data_cfg.get("dataset_dir", "data/dataset"))
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    exemplars_path = Path(data_cfg.get("exemplars_path", "data/exemplars/bias_free_exemplars.jsonl"))
    limit = data_cfg.get("limit")
    prompts_dir = Path(config.get("prompts_dir", "prompts"))
    index_path = config.get("rag", {}).get("index_path")
    if index_path:
        index_path = Path(index_path)

    from biasrepair.io import get_eval_instances, load_exemplars
    from biasrepair.rag import get_embedding_model

    instances = get_eval_instances(dataset_dir, splits_dir, "test", limit=limit)
    if not instances and dataset_dir == Path("data/dataset"):
        dataset_dir = Path("data/sample")
        splits_dir = Path("data/sample/splits")
        exemplars_path = Path("data/sample/exemplars.jsonl")
        instances = get_eval_instances(dataset_dir, splits_dir, "test", limit=limit)
    if not instances:
        print("No test instances found. Check data/dataset and data/splits.", file=sys.stderr)
        sys.exit(1)
    exemplars = load_exemplars(exemplars_path) if exemplars_path.exists() else []

    if config.get("generator", {}).get("backend") == "openai":
        from biasrepair.prompts import build_prompt
        from biasrepair.generator_gpt4o import generate as gen_gpt4o
        predictions = []
        for inst in instances:
            prompt = build_prompt(inst["sentence"], inst["category"], prompts_dir, exemplars=[])
            rew = gen_gpt4o(prompt, model_name=config["generator"].get("model_name", "gpt-4o"), max_new_tokens=config["generator"].get("max_new_tokens", 256), temperature=config["generator"].get("temperature", 0.2))
            predictions.append({"id": inst["id"], "category": inst["category"], "original": inst["sentence"], "rewrite": rew, "reference": inst.get("reference", ""), "flags": [], "chosen_candidate": rew})
    else:
        from biasrepair.pipeline import run_pipeline
        predictions = run_pipeline(instances, config, prompts_dir, exemplars, index_path, seed=config.get("seed", 42))

    # Save predictions
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Metrics
    refs = [p["reference"] for p in predictions]
    preds = [p["rewrite"] for p in predictions]
    emb_model = get_embedding_model(config.get("rag", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    from biasrepair.metrics import exact_match, bleu_score, cosine_similarity
    em = sum(exact_match(p, r) for p, r in zip(preds, refs)) / len(preds) if preds else 0
    bleu = sum(bleu_score(p, r) for p, r in zip(preds, refs)) / len(preds) if preds else 0
    cos = sum(cosine_similarity(p, r, emb_model) for p, r in zip(preds, refs)) / len(preds) if preds else 0
    metrics = {"exact_match": em, "bleu": bleu, "cosine_similarity": cos}
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    with open(run_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(get_provenance(), f, indent=2)

    print(f"Run saved to {run_dir}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
