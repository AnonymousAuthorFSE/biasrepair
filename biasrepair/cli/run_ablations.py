"""
Run all ablation configs and write summary table (BLEU, cosine sim, EM) and bootstrap p-values.
"""
import argparse
import json
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs", help="Directory containing YAML configs")
    parser.add_argument("--configs_dir", default=None, help="Alias for --config_dir")
    parser.add_argument("--output_dir", default="runs", help="Output directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000, help="Bootstrap resamples for p-values")
    args = parser.parse_args()
    config_dir = Path(args.configs_dir or args.config_dir)

    configs = [
        "full_system.yaml",
        "wout_rag.yaml",
        "wout_self_consistency.yaml",
        "wout_reflection.yaml",
        "zero_shot.yaml",
    ]
    for cfg in configs:
        path = config_dir / cfg
        if not path.exists():
            continue
        cmd = [
            "python", "-m", "biasrepair.cli.run_experiment",
            "--config", str(path),
            "--output_dir", args.output_dir,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        subprocess.run(cmd, check=True)

    runs = Path(args.output_dir)
    if not runs.exists():
        return
    latest = sorted(runs.iterdir(), key=lambda p: p.name, reverse=True)[:1]
    if not latest:
        return
    base = latest[0]
    summary = []
    for cfg in configs:
        d = base / cfg.replace(".yaml", "")
        if not d.exists():
            continue
        mpath = d / "metrics.json"
        if mpath.exists():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            summary.append({"config": cfg, **m})

    with open(base / "ablations_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown table
    md_lines = ["| Configuration | BLEU | Cosine Sim. | Exact Match |", "|---------------|------|-------------|-------------|"]
    for s in summary:
        md_lines.append("| {} | {:.2f} | {:.2f} | {:.2f} |".format(
            s["config"].replace(".yaml", ""),
            s.get("bleu", 0),
            s.get("cosine_similarity", 0),
            s.get("exact_match", 0),
        ))
    with open(base / "ablations_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Bootstrap: full_system vs zero_shot
    full_dir = base / "full_system"
    zero_dir = base / "zero_shot"
    bootstrap_out = {}
    if full_dir.exists() and zero_dir.exists():
        preds_path_f = full_dir / "predictions.jsonl"
        preds_path_z = zero_dir / "predictions.jsonl"
        if preds_path_f.exists() and preds_path_z.exists():
            refs = []
            preds_full = []
            preds_zero = []
            with open(preds_path_f, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        refs.append(row.get("reference", ""))
                        preds_full.append(row.get("rewrite", ""))
            with open(preds_path_z, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        preds_zero.append(row.get("rewrite", ""))
            if len(refs) == len(preds_full) == len(preds_zero) and len(refs) > 0:
                from biasrepair.rag import get_embedding_model
                from biasrepair.metrics import exact_match, bleu_score, cosine_similarity
                emb = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
                def em_fn(p, r): return sum(exact_match(a, b) for a, b in zip(p, r)) / len(p)
                def bleu_fn(p, r): return sum(bleu_score(a, b) for a, b in zip(p, r)) / len(p)
                def cos_fn(p, r): return sum(cosine_similarity(a, b, emb) for a, b in zip(p, r)) / len(p)
                from biasrepair.bootstrap import paired_bootstrap
                seed = 42
                _, _, p_em = paired_bootstrap(preds_full, refs, preds_zero, em_fn, n_resamples=args.bootstrap_resamples, seed=seed)
                _, _, p_bleu = paired_bootstrap(preds_full, refs, preds_zero, bleu_fn, n_resamples=args.bootstrap_resamples, seed=seed)
                _, _, p_cos = paired_bootstrap(preds_full, refs, preds_zero, cos_fn, n_resamples=args.bootstrap_resamples, seed=seed)
                bootstrap_out = {"full_vs_zero_shot": {"exact_match_p": p_em, "bleu_p": p_bleu, "cosine_similarity_p": p_cos}}
                with open(base / "bootstrap.json", "w", encoding="utf-8") as f:
                    json.dump(bootstrap_out, f, indent=2)
                print("Bootstrap p-values (full vs zero-shot):", bootstrap_out)

    print("Ablations summary:", base / "ablations_summary.json")
    print("Table:", base / "ablations_summary.md")


if __name__ == "__main__":
    main()
