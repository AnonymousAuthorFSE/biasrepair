from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np


def paired_bootstrap(
    preds_a: List[str],
    refs: List[str],
    preds_b: List[str],
    metric_fn: Callable[[List[str], List[str]], float],
    n_resamples: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:

    if not (len(preds_a) == len(refs) == len(preds_b)):
        raise ValueError("preds_a, preds_b, refs must have the same length")

    n = len(refs)
    rng = np.random.default_rng(seed)

    score_a = float(metric_fn(preds_a, refs))
    score_b = float(metric_fn(preds_b, refs))
    delta_obs = score_a - score_b

    deltas = np.empty(n_resamples, dtype=np.float64)

    for t in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a_sub = [preds_a[i] for i in idx]
        b_sub = [preds_b[i] for i in idx]
        r_sub = [refs[i] for i in idx]
        deltas[t] = float(metric_fn(a_sub, r_sub)) - float(metric_fn(b_sub, r_sub))

    p_value = float(np.mean(np.abs(deltas) >= abs(delta_obs)))

    alpha = 1.0 - ci
    ci_low = float(np.quantile(deltas, alpha / 2.0))
    ci_high = float(np.quantile(deltas, 1.0 - alpha / 2.0))

    return {
        "score_a": score_a,
        "score_b": score_b,
        "delta": float(delta_obs),
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": float(n),
        "n_resamples": float(n_resamples),
    }


def load_predictions_jsonl(run_dir: str | Path, rewrite_key: str = "rewrite") -> List[str]:
    run_dir = Path(run_dir)
    preds = []
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds.append(row.get(rewrite_key, ""))
    return preds


def run_paired_bootstrap_matrix(
    run_dirs: Dict[str, str | Path],
    refs: List[str],
    metric_fn: Callable[[List[str], List[str]], float],
    compare_to: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:

    preds_by = {name: load_predictions_jsonl(path) for name, path in run_dirs.items()}

    if compare_to not in preds_by:
        raise ValueError(f"compare_to='{compare_to}' not found in run_dirs")

    base = preds_by[compare_to]
    out: Dict[str, Any] = {}

    for name, preds in preds_by.items():
        if len(preds) != len(refs) or len(base) != len(refs):
            out[name] = {
                "error": "length_mismatch",
                "len_preds": len(preds),
                "len_base": len(base),
                "len_refs": len(refs),
            }
            continue

        if name == compare_to:
            out[name] = {
                "score": float(metric_fn(preds, refs)),
                "n": len(refs),
            }
            continue

        stats = paired_bootstrap(
            preds_a=preds,
            refs=refs,
            preds_b=base,
            metric_fn=metric_fn,
            n_resamples=n_resamples,
            seed=seed,
        )
        out[name] = stats

    return out
