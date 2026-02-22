"""
Paired bootstrap resampling for significance testing (e.g. full vs zero-shot; p < 0.05).
"""
import numpy as np
from typing import Any, Callable


def paired_bootstrap(
    preds_a: list[str],
    refs: list[str],
    preds_b: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute metric for A and B, then bootstrap distribution of (A - B).
    Returns (metric_a, metric_b, p_value) where p_value is proportion of resamples where B >= A (one-sided) or 2*min(p, 1-p) for two-sided.
    Paper: improvements of full over baseline are significant at p < 0.05; we do two-sided: reject if p < alpha.
    """
    assert len(preds_a) == len(refs) == len(preds_b)
    n = len(refs)
    rng = np.random.default_rng(seed)
    score_a = metric_fn(preds_a, refs)
    score_b = metric_fn(preds_b, refs)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a_sub = [preds_a[i] for i in idx]
        b_sub = [preds_b[i] for i in idx]
        r_sub = [refs[i] for i in idx]
        sa = metric_fn(a_sub, r_sub)
        sb = metric_fn(b_sub, r_sub)
        diffs.append(sa - sb)
    diffs = np.array(diffs)
    # Two-sided p-value: proportion where |A - B| (bootstrap) <= 0 when we observe A > B
    # P(observed difference could be due to chance): proportion of resamples where B >= A
    p_value = float(np.mean(diffs <= 0))
    p_value = 2 * min(p_value, 1 - p_value)
    return score_a, score_b, p_value


def run_bootstrap_comparisons(
    run_dirs: dict[str, str],
    refs: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """run_dirs: config_name -> path to predictions.jsonl. Load predictions, compare baseline vs full."""
    import json
    from pathlib import Path
    results = {}
    preds_by_config = {}
    for name, path in run_dirs.items():
        rows = []
        with open(Path(path) / "predictions.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        preds_by_config[name] = [r.get("rewrite", "") for r in rows]
    # Align by id if needed; here assume same order
    for name, preds in preds_by_config.items():
        if len(preds) != len(refs):
            continue
        results[name] = {
            "metric": metric_fn(preds, refs),
            "n": len(preds),
        }
    return results
