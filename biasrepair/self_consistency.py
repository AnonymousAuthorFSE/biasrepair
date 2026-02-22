"""
Self-consistency: n-candidate sampling, filter by residual bias markers, select by highest cosine similarity to original; tie-break: concise and fluent.
"""
from typing import Any

from biasrepair.bias_markers import has_residual_bias


def filter_candidates(
    candidates: list[str],
    category: str,
    config_markers: dict[str, list[str]] | None = None,
) -> list[str]:
    """Keep only candidates that pass residual-bias check (no category-specific lexical cues)."""
    return [c for c in candidates if c and not has_residual_bias(c, category, config_markers)]


def select_best(
    candidates: list[str],
    original_sentence: str,
    embeddings_fn,
    category: str,
    config_markers: dict[str, list[str]] | None = None,
) -> str:
    """
    Filter candidates by residual-bias check, then select highest cosine similarity to original.
    Tie-break: prefer shorter (concise) and fluent.
    """
    if not candidates:
        return ""
    filtered = filter_candidates(candidates, category, config_markers)
    if not filtered:
        return candidates[0]
    if len(filtered) == 1:
        return filtered[0]
    texts = [original_sentence] + filtered
    embs = embeddings_fn(texts)
    orig_emb = embs[0]
    cand_embs = embs[1:]
    import numpy as np
    orig_norm = np.array(orig_emb) / (np.linalg.norm(orig_emb) + 1e-9)
    sims = [np.dot(orig_norm, np.array(e) / (np.linalg.norm(e) + 1e-9)) for e in cand_embs]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    ties = [i for i, s in enumerate(sims) if abs(s - best_sim) < 1e-6]
    if len(ties) > 1:
        ties.sort(key=lambda i: (len(filtered[i]), filtered[i]))
        best_idx = ties[0]
    return filtered[best_idx]
