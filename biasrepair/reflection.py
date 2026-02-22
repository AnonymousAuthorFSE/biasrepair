"""
Reflection gate: post-hoc validation for (i) residual bias, (ii) semantic drift.
Retry budget = 2; if still fails, flag for manual review.
Semantic drift uses configurable length-ratio bounds and negation heuristics.
"""
from typing import Any

from biasrepair.bias_markers import has_residual_bias

DEFAULT_MIN_LENGTH_RATIO = 0.2
DEFAULT_MAX_LENGTH_RATIO = 5.0


def check_residual_bias(rewrite: str, category: str, config_markers: dict[str, list[str]] | None = None) -> bool:
    """True if rewrite passes (no residual markers)."""
    return not has_residual_bias(rewrite, category, config_markers)


def check_semantic_drift(
    original: str,
    rewrite: str,
    min_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    max_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
) -> bool:
    """
    Conservative heuristics for obvious semantic drift: length ratio and negation consistency.
    Returns True if no obvious drift detected.
    """
    if not rewrite or not original:
        return False
    orig_lower = original.lower()
    rew_lower = rewrite.lower()
    ratio = len(rewrite.split()) / (len(original.split()) + 1e-9)
    if ratio < min_ratio or ratio > max_ratio:
        return False
    neg_orig = "not " in orig_lower or "n't " in orig_lower or "no " in orig_lower
    neg_rew = "not " in rew_lower or "n't " in rew_lower or "no " in rew_lower
    if neg_orig != neg_rew and (("not " in orig_lower and "not " not in rew_lower) or ("not " in rew_lower and "not " not in orig_lower)):
        return False
    return True


def reflection_gate(
    original: str,
    rewrite: str,
    category: str,
    config_markers: dict[str, list[str]] | None = None,
    min_length_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
) -> tuple[bool, str]:
    """Returns (passed, reason). Passed if no residual bias and no obvious semantic drift."""
    if not check_residual_bias(rewrite, category, config_markers):
        return False, "residual_bias"
    if not check_semantic_drift(original, rewrite, min_ratio=min_length_ratio, max_ratio=max_length_ratio):
        return False, "semantic_drift"
    return True, ""
