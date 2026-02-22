"""
Category-specific lexical cues for residual-bias filtering (paper: filter candidates that contain residual markers).
Default marker lists can be overridden or extended via the config `bias_markers` (per-category lists).
"""
from typing import Any

# Lowercase tokens/phrases that suggest residual bias per category (non-exhaustive)
DEFAULT_BIAS_MARKERS: dict[str, list[str]] = {
    "GenericPronouns": [" he ", " his ", " him ", " she ", " her ", " himself ", " herself ", " he.", " she.", " his.", " her."],
    "ExclusionaryTerms": ["chairman", "businessman", "manpower", "cameraman", "congressman", "fireman", "policeman", "mailman", "mankind"],
    "StereotypingBias": ["wives to support", "mothers working", "fathers must", "women are", "men are", "girls are", "boys are"],
    "Sexism": ["incompetent", "for a girl", "for a woman", "like a man", "women can't", "men can't"],
    "SemanticBias": ["cookie", "woman's tongue", "kill a man", "old sayings"],
}


def get_markers_for_category(category: str, config_markers: dict[str, list[str]] | None = None) -> list[str]:
    combined = list(DEFAULT_BIAS_MARKERS.get(category, []))
    if config_markers and category in config_markers:
        combined = list(set(combined) | set(config_markers[category]))
    return combined


def has_residual_bias(text: str, category: str, config_markers: dict[str, list[str]] | None = None) -> bool:
    """True if text contains any residual bias marker for the given category."""
    if not text:
        return False
    lower = text.lower()
    for m in get_markers_for_category(category, config_markers):
        if m.lower() in lower:
            return True
    return False
