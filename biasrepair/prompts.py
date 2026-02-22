"""
Prompt loading, category guidance, and exemplar injection.
Skeleton enforces: output exactly one inclusive sentence; preserve technical meaning and pedagogical intent; no extra explanation.
"""
from pathlib import Path
from typing import Any

# Taxonomy categories (paper); Neutral is exemplars-only, not used as repair target in prompts
CATEGORIES = [
    "GenericPronouns",
    "ExclusionaryTerms",
    "StereotypingBias",
    "Sexism",
    "SemanticBias",
]

DEFAULT_GUIDANCE = "Rewrite the sentence to remove gender bias; preserve meaning and technical accuracy."


def load_skeleton(prompts_dir: str | Path) -> str:
    path = Path(prompts_dir) / "prompt_skeleton.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt skeleton not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_category_guidance(prompts_dir: str | Path, category: str) -> str:
    path = Path(prompts_dir) / "categories" / f"{category}.txt"
    if not path.exists():
        return DEFAULT_GUIDANCE
    return path.read_text(encoding="utf-8").strip()


def format_exemplars_block(exemplars: list[dict[str, Any]]) -> str:
    """Format exemplars for the prompt. Expects list of {original, rewrite} or {sentence, rewrite}."""
    if not exemplars:
        return "No exemplars provided."
    lines = []
    for i, ex in enumerate(exemplars, 1):
        orig = ex.get("original") or ex.get("sentence", "")
        rew = ex.get("rewrite", "")
        lines.append(f"Example {i}:\nOriginal: {orig}\nRewritten: {rew}")
    return "\n\n".join(lines)


def build_prompt(
    original_sentence: str,
    category: str,
    prompts_dir: str | Path,
    exemplars: list[dict[str, Any]] | None = None,
    skeleton: str | None = None,
) -> str:
    if skeleton is None:
        skeleton = load_skeleton(prompts_dir)
    guidance = load_category_guidance(prompts_dir, category)
    exemplars_block = format_exemplars_block(exemplars or [])
    return skeleton.format(
        category_guidance=guidance,
        exemplars_block=exemplars_block,
        original_sentence=original_sentence,
    )
