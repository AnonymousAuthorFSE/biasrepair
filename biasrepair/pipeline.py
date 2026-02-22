"""
BiasRepair pipeline: (i) taxonomy-driven prompting, (ii) optional RAG, (iii) candidate generation,
(iv) self-consistency selection, (v) reflection gate. Orchestrates from config.
"""
import json
import random
from pathlib import Path
from typing import Any

from biasrepair import io as bio
from biasrepair.prompts import build_prompt
from biasrepair.rag import get_embedding_model, query_index
from biasrepair.generator_llama import load_llama, generate as gen_llama
from biasrepair.self_consistency import select_best
from biasrepair.reflection import reflection_gate
from biasrepair.bias_markers import get_markers_for_category


def run_pipeline(
    instances: list[dict[str, Any]],
    config: dict[str, Any],
    prompts_dir: str | Path,
    exemplars: list[dict[str, Any]],
    index_path: str | Path | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Run full pipeline on instances. Each instance: {id, sentence, category, reference}.
    Returns list of {id, category, original, rewrite, reference, flags, chosen_candidate, candidates?}.
    """
    random.seed(seed)
    rag_enabled = config.get("rag", {}).get("enabled", False)
    sc_enabled = config.get("self_consistency", {}).get("enabled", False)
    reflection_enabled = config.get("reflection", {}).get("enabled", False)
    retry_budget = config.get("reflection", {}).get("retry_budget", 2)
    flag_manual = config.get("reflection", {}).get("flag_manual_review_on_failure", True)
    k_rag = config.get("rag", {}).get("k", 3)
    n_candidates = config.get("self_consistency", {}).get("n", 5)
    temp = config.get("generator", {}).get("temperature", 0.7)
    max_tokens = config.get("generator", {}).get("max_new_tokens", 256)
    model_name = config.get("generator", {}).get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
    emb_model_name = config.get("rag", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    random_exemplars_count = config.get("rag", {}).get("random_exemplars_count", 0)

    # Load generator (LLaMA)
    model, tokenizer = load_llama(model_name, seed=seed)
    emb_model = get_embedding_model(emb_model_name)

    def embed_fn(texts):
        return emb_model.encode(texts, convert_to_numpy=True).tolist()

    config_markers = config.get("bias_markers")  # optional per-category marker lists

    results = []
    for inst in instances:
        sid = inst["id"]
        original = inst["sentence"]
        category = inst["category"]
        reference = inst.get("reference", "")
        flags = []

        # Exemplars: RAG or random same count
        if rag_enabled and index_path and Path(index_path).exists():
            exemplar_list = query_index(index_path, original, k=k_rag, embedding_model_name=emb_model_name)
        elif random_exemplars_count and exemplars:
            exemplar_list = random.sample(exemplars, min(random_exemplars_count, len(exemplars)))
        else:
            exemplar_list = []

        prompt = build_prompt(original, category, prompts_dir, exemplar_list)
        candidates = []

        for attempt in range(1 + retry_budget if reflection_enabled else 1):
            if sc_enabled and n_candidates > 1:
                for _ in range(n_candidates):
                    c = gen_llama(model, tokenizer, prompt, max_new_tokens=max_tokens, temperature=temp, do_sample=True, seed=seed + attempt * 1000 + _)
                    if c:
                        candidates.append(c)
                chosen = select_best(candidates, original, embed_fn, category, config_markers) if candidates else ""
            else:
                chosen = gen_llama(model, tokenizer, prompt, max_new_tokens=max_tokens, temperature=temp, do_sample=True, seed=seed + attempt)
                candidates = [chosen] if chosen else []

            if not reflection_enabled:
                break
            passed, reason = reflection_gate(original, chosen, category, config_markers)
            if passed:
                break
            if attempt <= retry_budget - 1:
                prompt = prompt + "\n\n[Stricter constraint: Ensure no residual bias and no change in meaning.]\n\nRewrite again (one sentence only):"
            else:
                if flag_manual:
                    flags.append("manual_review")
                break

        results.append({
            "id": sid,
            "category": category,
            "original": original,
            "rewrite": chosen,
            "reference": reference,
            "flags": flags,
            "chosen_candidate": chosen,
        })
        if sc_enabled and len(candidates) > 1:
            results[-1]["candidates"] = candidates
    return results
