# Reproducibility

## Seeds and Determinism

- **Random seed**: Configurable via `seed` in YAML (default: 42). Used for bootstrap resampling, split shuffles, and random exemplar sampling (e.g. w/o RAG ablation).
- **LLaMA**: Decoding is stochastic. We set `torch.manual_seed(seed)` and use `do_sample=True` with `temperature` from config. Exact token sequences may vary across runs/hardware; report mean/std over multiple runs if needed.
- **RAG**: ChromaDB and Sentence-BERT (all-MiniLM-L6-v2) are deterministic for the same index. Index is built only from **train** (and bias-free exemplars); **dev and test instances are never indexed** (leakage prevention).
- **Bootstrap**: Paired bootstrap resampling uses the config seed; number of resamples and alpha (e.g. 1000, 0.05) are in config.

## Hardware

- **LLaMA 3.1–8B-Instruct**: ~16GB GPU VRAM recommended (e.g. single A100 or 2× consumer GPUs with `device_map`). CPU fallback supported but slow.
- **RAG**: ChromaDB and all-MiniLM-L6-v2 run on CPU by default; GPU optional for the embedding model.

## Logging and Outputs

Each run writes under `runs/<timestamp>/<config_name>/`:

- **predictions.jsonl**: `id`, `category`, `original`, `rewrite`, `reference`, `flags` (e.g. `manual_review`), `chosen_candidate`, optional `candidates`.
- **metrics.json**: EM, BLEU, cosine similarity (and per-category if applicable).
- **bootstrap.json**: Paired bootstrap p-values when enabled (e.g. full vs zero-shot).
- **config_used.yaml**: Copy of the config used.
- **provenance.json**: Git hash (if available), package versions.

## Leakage Prevention

- The retrieval index is populated **only** with:
  - Bias-free exemplars (inclusive SE sentences, neutral guidelines, bias-free rewrites from the **training** split).
- **Excluded from index**: All dev and test instances; original biased sentences.
- Dev split is used **only** for hyperparameter tuning (e.g. k ∈ {3,5,8}). Final evaluation is on the test split only; no evaluation instances are used for tuning or indexing.

## Ablations

- **Full**: RAG (k=3) + self-consistency (n=5, T=0.7) + reflection (retry 2, manual-review flag).
- **w/o RAG**: Same pipeline but retrieval replaced by the **same number** of randomly sampled inclusive exemplars (controls prompt length and example exposure).
- **w/o self-consistency**: Single candidate (n=1); no multi-candidate selection.
- **w/o reflection**: Reflection gate disabled; no retry, no manual-review flag.
- **Zero-shot**: No RAG, single shot (n=1, T=0.2), no reflection.
- **GPT-4o zero-shot**: Same category-specific prompts as LLaMA zero-shot; no RAG/SC/reflection; API key via env.

## Expected Output Ranges (from paper)

- Full system: BLEU ~0.71, cosine sim ~0.89, EM ~0.42.
- Zero-shot baseline: BLEU ~0.59, cosine sim ~0.83, EM ~0.32.
- Improvements of full over zero-shot are statistically significant (p < 0.05) for BLEU, cosine similarity, and exact match.
