# Artifact Checklist

All artifacts required for replication.

| Artifact | Location | Notes |
|----------|----------|--------|
| README | `README.md` | Reproduction steps, what to add |
| Artifact checklist | `ARTIFACT_CHECKLIST.md` | This file |
| Reproducibility guide | `REPRODUCIBILITY.md` | Seeds, determinism, hardware, leakage |
| License | `LICENSE` | MIT |
| Citation | `CITATION.cff` | BibTeX-style |
| Dependencies | `requirements.txt` | Python deps |
| Config – full system | `configs/full_system.yaml` | RAG + self-consistency + reflection |
| Config – w/o RAG | `configs/wout_rag.yaml` | Random inclusive exemplars (same count) |
| Config – w/o self-consistency | `configs/wout_self_consistency.yaml` | Single-pass generation |
| Config – w/o reflection | `configs/wout_reflection.yaml` | No reflection gate |
| Config – zero-shot | `configs/zero_shot.yaml` | n=1, T=0.2 |
| Config – GPT-4o zero-shot | `configs/gpt4o_zero_shot.yaml` | Wrapper; key via env |
| Prompt skeleton | `prompts/prompt_skeleton.txt` | One-sentence output, no explanation |
| Category: Generic Pronouns | `prompts/categories/GenericPronouns.txt` | |
| Category: Exclusionary Terms | `prompts/categories/ExclusionaryTerms.txt` | |
| Category: Stereotyping Bias | `prompts/categories/StereotypingBias.txt` | |
| Category: Sexism | `prompts/categories/Sexism.txt` | |
| Category: Semantic Bias | `prompts/categories/SemanticBias.txt` | |
| Dataset schema | `data/dataset/README.md` | consolidated_sentences, labels, ground_truth_rewrites |
| Consolidated sentences | `data/dataset/consolidated_sentences.jsonl` | User-provided |
| Labels multilabel | `data/dataset/labels_multilabel.jsonl` | User-provided |
| Ground truth rewrites | `data/dataset/ground_truth_rewrites.jsonl` | User-provided |
| Train IDs | `data/splits/train_ids.txt` | User-provided |
| Dev IDs | `data/splits/dev_ids.txt` | User-provided |
| Test IDs | `data/splits/test_ids.txt` | User-provided |
| Exemplars schema | `data/exemplars/README.md` | bias_free_exemplars |
| Bias-free exemplars | `data/exemplars/bias_free_exemplars.jsonl` | User-provided |
| Sample data | `data/sample/` | Synthetic; run without private data |
| Package: io | `biasrepair/io.py` | JSONL, splits, schema |
| Package: prompts | `biasrepair/prompts.py` | Skeleton + category + exemplar injection |
| Package: RAG | `biasrepair/rag.py` | ChromaDB, MiniLM, leakage checks |
| Package: generator (LLaMA) | `biasrepair/generator_llama.py` | LLaMA 3.1–8B-Instruct |
| Package: generator (GPT-4o) | `biasrepair/generator_gpt4o.py` | Stub; key via env |
| Package: self_consistency | `biasrepair/self_consistency.py` | n candidates, filter, select, tie-break |
| Package: reflection | `biasrepair/reflection.py` | Residual bias + drift + retry + flag |
| Package: pipeline | `biasrepair/pipeline.py` | 5-stage orchestrator |
| Package: metrics | `biasrepair/metrics.py` | EM, BLEU, cosine sim |
| Package: bootstrap | `biasrepair/bootstrap.py` | Paired bootstrap, p-value |
| CLI: run experiment | `biasrepair/cli/run_experiment.py` | Single config run |
| CLI: run ablations | `biasrepair/cli/run_ablations.py` | All ablation configs |
| CLI: build RAG index | `biasrepair/cli/build_rag_index.py` | Persistent ChromaDB from train exemplars |
| Descriptive stats | `scripts/descriptive_stats.py` | Counts per category (→ docs/descriptive_stats.md) |
| Manual eval guidelines | `manual_eval/guidelines.md` | 4 criteria, sampling, blinding, adjudication |
| Rating sheet template | `manual_eval/rating_sheet_template.csv` | |
| Generate rating sheet | `manual_eval/generate_rating_sheet.py` | From predictions.jsonl |
| Compute kappa | `manual_eval/compute_kappa.py` | Cohen's kappa from two CSVs |

