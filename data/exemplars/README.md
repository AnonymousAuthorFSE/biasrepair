# Bias-free exemplars for RAG

Place **bias_free_exemplars.jsonl** here. Each line: JSON with `sentence` (or `original`) and `rewrite`. These are inclusive/bias-free sentences from SE materials or neutral guidelines. The RAG index is built from this file plus bias-free rewrites from the **training** split only (dev/test excluded).
