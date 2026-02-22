# Dataset schema

Place the following files here.

- **consolidated_sentences.jsonl**: One JSON object per line. Fields: `id` (string), `sentence` (or `text`) (string).
- **labels_multilabel.jsonl**: One JSON object per line. Fields: `id` (string), `labels` (or `categories`) (array of strings). Allowed labels: `GenericPronouns`, `ExclusionaryTerms`, `StereotypingBias`, `Sexism`, `SemanticBias`, `Neutral`. Neutral is used only for exemplars; excluded from repair evaluation.
- **ground_truth_rewrites.jsonl**: One JSON object per line. Fields: `id` (string), `rewrite` (or `reference`) (string).

IDs must align across the three files. The consolidated corpus (from the paper) has 7,820 unique sentences with 11,868 total bias labels in multi-label format.
