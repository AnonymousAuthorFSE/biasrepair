# BiasRepair: Mitigating Gender Bias in Software Engineering Course Materials

Replication package for the paper **"Mitigating Gender Bias in Software Engineering Course Materials Using Prompt Engineering and LLaMA"** (FSE 2025).

## Contents

- **Pipeline**: Taxonomy-driven prompting, optional RAG (ChromaDB + Sentence-BERT all-MiniLM-L6-v2), LLaMA 3.1–8B-Instruct generation, self-consistency selection (n=5, T=0.7), and reflection gate (retry budget 2, manual-review flag).
- **Configs**: Full system; w/o RAG (replaced by same number of random inclusive exemplars); w/o self-consistency; w/o reflection; zero-shot baseline (n=1, T=0.2); GPT-4o zero-shot baseline (env API key).
- **Evaluation**: Exact Match, BLEU (sacrebleu), cosine similarity; paired bootstrap resampling (p < 0.05).
- **Manual evaluation**: Guidelines (4 Likert criteria, 100 sentences / 20 per category), rating sheet template, and scripts to produce rating sheets from predictions and to compute Cohen's kappa.
- **Data layout**: Schema and expected paths for the consolidated dataset, splits, and bias-free exemplars; a small sample under `data/sample/` for local runs.

## Setup

```bash
pip install -r requirements.txt
```

## Data placement

For full replication, add:

- **Dataset**: `data/dataset/consolidated_sentences.jsonl`, `labels_multilabel.jsonl`, `ground_truth_rewrites.jsonl` (see `data/dataset/README.md`).
- **Splits**: `data/splits/train_ids.txt`, `dev_ids.txt`, `test_ids.txt` (one ID per line). Dev is used only for hyperparameter tuning; dev and test are excluded from the RAG index.
- **Exemplars**: `data/exemplars/bias_free_exemplars.jsonl` (bias-free / inclusive sentences).

If `data/dataset/` is empty, the CLI falls back to `data/sample/` so experiments can be run locally without real data.

## Build RAG index

After adding train data and exemplars:

```bash
python -m biasrepair.cli.build_rag_index --config configs/full_system.yaml
```

The index is built only from bias-free exemplars and bias-free rewrites from the training split (no dev/test, no original biased sentences).

## Run experiments

**Single config (e.g. zero-shot on sample data):**

```bash
python -m biasrepair.cli.run_experiment --config configs/zero_shot.yaml --output_dir runs
```

With a limit for a quick test:

```bash
python -m biasrepair.cli.run_experiment --config configs/zero_shot.yaml --output_dir runs --limit 5
```

**All ablations (full, w/o RAG, w/o self-consistency, w/o reflection, zero-shot):**

```bash
python -m biasrepair.cli.run_ablations --config_dir configs --output_dir runs
```

Alias:

```bash
python -m biasrepair.cli.run_ablations --configs_dir configs --output_dir runs
```

With a limit:

```bash
python -m biasrepair.cli.run_ablations --config_dir configs --output_dir runs --limit 10
```

**GPT-4o zero-shot baseline** (requires `OPENAI_API_KEY` in the environment):

```bash
python -m biasrepair.cli.run_experiment --config configs/gpt4o_zero_shot.yaml --output_dir runs --limit 50
```

## Outputs

Each run writes to `runs/<timestamp>/<config_name>/`:

- `predictions.jsonl` — id, category, original, rewrite, reference, flags, chosen_candidate, optional candidates
- `metrics.json` — exact_match, bleu, cosine_similarity
- `config_used.yaml` — config used
- `provenance.json` — git hash, package versions

When running ablations, the latest timestamp folder also gets:

- `ablations_summary.json` — metrics per config
- `ablations_summary.md` — table (BLEU, cosine sim, EM)
- `bootstrap.json` — p-values for full vs zero-shot (when both runs exist)

## Descriptive statistics

To generate counts per category (and per source if a `source` field exists):

```bash
python scripts/descriptive_stats.py --dataset_dir data/dataset --output docs/descriptive_stats.md
```

## Manual evaluation

See `manual_eval/guidelines.md` for the four criteria, sampling (100 sentences, 20 per category), blinding, and adjudication. Generate a rating sheet from a run:

```bash
python manual_eval/generate_rating_sheet.py runs/<timestamp>/full_system/predictions.jsonl --output manual_eval/rating_sheet.csv --per_category 20
```

Compute Cohen's kappa from two annotator CSVs:

```bash
python manual_eval/compute_kappa.py manual_eval/ratings_annotator1.csv manual_eval/ratings_annotator2.csv
```

## Model access

- **LLaMA 3.1–8B-Instruct**: e.g. Hugging Face `meta-llama/Llama-3.1-8B-Instruct`; set `HF_TOKEN` or config as needed.
- **GPT-4o baseline**: set `OPENAI_API_KEY` in the environment; the key is not stored in the repository.

## Reproducibility and artifacts

- **REPRODUCIBILITY.md** — seeds, determinism, hardware, leakage prevention.
- **ARTIFACT_CHECKLIST.md** — list of artifacts and locations.
