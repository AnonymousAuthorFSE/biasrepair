# Splits (IDs only)

Place here:

- **train_ids.txt**: One sentence ID per line (used for RAG index and tuning; never use dev/test in index).
- **dev_ids.txt**: One sentence ID per line (used only for hyperparameter tuning, e.g. k ∈ {3,5,8}).
- **test_ids.txt**: One sentence ID per line (used only for final evaluation).

No evaluation instances are used during tuning or in the retrieval index (leakage prevention).
