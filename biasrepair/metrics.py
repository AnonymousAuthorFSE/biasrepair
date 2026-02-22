"""
Evaluation metrics: Exact Match, BLEU (sacrebleu), cosine similarity (MiniLM).
"""
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def bleu_score(pred: str, ref: str) -> float:
    try:
        import sacrebleu
        bleu = sacrebleu.sentence_bleu(pred.strip(), [ref.strip()])
        return bleu.score / 100.0
    except Exception:
        return 0.0


def cosine_similarity(pred: str, ref: str, model: SentenceTransformer | None = None) -> float:
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode([pred.strip(), ref.strip()])
    a, b = np.array(embs[0]), np.array(embs[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_metrics(
    predictions: list[str],
    references: list[str],
    embedding_model: SentenceTransformer | None = None,
) -> dict[str, float]:
    if not predictions or not references or len(predictions) != len(references):
        return {"exact_match": 0.0, "bleu": 0.0, "cosine_similarity": 0.0}
    em_sum = sum(exact_match(p, r) for p, r in zip(predictions, references))
    bleu_sum = sum(bleu_score(p, r) for p, r in zip(predictions, references))
    if embedding_model is None:
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cos_sum = sum(cosine_similarity(p, r, embedding_model) for p, r in zip(predictions, references))
    n = len(predictions)
    return {
        "exact_match": em_sum / n,
        "bleu": bleu_sum / n,
        "cosine_similarity": cos_sum / n,
    }
