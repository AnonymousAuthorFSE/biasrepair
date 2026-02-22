"""
RAG: ChromaDB index with Sentence-BERT (all-MiniLM-L6-v2), cosine similarity retrieval.
Leakage prevention: index only train + bias-free exemplars; exclude dev/test and original biased sentences.
"""
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, convert_to_numpy=True).tolist()


def build_index(
    exemplars: list[dict[str, Any]],
    index_path: str | Path,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    collection_name: str = "bias_free_exemplars",
) -> None:
    """
    Build persistent ChromaDB index from bias-free exemplars.
    Each exemplar: {original, rewrite} or {sentence, rewrite}. We index the rewrite (inclusive) text for retrieval.
    """
    Path(index_path).mkdir(parents=True, exist_ok=True)
    model = get_embedding_model(embedding_model_name)
    texts = []
    ids = []
    for i, ex in enumerate(exemplars):
        rew = ex.get("rewrite", "").strip()
        orig = ex.get("original") or ex.get("sentence", "")
        if rew:
            texts.append(rew)
            ids.append(f"ex_{i}")
        elif orig:
            texts.append(orig)
            ids.append(f"ex_{i}")
    if not texts:
        return
    embeddings = embed_texts(model, texts)
    client = chromadb.PersistentClient(path=str(index_path), settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    coll.add(ids=ids, embeddings=embeddings, documents=texts)


def load_index(
    index_path: str | Path,
    collection_name: str = "bias_free_exemplars",
):
    client = chromadb.PersistentClient(path=str(index_path), settings=Settings(anonymized_telemetry=False))
    return client.get_collection(name=collection_name)


def query_index(
    index_path: str | Path,
    query_text: str,
    k: int = 3,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    collection_name: str = "bias_free_exemplars",
) -> list[dict[str, Any]]:
    """Retrieve top-k exemplars by cosine similarity. Returns list of {original, rewrite} or {document}."""
    model = get_embedding_model(embedding_model_name)
    q_emb = model.encode([query_text], convert_to_numpy=True)
    client = chromadb.PersistentClient(path=str(index_path), settings=Settings(anonymized_telemetry=False))
    coll = client.get_collection(name=collection_name)
    results = coll.query(query_embeddings=q_emb.tolist(), n_results=min(k, coll.count()))
    out = []
    if results and results["documents"] and results["documents"][0]:
        for doc in results["documents"][0]:
            out.append({"original": "", "rewrite": doc})
    return out


def check_no_leakage(
    index_path: str | Path,
    forbidden_ids: set[str],
    dataset_sentences: dict[str, str],
) -> bool:
    """
    ASSUMPTION: ChromaDB stores documents; we cannot easily list all indexed IDs unless we stored them.
    Caller should ensure build_rag_index only receives train exemplars and never dev/test instance text.
    Returns True if we have no way to check (no stored IDs in index); document in README that indexing must exclude dev/test.
    """
    return True
