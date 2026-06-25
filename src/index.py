"""FAISS Index — build and query the candidate ANN index.

Uses IndexFlatIP (inner product on L2-normalised vectors = cosine similarity).
For 100K vectors × 768 dims this is ~295MB in RAM — well within the 16GB budget.

If the dataset grows to 1M+, swap to IndexHNSWFlat for sub-linear query time.
Current design: exact search, ~200ms query on 100K on a single CPU core.
"""

from __future__ import annotations

import json
import logging

import faiss
import numpy as np

from src.config import EMBEDDING_DIM, FAISS_INDEX_PATH, CANDIDATE_IDS_PATH

logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray, candidate_ids: list[str]) -> faiss.Index:
    """Build a flat inner-product FAISS index from pre-computed embeddings."""
    assert embeddings.shape[1] == EMBEDDING_DIM, (
        f"Expected {EMBEDDING_DIM}-dim embeddings, got {embeddings.shape[1]}"
    )
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors", index.ntotal)
    return index


def save_index(index: faiss.Index, candidate_ids: list[str]) -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CANDIDATE_IDS_PATH, "w") as f:
        json.dump(candidate_ids, f)
    logger.info(
        "Saved FAISS index → %s  |  IDs → %s",
        FAISS_INDEX_PATH, CANDIDATE_IDS_PATH,
    )


def load_index() -> tuple[faiss.Index, list[str]]:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run: python rank.py --precompute --candidates <path>"
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CANDIDATE_IDS_PATH) as f:
        candidate_ids = json.load(f)
    logger.info("Loaded FAISS index (%d vectors)", index.ntotal)
    return index, candidate_ids


def query_index(
    index: faiss.Index,
    candidate_ids: list[str],
    jd_vector: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    """ANN search. Returns [(candidate_id, cosine_similarity), ...]."""
    jd_vec = jd_vector.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(jd_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(candidate_ids):
            continue
        results.append((candidate_ids[idx], float(dist)))
    return results


def hybrid_query_index(
    faiss_index: faiss.Index,
    bm25_index,
    candidate_ids: list[str],
    jd_vector: np.ndarray,
    query_text: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """Hybrid search combining FAISS dense results and BM25 sparse results using RRF.
    
    Returns [(candidate_id, cosine_similarity), ...] sorted by RRF rank.
    """
    jd_vec = jd_vector.astype(np.float32).reshape(1, -1)
    
    # 1. Query FAISS for a larger candidate set (top 5000) to get semantic scores
    dense_k = min(5000, len(candidate_ids))
    distances, indices = faiss_index.search(jd_vec, dense_k)
    
    dense_scores = {}
    dense_rank = {}
    for rank_idx, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(candidate_ids):
            continue
        cid = candidate_ids[idx]
        dense_scores[cid] = float(dist)
        if rank_idx < top_k:
            dense_rank[cid] = rank_idx + 1

    # 2. Query BM25 for top_k sparse results
    bm25_results = bm25_index.query(query_text, top_k=top_k)
    bm25_rank = {cid: rank_idx + 1 for rank_idx, (cid, _) in enumerate(bm25_results)}

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    all_candidates = set(dense_rank.keys()).union(bm25_rank.keys())
    
    for cid in all_candidates:
        score = 0.0
        if cid in dense_rank:
            score += 1.0 / (60.0 + dense_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (60.0 + bm25_rank[cid])
        rrf_scores[cid] = score

    # Sort candidates by RRF score descending
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Map back to the expected output format with their FAISS similarity score (defaulting to 0.40 if outside top 5000)
    results = []
    for cid, _ in sorted_candidates:
        sem_score = dense_scores.get(cid, 0.40)
        results.append((cid, sem_score))
        
    return results
