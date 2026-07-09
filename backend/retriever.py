"""
retriever.py
Hybrid retrieval: FAISS (semantic) + BM25 (keyword), merged via Reciprocal Rank Fusion.
"""

import os, pickle, logging, sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

sys.path.append(os.path.dirname(__file__))
import config

log = logging.getLogger(__name__)


def load_index():
    index_path  = os.path.join(config.VECTOR_STORE_PATH, "index.faiss")
    chunks_path = os.path.join(config.VECTOR_STORE_PATH, "chunks.pkl")
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Vector store not found. Run `python backend/ingest.py` or  `python backend/ingest.py --all` first.")
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    model    = SentenceTransformer(config.EMBEDDING_MODEL)
    tokenised = [c.lower().split() for c in chunks]
    bm25     = BM25Okapi(tokenised)
    log.info(f"Retriever ready: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks, model, bm25


def _faiss_search(query, index, model, top_k):
    vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(vec, top_k)
    return [int(i) for i in indices[0] if i != -1]


def _bm25_search(query, bm25, top_k):
    scores = bm25.get_scores(query.lower().split())
    return [int(i) for i in np.argsort(scores)[-top_k:][::-1]]


def _rrf_merge(faiss_idx, bm25_idx, k=60):
    scores = {}
    for rank, idx in enumerate(faiss_idx):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    for rank, idx in enumerate(bm25_idx):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def retrieve(query, index, chunks, model, bm25, top_k=config.TOP_K):
    merged = _rrf_merge(_faiss_search(query, index, model, top_k), _bm25_search(query, bm25, top_k))
    return [chunks[i] for i in merged[:top_k] if i < len(chunks)]
