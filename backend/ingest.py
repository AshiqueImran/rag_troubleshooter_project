"""
ingest.py
Loads knowledge from configured source, chunks, embeds, and saves FAISS index.

Run: python backend/ingest.py

Which methods get built is controlled by CHUNKING_METHODS in .env.
Which method the server uses is controlled by CHUNKING_METHOD in .env.

Each method saves to its own subfolder:
    vector_store/fixed/
    vector_store/paragraph/
    vector_store/semantic/

Switch methods by changing CHUNKING_METHOD in .env and restarting the server.
Only re-run ingest when documents change.
"""

import os, sys, pickle, logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(__file__))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Source loaders ────────────────────────────────────────────────────────────

def _load_documents() -> list[str]:
    from pypdf import PdfReader
    texts = []
    folder = config.DOCUMENTS_DIR
    if not os.path.exists(folder):
        log.warning(f"Documents folder not found: {folder}")
        return texts
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        ext  = filename.lower().split(".")[-1]
        if ext == "pdf":
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            log.info(f"Loaded PDF: {filename}")
        elif ext in ("txt", "md"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
            log.info(f"Loaded: {filename}")
    return texts


def _load_database() -> list[str]:
    from sqlalchemy import create_engine, text
    if not config.DB_URL or not config.DB_QUERY:
        log.warning("DB_URL or DB_QUERY not set.")
        return []
    engine = create_engine(config.DB_URL)
    texts  = []
    with engine.connect() as conn:
        result  = conn.execute(text(config.DB_QUERY))
        columns = result.keys()
        for row in result:
            texts.append("  ".join(f"{col}: {val}" for col, val in zip(columns, row)))
    log.info(f"Loaded {len(texts)} rows from database.")
    return texts


def _load_web() -> list[str]:
    import requests
    from bs4 import BeautifulSoup
    texts = []
    for url in config.WEB_URLS:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            texts.append(soup.get_text(separator="\n", strip=True))
            log.info(f"Loaded: {url}")
        except Exception as e:
            log.error(f"Failed {url}: {e}")
    return texts


# ── Chunking methods ──────────────────────────────────────────────────────────

def _chunk_fixed(texts: list[str]) -> list[str]:
    chunks = []
    for text in texts:
        words = text.split()
        start = 0
        while start < len(words):
            chunks.append(" ".join(words[start:start + config.CHUNK_SIZE]))
            start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    log.info(f"Fixed chunking: {len(chunks)} chunks")
    return chunks


def _chunk_paragraph(texts: list[str]) -> list[str]:
    chunks = []
    for text in texts:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        buffer = ""
        for para in paragraphs:
            if len(buffer.split()) < 10:
                buffer = (buffer + " " + para).strip()
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = para
        if buffer:
            chunks.append(buffer)
    log.info(f"Paragraph chunking: {len(chunks)} chunks")
    return chunks


def _chunk_semantic(texts: list[str], model: SentenceTransformer) -> list[str]:
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
    except ImportError:
        log.warning("nltk not installed — falling back to fixed chunking.")
        return _chunk_fixed(texts)

    from sklearn.metrics.pairwise import cosine_similarity
    chunks = []
    for text in texts:
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            chunks.append(text)
            continue
        vectors       = model.encode(sentences, convert_to_numpy=True)
        current_chunk = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = cosine_similarity([vectors[i - 1]], [vectors[i]])[0][0]
            if sim < config.SEMANTIC_THRESHOLD:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    log.info(f"Semantic chunking: {len(chunks)} chunks (threshold={config.SEMANTIC_THRESHOLD})")
    return chunks


def _chunk(texts: list[str], method: str, model: SentenceTransformer) -> list[str]:
    if method == "paragraph":
        return _chunk_paragraph(texts)
    elif method == "semantic":
        return _chunk_semantic(texts, model)
    else:
        if method != "fixed":
            log.warning(f"Unknown method '{method}' — using fixed.")
        return _chunk_fixed(texts)


# ── Embed + index ─────────────────────────────────────────────────────────────

def _embed_and_index(chunks: list[str], model: SentenceTransformer) -> faiss.IndexFlatL2:
    log.info(f"Embedding {len(chunks)} chunks...")
    vectors = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    index   = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    log.info(f"FAISS index: {index.ntotal} vectors")
    return index


def _save(index, chunks, path: str):
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    log.info(f"Saved {len(chunks)} chunks → {path}/")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def build_index(raw: list[str], model: SentenceTransformer, method: str):
    """Build and save index for one method. Raw text and model passed in to
    avoid reloading documents and model for every method in a multi-build."""
    log.info(f"--- Building: method={method} ---")
    chunks = _chunk(raw, method, model)
    index  = _embed_and_index(chunks, model)
    _save(index, chunks, config.get_vector_store_path(method))
    log.info(f"Done: {method} → {config.get_vector_store_path(method)}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Source: {config.SOURCE_TYPE}")
    log.info(f"Methods to build: {config.CHUNKING_METHODS}")

    # Load documents once — reused for all methods
    loaders = {"documents": _load_documents, "database": _load_database, "web": _load_web}
    loader  = loaders.get(config.SOURCE_TYPE)
    if not loader:
        raise ValueError(f"Unknown SOURCE_TYPE: '{config.SOURCE_TYPE}'")

    raw = loader()
    if not raw:
        log.warning("No content loaded.")
        sys.exit(1)

    # Load embedding model once — reused for all methods
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    # Build an index for each method listed in CHUNKING_METHODS
    for method in config.CHUNKING_METHODS:
        build_index(raw, model, method)

    log.info("Ingestion complete.")
    log.info(f"Active method (CHUNKING_METHOD): {config.CHUNKING_METHOD}")
    log.info(f"Server will load from: {config.get_vector_store_path()}")