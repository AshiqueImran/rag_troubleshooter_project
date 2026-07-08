"""
ingest.py
Loads knowledge from configured source, chunks, embeds, and saves FAISS index.
Run once before starting server: python backend/ingest.py
"""

import os, pickle, logging, sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(__file__))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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


def _chunk(texts: list[str], size: int, overlap: int) -> list[str]:
    chunks = []
    for text in texts:
        words = text.split()
        start = 0
        while start < len(words):
            chunks.append(" ".join(words[start:start + size]))
            start += size - overlap
    return chunks


def _embed_and_index(chunks: list[str], model: SentenceTransformer) -> faiss.IndexFlatL2:
    log.info(f"Embedding {len(chunks)} chunks...")
    vectors = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    index   = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    log.info(f"FAISS index: {index.ntotal} vectors")
    return index


def _save(index, chunks):
    os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(config.VECTOR_STORE_PATH, "index.faiss"))
    with open(os.path.join(config.VECTOR_STORE_PATH, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    log.info(f"Saved {len(chunks)} chunks to {config.VECTOR_STORE_PATH}/")


def build_index():
    log.info(f"Source: {config.SOURCE_TYPE}")
    loaders = {"documents": _load_documents, "database": _load_database, "web": _load_web}
    loader  = loaders.get(config.SOURCE_TYPE)
    if not loader:
        raise ValueError(f"Unknown SOURCE_TYPE: '{config.SOURCE_TYPE}'")
    raw = loader()
    if not raw:
        log.warning("No content loaded.")
        return
    chunks = _chunk(raw, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    log.info(f"{len(chunks)} chunks from {len(raw)} source blocks.")
    model  = SentenceTransformer(config.EMBEDDING_MODEL)
    index  = _embed_and_index(chunks, model)
    _save(index, chunks)
    log.info("Ingestion complete.")


if __name__ == "__main__":
    build_index()
