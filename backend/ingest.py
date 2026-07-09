"""
ingest.py
Loads knowledge from configured source, chunks, embeds, and saves FAISS index.
Run once before starting server: python backend/ingest.py

Chunking methods (set CHUNKING_METHOD in .env):
  fixed     → split every N words with overlap (default)
  paragraph → split on blank lines, preserves natural sections
  semantic  → split on topic boundaries using embedding similarity
"""

import os, pickle, logging, sys
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

def _chunk_fixed(texts: list[str], size: int, overlap: int) -> list[str]:
    """
    Split every N words with overlap between chunks.
    Simple and fast. Works for any document type.
    Best when document structure is unknown or inconsistent.
    """
    chunks = []
    for text in texts:
        words = text.split()
        start = 0
        while start < len(words):
            chunks.append(" ".join(words[start:start + size]))
            start += size - overlap
    return chunks


def _chunk_paragraph(texts: list[str]) -> list[str]:
    """
    Split on blank lines (double newlines).
    Each paragraph becomes one chunk.
    Preserves natural document sections — great for structured manuals,
    FAQs, and troubleshooting guides where each block is a complete idea.

    Very short paragraphs (under 10 words) are merged with the next one
    to avoid creating nearly-empty chunks from headings or labels.
    """
    chunks = []
    for text in texts:
        # split on one or more blank lines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        merged = []
        buffer = ""
        for para in paragraphs:
            if len(buffer.split()) < 10:
                # buffer too short — merge with next paragraph
                buffer = (buffer + " " + para).strip()
            else:
                if buffer:
                    merged.append(buffer)
                buffer = para
        if buffer:
            merged.append(buffer)
        chunks.extend(merged)
    log.info(f"Paragraph chunking: {len(chunks)} chunks")
    return chunks


def _chunk_semantic(texts: list[str], model: SentenceTransformer, threshold: float) -> list[str]:
    """
    Split on topic boundaries detected by embedding similarity.
    Embeds every sentence, measures cosine similarity between consecutive
    sentences, and splits where similarity drops below the threshold.

    This keeps semantically related content together regardless of
    paragraph breaks or word count — the most accurate chunking method.

    threshold: float between 0 and 1. Lower = fewer, larger chunks.
                                       Higher = more, smaller chunks.
                0.5 is a good default for most documents.
    """
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
    except ImportError:
        log.warning("nltk not installed. Falling back to fixed chunking.")
        return _chunk_fixed(texts, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    from sklearn.metrics.pairwise import cosine_similarity

    chunks = []

    for text in texts:
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            chunks.append(text)
            continue

        # embed all sentences at once — faster than one by one
        vectors = model.encode(sentences, convert_to_numpy=True)

        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([vectors[i - 1]], [vectors[i]])[0][0]

            if sim < threshold:
                # topic boundary detected — save current chunk, start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                # same topic — keep adding to current chunk
                current_chunk.append(sentences[i])

        # save the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

    log.info(f"Semantic chunking: {len(chunks)} chunks (threshold={threshold})")
    return chunks


def _chunk(texts: list[str], model: SentenceTransformer) -> list[str]:
    """
    Router — picks chunking method from CHUNKING_METHOD in .env.
    All methods return list[str]. Everything downstream is identical.
    """
    method = config.CHUNKING_METHOD
    log.info(f"Chunking method: {method}")

    if method == "paragraph":
        return _chunk_paragraph(texts)

    elif method == "semantic":
        return _chunk_semantic(texts, model, threshold=config.SEMANTIC_THRESHOLD)

    else:
        # default: fixed
        if method != "fixed":
            log.warning(f"Unknown CHUNKING_METHOD '{method}' — using fixed.")
        return _chunk_fixed(texts, config.CHUNK_SIZE, config.CHUNK_OVERLAP)


# ── Embed + index ─────────────────────────────────────────────────────────────

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


# ── Entry point ───────────────────────────────────────────────────────────────

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

    # load model once — reused for both semantic chunking and embedding
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    chunks = _chunk(raw, model)
    log.info(f"{len(chunks)} chunks from {len(raw)} source blocks.")

    index = _embed_and_index(chunks, model)
    _save(index, chunks)
    log.info("Ingestion complete.")


if __name__ == "__main__":
    build_index()