"""
main.py
FastAPI application — the entry point for the RAG Troubleshooting Assistant.

Routes:
  GET  /          → serves frontend index.html
  GET  /health    → health check (used by CI/CD and load balancers)
  POST /ask       → main query endpoint
  POST /ingest    → trigger re-ingestion (admin only, protected by API key)

Run:
  uvicorn backend.main:app --reload
"""

import time
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(__file__))

import config
from retriever       import load_index, retrieve
from query_parser    import parse
from context_builder import build_context
from llm             import ask
from response_builder import build
from logger          import log_request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── App state ─────────────────────────────────────────────────────────────────
# Loaded once at startup, shared across all requests.

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index at startup. FastAPI lifespan replaces @app.on_event."""
    log.info("Loading retrieval index...")
    try:
        index, chunks, model, bm25 = load_index()
        app_state["index"]  = index
        app_state["chunks"] = chunks
        app_state["model"]  = model
        app_state["bm25"]   = bm25
        log.info("Index loaded. Server ready.")
    except FileNotFoundError as e:
        log.error(str(e))
        log.error("Run `python backend/ingest.py` to build the index first.")
    yield
    app_state.clear()


app = FastAPI(
    title="RAG Troubleshooting Assistant",
    description="Hybrid retrieval (FAISS + BM25) + GPT-4o function calling.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ── Request / response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    source_type: str | None = None   # override SOURCE_TYPE if provided


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the frontend HTML."""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "RAG Troubleshooting Assistant API. See /docs for usage."}


@app.get("/health")
def health():
    """Health check — returns index status."""
    return {
        "status": "ok",
        "index_loaded": "index" in app_state,
        "chunks": len(app_state.get("chunks", [])),
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Main query endpoint.
    Runs the full RAG pipeline: parse → retrieve → context → LLM → response.
    """
    if "index" not in app_state:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run ingest.py first."
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start = time.time()

    # 1. Parse query
    parsed = parse(request.query)

    # 2. Retrieve relevant chunks
    chunks = retrieve(
        query  = parsed.search_str,
        index  = app_state["index"],
        chunks = app_state["chunks"],
        model  = app_state["model"],
        bm25   = app_state["bm25"],
    )

    # 3. Build context string
    context = build_context(chunks)

    # 4. Ask LLM
    llm_result = ask(query=parsed.original, context=context)

    # 5. Build structured response
    response = build(
        llm_result  = llm_result,
        intent      = parsed.intent,
        chunks_used = len(chunks),
    )

    # 6. Log
    latency_ms = (time.time() - start) * 1000
    log_request(
        query       = request.query,
        intent      = parsed.intent,
        confidence  = response["confidence"],
        chunks_used = len(chunks),
        latency_ms  = latency_ms,
        error       = llm_result.get("error"),
    )

    return response


@app.post("/ingest")
def trigger_ingest(
    request: IngestRequest,
    x_admin_key: str = Header(default=None),
):
    """
    Trigger re-ingestion of the knowledge source.
    Protected by X-Admin-Key header matching OPENAI_API_KEY (simple guard).
    After ingestion, reloads the index into app_state.
    """
    if x_admin_key != config.OPENAI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key.")

    # Allow source_type override from request
    if request.source_type:
        os.environ["SOURCE_TYPE"] = request.source_type

    from ingest import build_index
    build_index()

    # Reload index into memory
    index, chunks, model, bm25 = load_index()
    app_state["index"]  = index
    app_state["chunks"] = chunks
    app_state["model"]  = model
    app_state["bm25"]   = bm25

    return {"status": "ingestion complete", "chunks": len(chunks)}
