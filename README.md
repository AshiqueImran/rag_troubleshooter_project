# RAG Troubleshooting Assistant

![CI](https://github.com/AshiqueImran/rag_troubleshooter_project/actions/workflows/ci.yml/badge.svg)

An intelligent troubleshooting assistant built on **Retrieval-Augmented Generation (RAG)**.

Hybrid retrieval (FAISS semantic search + BM25 keyword search) feeds relevant context to Groq (Llama 3.1) via function calling, returning structured answers with confidence scores.

---

## Architecture

```
User Query
  → Query Parser          (intent + keyword extraction)
  → Hybrid Retriever      (FAISS + BM25 → RRF merge)
  → Context Builder       (deduplicate + trim)
  → LLM (Groq Llama 3.1) (function calling → structured JSON)
  → Response Builder      (consistent API response)
  → Logger                (append-only JSONL metrics)
```

---

## Knowledge Sources

Set `SOURCE_TYPE` in `.env` — the ingestion layer handles the rest:

| Source | Config | Use case |
|--------|--------|----------|
| `documents` | Drop files in `data/documents/` | PDF, TXT, MD manuals |
| `database` | Set `DB_URL` + `DB_QUERY` | SQL knowledge base |
| `web` | Set `WEB_URLS` | Scraped documentation pages |

---

## Chunking Methods

Set `CHUNKING_METHOD` in `.env` to control how documents are split before indexing:

| Method | Config | Best for |
|--------|--------|----------|
| `fixed` | `CHUNK_SIZE` + `CHUNK_OVERLAP` | Default. Works for any document type |
| `paragraph` | — | Structured manuals, FAQs, troubleshooting guides |
| `semantic` | `SEMANTIC_THRESHOLD` | Most accurate. Splits on topic boundaries |

**Whenever you change `CHUNKING_METHOD`, re-run ingest to rebuild the index:**

```bash
python backend/ingest.py
```

Then restart the server, or trigger a live reload via the admin endpoint:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-Admin-Key: your-groq-api-key" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "documents"}'
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/AshiqueImran/rag_troubleshooter_project
cd rag_troubleshooter_project
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY, SOURCE_TYPE, and CHUNKING_METHOD at minimum
```

### 3. Add knowledge

For `SOURCE_TYPE=documents`: drop `.pdf`, `.txt`, or `.md` files into `data/documents/`.

Sample documents are included in `data/documents/` for testing.

### 4. Build the index

```bash
python backend/ingest.py
```

This creates `vector_store/index.faiss` and `vector_store/chunks.pkl`.

Re-run whenever you change documents or switch chunking method.

### 5. Run the server

```bash
uvicorn backend.main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## API

### `POST /ask`

```json
{ "query": "my wifi keeps disconnecting" }
```

Response:
```json
{
  "answer": "Intermittent wifi disconnections are often caused by channel interference...",
  "confidence": "high",
  "sources": [1, 2],
  "follow_up": "Is this happening on all devices or just one?",
  "intent": "troubleshoot",
  "chunks_used": 5
}
```

### `GET /health`

```json
{ "status": "ok", "index_loaded": true, "chunks": 42 }
```

### `POST /ingest` *(admin)*

Triggers re-ingestion and reloads the index without restarting the server.
Requires `X-Admin-Key` header.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-Admin-Key: your-groq-api-key" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "documents"}'
```

---

## Project Structure

```
rag_troubleshooter_project/
├── backend/
│   ├── main.py             # FastAPI app + routes
│   ├── ingest.py           # Loader + chunker (fixed/paragraph/semantic) + embedder
│   ├── retriever.py        # Hybrid FAISS + BM25 search
│   ├── query_parser.py     # Intent + keyword extraction
│   ├── context_builder.py  # Chunk dedup + context assembly
│   ├── llm.py              # Groq Llama 3.1 function calling
│   ├── response_builder.py # Structured JSON response
│   ├── logger.py           # Request/response logging
│   └── config.py           # ENV-driven configuration
├── data/
│   └── documents/          # Drop knowledge files here
├── vector_store/           # Auto-created by ingest.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── logs/
│   └── requests.jsonl      # Auto-created at runtime
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
├── requirements.txt
├── .env.example
└── README.md
```

---

## CI/CD

GitHub Actions runs automatically on every push to `master`:

- Syntax check all Python files
- Build vector index from sample documents
- Start the FastAPI server
- Hit `/health` and confirm index is loaded

```yaml
- name: Health check
  run: curl --fail http://localhost:8000/health
```

---

## Tech Stack

- **FastAPI** — async Python web framework
- **FAISS** — vector similarity search (Meta)
- **sentence-transformers** — local text embeddings (all-MiniLM-L6-v2)
- **rank-bm25** — BM25 keyword search
- **Groq (Llama 3.1)** — LLM with function calling (OpenAI SDK compatible)
- **nltk + scikit-learn** — sentence tokenisation and cosine similarity for semantic chunking
- **SQLAlchemy** — database source connector
- **BeautifulSoup4** — web scraping source connector

---

## Author

**Md Ashique Imran**
MSc Applied Computer Science, University of Winnipeg
[linkedin.com/in/ashique-imran](https://linkedin.com/in/ashique-imran) · [github.com/ashiqueimran](https://github.com/ashiqueimran)