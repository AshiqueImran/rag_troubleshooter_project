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

---

## How Retrieval Works

### Why Two Search Systems?

Most RAG systems use only vector (semantic) search. This project uses two systems in parallel  and for good reason.

**FAISS (semantic search)** converts text into vectors that capture meaning. It finds relevant chunks even when the user's words differ from the document's words:
- Query: `"internet keeps cutting out"` → finds chunk about `"intermittent connectivity loss"` 
- Query: `"ERR_CONNECTION_TIMED_OUT"` → finds weakly  error codes have no semantic meaning 

**BM25 (keyword search)** scores chunks by how frequently and how uniquely the query terms appear. It excels at exact matches:
- Query: `"ERR_CONNECTION_TIMED_OUT"` → finds the exact error code instantly 
- Query: `"internet keeps cutting out"` → misses chunks that use different wording 

Neither system alone is reliable. Together they cover each other's blind spots.

---

### Why Not Just Add the Raw Scores?

The two systems produce fundamentally incompatible numbers:

```
FAISS → L2 distances   (lower = more relevant)   range: 0.0 – 2.0
BM25  → relevance weights (higher = more relevant) range: 0.0 – 50+
```

Adding them directly causes **scale domination**  BM25's larger numbers completely overwhelm FAISS scores:

```
Chunk A: FAISS=0.002 + BM25=14.3 = 14.302  ← BM25 decided this, FAISS contributed nothing
Chunk B: FAISS=0.9   + BM25=0.1  =  1.000  ← penalised unfairly despite strong semantic match
```

FAISS has effectively been thrown away. The result is no better than BM25 alone.

---

### Reciprocal Rank Fusion (RRF)

RRF solves the scale problem by discarding raw scores entirely and working with **rank positions** instead. Both systems rank chunks 1st, 2nd, 3rd  ranks are always on the same scale regardless of the underlying scoring method.

**The equation:**

```
RRF score = 1 / (rank_in_FAISS + k) + 1 / (rank_in_BM25 + k)
```

**Why this equation specifically:**

The `1/rank` formula gives diminishing returns as rank increases  1st place scores higher than 2nd, but the gap shrinks as ranks grow. This reflects the real-world intuition that the difference between rank 1 and rank 2 matters more than the difference between rank 40 and rank 41.

**Why `+k` (where k=60):**

Without k, the gap between rank 1 and rank 2 is enormous:
```
1/1 = 1.000  vs  1/2 = 0.500  → rank 1 is 2× better than rank 2
```

A chunk ranked #1 by FAISS but #50 by BM25 would dominate everything  rewarding a single-system fluke over consistent relevance.

With k=60, nearby ranks produce very similar scores:
```
1/61 = 0.01639  vs  1/62 = 0.01613  → gap is tiny
```

A chunk must perform **consistently well in both systems** to win  not spectacularly in one. k=60 is the value empirically validated in the original RRF paper (Cormack et al., 2009) across a wide range of retrieval tasks.

---

### Concrete Example

Query: `"screen shows no signal"`

| Chunk | FAISS rank | BM25 rank | RRF score | Final rank |
|-------|-----------|-----------|-----------|------------|
| A  Monitor cable troubleshooting | 1 | 4 | 1/61 + 1/64 = **0.0320** | 2nd |
| B  Display signal loss resolution | 2 | 1 | 1/62 + 1/61 = **0.0326** | **1st**  |
| C  GPU seating guide | 3 | 6 | 1/63 + 1/66 = **0.0310** | 3rd |

**Chunk B wins**  it ranked well in both systems. Chunk A was ranked #1 by FAISS alone but #4 by BM25, so it finishes second. A chunk that is consistently good beats one that is excellent in only one dimension.

This is the core principle: **hybrid retrieval rewards consistent relevance, not single-system dominance.**

---

## Knowledge Sources

Set `SOURCE_TYPE` in `.env`  the ingestion layer handles the rest:

| Source | Config | Use case |
|--------|--------|----------|
| `documents` | Drop files in `data/documents/` | PDF, TXT, MD manuals |
| `database` | Set `DB_URL` + `DB_QUERY` | SQL knowledge base |
| `web` | Set `WEB_URLS` | Scraped documentation pages |

---

## Chunking Methods

Two `.env` variables control chunking:

| Variable | Purpose |
|----------|---------|
| `CHUNKING_METHODS` | Comma-separated list of methods ingest builds indexes for |
| `CHUNKING_METHOD` | Single method the server loads at runtime |

| Method | Best for |
|--------|----------|
| `fixed` | Default. Works for any document type. Splits every N words with overlap |
| `paragraph` | Structured manuals, FAQs, troubleshooting guides. Splits on blank lines |
| `semantic` | Most accurate. Splits on topic boundaries using embedding similarity |

Each method gets its own subfolder in `vector_store/`:

```
vector_store/
├── fixed/
├── paragraph/
└── semantic/
```

**Build all indexes at once:**
```bash
# .env: CHUNKING_METHODS=fixed,paragraph,semantic
python backend/ingest.py
```

**Switch active method instantly  no rebuild needed:**
```bash
# .env: change CHUNKING_METHOD=paragraph
# restart server  loads vector_store/paragraph/ automatically
uvicorn backend.main:app --reload
```

**Only re-run ingest when documents change.**

You can also trigger a live reload without restarting via the admin endpoint:
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
# Edit .env  set GROQ_API_KEY, SOURCE_TYPE, CHUNKING_METHOD, CHUNKING_METHODS
```

### 3. Add knowledge

For `SOURCE_TYPE=documents`: drop `.pdf`, `.txt`, or `.md` files into `data/documents/`.

Sample documents are included in `data/documents/` for testing.

### 4. Build the index

```bash
python backend/ingest.py
```

Builds a FAISS index for each method listed in `CHUNKING_METHODS`.
Re-run only when documents change.

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
│   ├── fixed/              # Index built with fixed chunking
│   ├── paragraph/          # Index built with paragraph chunking
│   └── semantic/           # Index built with semantic chunking
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

- **FastAPI**  async Python web framework
- **FAISS**  vector similarity search (Meta)
- **sentence-transformers**  local text embeddings (all-MiniLM-L6-v2)
- **rank-bm25**  BM25 keyword search
- **Groq (Llama 3.1)**  LLM with function calling (OpenAI SDK compatible)
- **nltk + scikit-learn**  sentence tokenisation and cosine similarity for semantic chunking
- **SQLAlchemy**  database source connector
- **BeautifulSoup4**  web scraping source connector

---

## Author

**Md Ashique Imran**
MSc Applied Computer Science, University of Winnipeg
[linkedin.com/in/ashique-imran](https://linkedin.com/in/ashique-imran) · [github.com/ashiqueimran](https://github.com/ashiqueimran)
