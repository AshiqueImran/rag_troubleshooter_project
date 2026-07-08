import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM — Groq (active) ───────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str   = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# ── LLM — OpenAI (alternative) ───────────────────────────────────────────────
# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-4o")

MAX_TOKENS: int      = int(os.getenv("MAX_TOKENS", "1024"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K: int           = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE: int      = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int   = int(os.getenv("CHUNK_OVERLAP", "64"))
SOURCE_TYPE: str     = os.getenv("SOURCE_TYPE", "documents")
DOCUMENTS_DIR: str   = os.getenv("DOCUMENTS_DIR", "data/documents")
DB_URL: str          = os.getenv("DB_URL", "")
DB_QUERY: str        = os.getenv("DB_QUERY", "")
WEB_URLS: list[str]  = [u.strip() for u in os.getenv("WEB_URLS", "").split(",") if u.strip()]
VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "vector_store")
LOG_PATH: str        = os.getenv("LOG_PATH", "logs/requests.jsonl")
HOST: str            = os.getenv("HOST", "0.0.0.0")
PORT: int            = int(os.getenv("PORT", "8000"))
