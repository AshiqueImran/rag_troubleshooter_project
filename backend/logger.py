"""
logger.py
Append-only request/response logging to a .jsonl file.
Each line is a self-contained JSON object — easy to grep, parse, or load into pandas.

Captures: timestamp, query, intent, confidence, latency, chunk count.
Does NOT log full context or full answer — keeps log file lean.
"""

import json
import logging
import os
from datetime import datetime, timezone

import sys
sys.path.append(os.path.dirname(__file__))
import config

log = logging.getLogger(__name__)


def log_request(
    query:      str,
    intent:     str,
    confidence: str,
    chunks_used: int,
    latency_ms: float,
    error:      str | None = None,
) -> None:
    """
    Append one log entry to the JSONL log file.
    Creates the log file and parent directory if they don't exist.
    Silently swallows logging errors — a log failure must never crash the server.
    """
    try:
        os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)

        entry = {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "query":       query[:200],   # truncate very long queries
            "intent":      intent,
            "confidence":  confidence,
            "chunks_used": chunks_used,
            "latency_ms":  round(latency_ms, 2),
        }
        if error:
            entry["error"] = error

        with open(config.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    except Exception as e:
        log.warning(f"Failed to write log entry: {e}")
