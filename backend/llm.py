"""
llm.py
Groq LLM wrapper with function calling.
Groq is free, fast, and OpenAI-SDK-compatible — minimal code difference.

To switch to OpenAI instead:
  1. pip install openai
  2. Uncomment the OpenAI lines below and comment out the Groq lines
  3. Set OPENAI_API_KEY in .env instead of GROQ_API_KEY
  4. Change model to "gpt-4o" in .env
"""

import json
import logging
import sys
import os

# ── LLM client — Groq (active) ────────────────────────────────────────────────
from groq import Groq
# ── LLM client — OpenAI (alternative) ────────────────────────────────────────
# from openai import OpenAI

sys.path.append(os.path.dirname(__file__))
import config

log = logging.getLogger(__name__)

# Active: Groq
client = Groq(api_key=config.GROQ_API_KEY)
# Alternative: OpenAI
# client = OpenAI(api_key=config.OPENAI_API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent troubleshooting assistant.
Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly.
Do not make up information. Be concise and direct.
Always respond by calling the submit_answer function."""

# ── Function schema ───────────────────────────────────────────────────────────
# Function calling forces the model to return structured JSON
# instead of free-form text — predictable, parseable, consistent.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer to the user's troubleshooting question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Clear, direct answer. Use numbered steps if resolution involves multiple actions.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Context section numbers used (e.g. [1, 3]).",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "high = context directly answers. medium = partial. low = loosely related.",
                    },
                    "follow_up": {
                        "type": "string",
                        "description": "Optional: one clarifying question if more info would help.",
                    },
                },
                "required": ["answer", "sources", "confidence"],
            },
        },
    }
]


# ── Public interface ──────────────────────────────────────────────────────────

def ask(query: str, context: str) -> dict:
    """
    Send query + context to Groq and return parsed function call result.
    Returns dict with keys: answer, sources, confidence, follow_up.
    Falls back to error dict if call fails.
    """
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            # Alternative OpenAI: model=config.OPENAI_MODEL,
            max_tokens=config.MAX_TOKENS,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "submit_answer"}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )

        tool_call = response.choices[0].message.tool_calls[0]
        result    = json.loads(tool_call.function.arguments)

        # Normalize sources — Llama sometimes returns "[1]" string instead of [1] array
        sources = result.get("sources", [])
        if isinstance(sources, str):
            try:
                result["sources"] = json.loads(sources)
            except Exception:
                result["sources"] = []

        log.debug(f"LLM | confidence={result.get('confidence')} | sources={result.get('sources')}")
        return result

    except Exception as e:
        log.error(f"LLM call failed: {e}")
        print(f"LLM ERROR: {e}")
        return {
            "answer":     "Sorry, I was unable to process your request. Please try again.",
            "sources":    [],
            "confidence": "low",
            "follow_up":  None,
            "error":      str(e),
        }
