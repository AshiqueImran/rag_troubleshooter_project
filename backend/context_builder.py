"""
context_builder.py
Takes raw retrieved chunks and assembles a clean context string
ready to inject into the LLM prompt.

Responsibilities:
  - Deduplicate chunks with high text overlap
  - Trim to stay within token budget
  - Format with clear separators so the LLM can reason over sections
"""

import logging

log = logging.getLogger(__name__)

# Rough token budget for context.
# GPT-4o supports 128k tokens but we keep context tight
# so the LLM focuses on signal, not noise.
MAX_CONTEXT_WORDS = 1200  # ~1500 tokens, leaves room for prompt + response


def _is_duplicate(a: str, b: str, threshold: float = 0.7) -> bool:
    """
    Check if two chunks overlap significantly.
    Uses word-set Jaccard similarity: |intersection| / |union|
    threshold=0.7 means 70% of words in common → considered duplicate.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) >= threshold


def _deduplicate(chunks: list[str]) -> list[str]:
    """
    Remove chunks that are too similar to an already-kept chunk.
    Iterates in order — earlier chunks (higher ranked) are kept.
    """
    kept = []
    for chunk in chunks:
        if not any(_is_duplicate(chunk, k) for k in kept):
            kept.append(chunk)
    return kept


def _trim_to_budget(chunks: list[str], max_words: int) -> list[str]:
    """
    Keep chunks until we hit the word budget.
    Drops lower-ranked chunks first (they're at the end of the list).
    """
    trimmed = []
    total = 0
    for chunk in chunks:
        word_count = len(chunk.split())
        if total + word_count > max_words:
            break
        trimmed.append(chunk)
        total += word_count
    return trimmed


def build_context(chunks: list[str]) -> str:
    """
    Deduplicate, trim, and format chunks into a single context string.

    Output format:
        [1] chunk text here...

        [2] next chunk text here...

    Numbered sections help the LLM reference specific parts if needed.
    Returns empty string if no usable chunks remain.
    """
    if not chunks:
        return ""

    deduped = _deduplicate(chunks)
    trimmed = _trim_to_budget(deduped, MAX_CONTEXT_WORDS)

    log.debug(
        f"Context: {len(chunks)} raw → {len(deduped)} deduped "
        f"→ {len(trimmed)} after budget trim"
    )

    sections = [f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(trimmed)]
    return "\n\n".join(sections)
