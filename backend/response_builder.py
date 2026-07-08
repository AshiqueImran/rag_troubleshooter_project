"""
response_builder.py
Assembles the final API response from LLM output + retrieval metadata.

Keeps response shape consistent regardless of what the LLM returns.
The frontend can always expect the same fields.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TroubleshootingResponse:
    """
    The structured response returned by POST /ask.

    answer      : the LLM's answer
    confidence  : high | medium | low
    sources     : which context sections were used
    follow_up   : optional clarifying question from LLM
    intent      : parsed intent from the query (troubleshoot / how_to / explain / unknown)
    chunks_used : number of context chunks passed to LLM
    """
    answer:      str
    confidence:  str
    sources:     list[int]
    follow_up:   Optional[str]
    intent:      str
    chunks_used: int


def build(llm_result: dict, intent: str, chunks_used: int) -> dict:
    """
    Merge LLM output with retrieval metadata into a clean response dict.
    Always returns all fields — missing LLM fields get safe defaults.
    """
    response = TroubleshootingResponse(
        answer      = llm_result.get("answer", "No answer returned."),
        confidence  = llm_result.get("confidence", "low"),
        sources     = llm_result.get("sources", []),
        follow_up   = llm_result.get("follow_up", None),
        intent      = intent,
        chunks_used = chunks_used,
    )
    return asdict(response)
