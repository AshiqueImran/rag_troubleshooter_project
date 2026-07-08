"""
query_parser.py
Lightweight intent detection and keyword extraction. No LLM, no network call.
"""

import re, logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

INTENT_PATTERNS = {
    "troubleshoot": ["not working","doesn't work","won't","can't","error","fail","broken",
                     "issue","problem","keeps","crash","stuck","freeze","dropping","fix"],
    "how_to":       ["how do i","how to","steps to","guide","configure","set up","install",
                     "enable","disable","create","connect"],
    "explain":      ["what is","what are","explain","describe","why does","tell me about"],
}

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but","my","me",
    "i","we","our","your","this","that","with","from","by","be","are","was","were","have",
    "has","do","does","did","so","if","when","where","how","what","why","who","which",
    "there","their","they","can","will","would","should","could","may","might","just",
    "also","about","up","out","any","all","not","no","very","keep","keeps","every","get",
}


@dataclass
class ParsedQuery:
    original:   str
    cleaned:    str
    intent:     str
    keywords:   list[str]
    search_str: str


def _clean(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", " ", text)
    return re.sub(r"\s+", " ", text)


def _detect_intent(cleaned):
    for intent, triggers in INTENT_PATTERNS.items():
        if any(t in cleaned for t in triggers):
            return intent
    return "unknown"


def _extract_keywords(cleaned):
    seen, kws = set(), []
    for w in cleaned.split():
        if w not in STOPWORDS and len(w) > 2 and w not in seen:
            kws.append(w); seen.add(w)
    return kws


def parse(query: str) -> ParsedQuery:
    if not query or not query.strip():
        return ParsedQuery("", "", "unknown", [], "")
    cleaned  = _clean(query)
    keywords = _extract_keywords(cleaned)
    return ParsedQuery(
        original   = query,
        cleaned    = cleaned,
        intent     = _detect_intent(cleaned),
        keywords   = keywords,
        search_str = " ".join(keywords) if keywords else cleaned,
    )
