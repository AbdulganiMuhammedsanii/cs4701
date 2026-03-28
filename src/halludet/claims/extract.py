"""Segment LLM (or any) text into atomic candidate factual claims."""

from __future__ import annotations

import re

# Lazy punkt download
_punkt_ready = False


def _ensure_punkt() -> None:
    global _punkt_ready
    if _punkt_ready:
        return
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    _punkt_ready = True


def _split_sentences_nltk(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize

    _ensure_punkt()
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def _split_sentences_regex(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _is_candidate_claim(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 12:
        return False
    lower = s.lower()
    if lower.startswith(("however,", "therefore,", "in conclusion,", "note:")):
        return False
    if re.match(r"^(yes|no|ok|sure|thanks)\b", lower):
        return False
    return True


def extract_claims(
    response: str,
    *,
    use_nltk: bool = True,
    filter_candidates: bool = True,
) -> list[str]:
    """
    Split `response` into sentences; optionally drop obvious non-factual lines.

    This is intentionally lightweight (segmentation + simple rules), matching
    the project spec; it is not a full OpenIE / dependency-parse pipeline.
    """
    text = response.strip()
    if not text:
        return []
    try:
        sents = _split_sentences_nltk(text) if use_nltk else _split_sentences_regex(text)
    except Exception:
        sents = _split_sentences_regex(text)
    if not filter_candidates:
        return sents
    return [s for s in sents if _is_candidate_claim(s)]
