"""
InnSight — NLP Engine (Layer 1: Aspect Extraction + Sentiment Analysis)
------------------------------------------------------------------------
Input:  raw review text
Output: list of {aspect, sentiment (-1..1), confidence, emotion_label}

Rule-based ABSA: explainable, offline, no API dependency.
"""

import re
from .lexicon import (
    ASPECT_KEYWORDS, POSITIVE_WORDS, NEGATIVE_WORDS,
    NEGATION_WORDS, INTENSIFIERS,
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
WORD_RE = re.compile(r"[a-zA-Z']+")

_ASPECT_PATTERNS = None


def _build_aspect_patterns():
    """Compile word-boundary regex per keyword so short words like 'ac' or
    'tv' don't match as substrings inside unrelated words (e.g. 'spacious')."""
    global _ASPECT_PATTERNS
    if _ASPECT_PATTERNS is not None:
        return _ASPECT_PATTERNS
    patterns = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        compiled = []
        for kw in keywords:
            escaped = re.escape(kw)
            pattern = re.compile(r'(?<![a-zA-Z])' + escaped + r'(?![a-zA-Z])')
            compiled.append(pattern)
        patterns[aspect] = compiled
    _ASPECT_PATTERNS = patterns
    return patterns


def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _score_sentence_sentiment(sentence: str):
    """
    Word-level polarity scoring with a simple negation window (flips polarity
    of the next sentiment word within 3 tokens) and intensifier boosting.
    """
    tokens = WORD_RE.findall(sentence.lower())
    score = 0.0
    hits = 0
    negate_window = 0
    boost = 1.0

    for tok in tokens:
        if tok in NEGATION_WORDS:
            negate_window = 3
            continue
        if tok in INTENSIFIERS:
            boost = INTENSIFIERS[tok]
            continue

        polarity = None
        if tok in POSITIVE_WORDS:
            polarity = POSITIVE_WORDS[tok]
        elif tok in NEGATIVE_WORDS:
            polarity = NEGATIVE_WORDS[tok]

        if polarity is not None:
            applied = polarity * boost
            if negate_window > 0:
                applied = -applied * 0.8
            score += applied
            hits += 1
            boost = 1.0

        if negate_window > 0:
            negate_window -= 1

    return score, hits


def _match_aspects(sentence: str):
    """Return set of aspects whose keywords appear in this sentence."""
    lower = sentence.lower()
    matched = set()
    patterns = _build_aspect_patterns()
    for aspect, compiled_list in patterns.items():
        for pat in compiled_list:
            if pat.search(lower):
                matched.add(aspect)
                break
    return matched


def extract_aspects(review_text: str):
    """
    Main entry point. Returns a list of dicts:
    [{aspect, sentiment, confidence, emotion_label, mention_count}, ...]
    """
    sentences = split_sentences(review_text)
    if not sentences:
        return []

    aspect_scores = {}
    for sentence in sentences:
        aspects_in_sentence = _match_aspects(sentence)
        if not aspects_in_sentence:
            continue
        score, hits = _score_sentence_sentiment(sentence)
        for aspect in aspects_in_sentence:
            aspect_scores.setdefault(aspect, []).append((score, hits))

    results = []
    for aspect, entries in aspect_scores.items():
        total_score = sum(s for s, h in entries)
        total_hits = sum(h for s, h in entries)
        n_sentences = len(entries)

        if total_hits > 0:
            avg = total_score / max(total_hits, 1)
            sentiment = max(-1.0, min(1.0, avg / 2.0))
        else:
            sentiment = 0.0

        confidence = min(0.95, 0.35 + 0.15 * total_hits + 0.05 * n_sentences)

        if sentiment > 0.25:
            emotion = "satisfied"
        elif sentiment < -0.25:
            emotion = "frustrated"
        else:
            emotion = "neutral"

        results.append({
            "aspect": aspect,
            "sentiment": round(sentiment, 3),
            "confidence": round(confidence, 3),
            "emotion_label": emotion,
            "mention_count": n_sentences,
        })

    return results


if __name__ == "__main__":
    sample = (
        "The staff was very helpful and the room was clean and spacious. "
        "However the food was not good and breakfast was overpriced. "
        "Location is great, close to the metro station."
    )
    import json
    print(json.dumps(extract_aspects(sample), indent=2))