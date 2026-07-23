"""
InnSight — Priority Matcher (freeform "what do you want" search)
---------------------------------------------------------------------
Lets a traveler type what they care about in plain language (e.g. "quiet
hotel near the metro with good breakfast") and ranks hotels accordingly.

Design note: this reuses the SAME aspect keyword lexicon that powers review
extraction (engine/lexicon.py) — we detect which aspects the traveler
mentioned by keyword match, turn that into a weight vector, and reuse the
exact same weighted-scoring math as the Persona Engine (persona.py).

IMPORTANT HONESTY FIX: aspect-level sentiment (e.g. "Amenities: 100%") is an
AVERAGE across many different things (gym, pool, wifi, parking, pets, AC,
etc.). A hotel can score perfectly on "Amenities" from great wifi/pool
reviews while having ZERO reviews that ever mention pets. That produced a
real bug: searching "pet friendly" returned "100% match" hotels with no pet
mentions at all. So beyond the aspect-level score, we now verify whether the
LITERAL keywords the traveler typed actually appear in that hotel's real
review text, and rank/label hotels with confirmed direct evidence above
those that only match at the broader category level.
"""

import re
from .lexicon import ASPECT_KEYWORDS
from .aggregate import ASPECT_LIST

WORD_RE = re.compile(r"[a-zA-Z']+")


def extract_priorities_from_text(text: str) -> dict:
    """
    Detects which aspects a traveler cares about from free text, weighted by
    how many times each aspect's keywords appear. Returns a weight dict like
    {"noise": 0.4, "location": 0.35, "food": 0.25}. If nothing recognizable
    is detected, returns an empty dict (caller should fall back to a
    neutral/overall ranking rather than guessing).
    """
    lower = text.lower()
    hits = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        count = 0
        for kw in keywords:
            pattern = re.compile(r'(?<![a-zA-Z])' + re.escape(kw) + r'(?![a-zA-Z])')
            count += len(pattern.findall(lower))
        if count > 0:
            hits[aspect] = count

    total = sum(hits.values())
    if total == 0:
        return {}
    return {aspect: count / total for aspect, count in hits.items()}


def _extract_literal_keywords(text: str) -> list:
    """Returns the actual keyword strings detected in the query (not just
    which aspect they belong to) — used to verify real evidence exists,
    rather than trusting the aspect-level average alone."""
    lower = text.lower()
    found = []
    for keywords in ASPECT_KEYWORDS.values():
        for kw in keywords:
            pattern = re.compile(r'(?<![a-zA-Z])' + re.escape(kw) + r'(?![a-zA-Z])')
            if pattern.search(lower):
                found.append(kw)
    return found


def _weighted_match_score(profile: dict, weights: dict):
    """Same math as persona.compute_persona_match, generalized to take an
    arbitrary weight dict instead of a named persona."""
    aspect_scores = profile["aspect_scores"]
    weighted_sum = 0.0
    weight_total = 0.0
    matched_aspects = []

    for aspect, weight in weights.items():
        stats = aspect_scores.get(aspect)
        if not stats or stats["review_count"] == 0:
            continue
        weighted_sum += stats["avg_sentiment"] * weight
        weight_total += weight
        if stats["avg_sentiment"] >= 0.2:
            matched_aspects.append(aspect.replace("_", " ").title())

    if weight_total == 0:
        return None, []

    avg_weighted_sentiment = weighted_sum / weight_total
    match_pct = round((avg_weighted_sentiment + 1) * 50, 1)  # -1..1 -> 0..100
    return match_pct, matched_aspects


def rank_hotels_by_priorities(profiles: dict, priority_text: str, processed_df=None, top_n: int = 5):
    """
    Main entry point. `profiles` is {hotel_id: profile_dict}. `processed_df`
    (optional) is the full reviews DataFrame — when provided, results are
    checked for literal keyword evidence in real review text, not just the
    broader aspect-level average. Returns:
    {
        "detected_priorities": {...} or {},
        "is_fallback": bool,
        "results": [{..., matched_on, has_direct_evidence}]
    }
    """
    weights = extract_priorities_from_text(priority_text)
    literal_keywords = _extract_literal_keywords(priority_text)

    if not weights:
        results = []
        for profile in profiles.values():
            results.append({
                "hotel_id": profile["hotel_id"],
                "hotel_name": profile["hotel_name"],
                "area": profile["area"],
                "match_pct": round(profile["trust"]["trust_score"], 1),
                "matched_on": [],
                "has_direct_evidence": None,
            })
        results.sort(key=lambda r: r["match_pct"], reverse=True)
        return {"detected_priorities": {}, "is_fallback": True, "results": results[:top_n]}

    # Precompute, per hotel_id, the lowercase concatenation of its review
    # text — only if we have literal keywords worth checking and the raw
    # reviews were provided. This lets us confirm real evidence exists
    # rather than trusting the aspect-level average alone.
    hotel_text_cache = {}
    keyword_patterns = []
    if processed_df is not None and literal_keywords:
        grouped = processed_df.groupby("hotel_id")["review_text"].apply(
            lambda texts: " ".join(t.lower() for t in texts)
        )
        hotel_text_cache = grouped.to_dict()
        # Word-boundary patterns — a naive substring check would match "pet"
        # inside "competent", "carpet", "repeat", etc. Same fix pattern used
        # throughout the NLP engine. Optional trailing 's' so singular/plural
        # forms both count (a query for "pet" should still find a review
        # that says "pets").
        keyword_patterns = [
            re.compile(r'(?<![a-zA-Z])' + re.escape(kw) + r's?(?![a-zA-Z])')
            for kw in literal_keywords
        ]

    results = []
    for profile in profiles.values():
        match_pct, matched_aspects = _weighted_match_score(profile, weights)
        if match_pct is None:
            continue

        has_direct_evidence = None
        if hotel_text_cache:
            hotel_text = hotel_text_cache.get(profile["hotel_id"], "")
            has_direct_evidence = any(pat.search(hotel_text) for pat in keyword_patterns)

        results.append({
            "hotel_id": profile["hotel_id"],
            "hotel_name": profile["hotel_name"],
            "area": profile["area"],
            "match_pct": match_pct,
            "matched_on": matched_aspects,
            "has_direct_evidence": has_direct_evidence,
        })

    if hotel_text_cache:
        # confirmed-evidence hotels first, then by match score within each group
        results.sort(key=lambda r: (r["has_direct_evidence"] is not True, -r["match_pct"]))
    else:
        results.sort(key=lambda r: r["match_pct"], reverse=True)

    return {"detected_priorities": weights, "is_fallback": False, "results": results[:top_n]}