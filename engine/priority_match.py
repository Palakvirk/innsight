"""
InnSight — Priority Matcher (freeform "what do you want" search)
---------------------------------------------------------------------
Lets a traveler type what they care about in plain language (e.g. "quiet
hotel near the metro with good breakfast") and ranks hotels accordingly.

Design note: this reuses the SAME aspect keyword lexicon that powers review
extraction (engine/lexicon.py) — we detect which aspects the traveler
mentioned by keyword match, turn that into a weight vector, and reuse the
exact same weighted-scoring math as the Persona Engine (persona.py). This
is deliberate: one lexicon, one scoring formula, three different features
(review analysis, persona matching, free-text matching) — not three
separate systems to maintain.
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


def rank_hotels_by_priorities(profiles: dict, priority_text: str, top_n: int = 5):
    """
    Main entry point. `profiles` is {hotel_id: profile_dict} (as cached in
    the API's in-memory store). Returns:
    {
        "detected_priorities": {...} or {},
        "results": [{hotel_id, hotel_name, area, match_pct, matched_on: [...]}]
    }
    If no recognizable priorities were detected in the text, falls back to
    ranking by overall trust_score + avg_rating instead of guessing intent.
    """
    weights = extract_priorities_from_text(priority_text)

    results = []
    if not weights:
        for profile in profiles.values():
            results.append({
                "hotel_id": profile["hotel_id"],
                "hotel_name": profile["hotel_name"],
                "area": profile["area"],
                "match_pct": round(profile["trust"]["trust_score"], 1),
                "matched_on": [],
            })
        results.sort(key=lambda r: r["match_pct"], reverse=True)
        return {"detected_priorities": {}, "results": results[:top_n]}

    for profile in profiles.values():
        match_pct, matched_aspects = _weighted_match_score(profile, weights)
        if match_pct is None:
            continue
        results.append({
            "hotel_id": profile["hotel_id"],
            "hotel_name": profile["hotel_name"],
            "area": profile["area"],
            "match_pct": match_pct,
            "matched_on": matched_aspects,
        })

    results.sort(key=lambda r: r["match_pct"], reverse=True)
    return {"detected_priorities": weights, "results": results[:top_n]}