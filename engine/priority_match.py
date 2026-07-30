"""
InnSight — Priority Matcher (freeform "what do you want" search)
---------------------------------------------------------------------
Lets a traveler type what they care about in plain language (e.g. "quiet
hotel near the metro with good breakfast") and ranks hotels accordingly.

Design note: this reuses the SAME aspect keyword lexicon that powers review
extraction (engine/lexicon.py) — we detect which aspects the traveler
mentioned by keyword match, turn that into a weight vector, and reuse the
exact same weighted-scoring math as the Persona Engine (persona.py).

IMPORTANT HONESTY FIXES (two layers deep):
1. Aspect-level sentiment (e.g. "Amenities: 100%") is an AVERAGE across many
   different things (gym, pool, wifi, parking, pets, AC, etc.). A hotel can
   score perfectly on "Amenities" from great wifi/pool reviews while having
   ZERO reviews that ever mention pets. Fixed by verifying literal keyword
   evidence in real review text before calling something a match.
2. Even among hotels with confirmed evidence, a rule-based lexicon can't
   tell "hotel confirms guests can bring pets" apart from "hotel happens to
   have its own pet on-site" — that's a semantic judgment, not a keyword-
   matching problem. Rule-based sentence sentiment scoring actively got
   this backwards (an idiom like "life savers for people with pets" scored
   neutral, while an unrelated "good" elsewhere in another hotel's sentence
   outscored it). So for the small shortlist of keyword-confirmed
   candidates, we ask an LLM (Gemini) whether the evidence genuinely
   supports the traveler's request — same grounded approach as the RAG
   chat, just applied to ranking a handful of candidates instead of
   answering one question. This only costs one extra API call, since it
   only runs on the few hotels that already passed the keyword filter.
"""

import os
import re
import json
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


def _get_evidence_sentences(hotel_reviews_text: str, patterns: list, max_sentences: int = 3) -> str:
    """Extract just the sentence(s) containing a keyword match, so the LLM
    verification prompt stays short and focused rather than sending whole
    review corpora."""
    from .nlp_engine import split_sentences
    sentences = split_sentences(hotel_reviews_text)
    matched = []
    for s in sentences:
        if any(pat.search(s.lower()) for pat in patterns):
            matched.append(s.strip())
        if len(matched) >= max_sentences:
            break
    return " ".join(matched)


def _llm_verify_candidates(candidates: list, priority_text: str) -> dict:
    """
    For a SHORT shortlist of keyword-confirmed candidates, asks an LLM
    whether the extracted evidence sentences genuinely support what the
    traveler asked for — not just whether a keyword happened to appear.
    Returns {hotel_id: {"confirmed": bool, "strength": 0-100}}.

    Fails safe: if the API key is missing, the call errors, or the response
    can't be parsed, returns an empty dict — callers should fall back to
    trust-score ordering rather than blocking on this.
    """
    if not candidates:
        return {}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {}

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        candidate_block = "\n".join(
            f'{i+1}. hotel_id={c["hotel_id"]} | "{c["hotel_name"]}" | '
            f'evidence: "{c["evidence_text"]}"'
            for i, c in enumerate(candidates)
        )

        prompt = f"""A traveler is searching for hotels matching this request: "{priority_text}"

Below are hotels whose reviews happened to contain a related keyword, along
with the exact evidence sentence(s) from real guest reviews. For EACH
hotel, judge whether the evidence genuinely CONFIRMS the traveler's request
(not just an incidental/unrelated mention of a similar word).

{candidate_block}

Respond with ONLY a JSON array, no other text, in this exact format:
[{{"hotel_id": <int>, "confirmed": true/false, "strength": <0-100>}}, ...]

"confirmed" = does the evidence genuinely support the traveler's request.
"strength" = how strongly/clearly it confirms it (0 = barely relevant, 100 = explicit and clear)."""

        model = genai.GenerativeModel("gemini-2.5-flash")

        # The google-generativeai SDK's own timeout handling is unreliable
        # (well-documented open issues where it's simply not respected), so
        # we enforce a hard cutoff ourselves via a thread — if Gemini takes
        # too long, we give up and fall back to trust-score ordering rather
        # than let the request hang.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.generate_content, prompt)
            response = future.result(timeout=12)

        text = response.text.strip()
        # strip markdown code fences if the model added them despite instructions
        text = re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(text)

        return {
            item["hotel_id"]: {"confirmed": item["confirmed"], "strength": item["strength"]}
            for item in parsed
        }
    except Exception:
        return {}


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
            "trust_score": profile["trust"]["trust_score"],
        })

    if hotel_text_cache:
        # Ask an LLM to verify only a BOUNDED shortlist of the strongest
        # keyword-confirmed candidates — see module docstring for why
        # rule-based sentiment can't reliably do this verification itself.
        # IMPORTANT: for common keywords (e.g. "metro", "clean"), dozens or
        # even hundreds of hotels can match — sending all of them into one
        # LLM prompt would be slow and unbounded. We cap it to a small,
        # fixed number (ranked by trust score first) so latency stays
        # predictable no matter how common the query terms are. Anything
        # outside this cap still gets the keyword-match label, just without
        # LLM verification — it falls back to trust-score ordering below.
        MAX_LLM_CANDIDATES = 12

        evidence_confirmed = [r for r in results if r["has_direct_evidence"]]
        evidence_confirmed.sort(key=lambda r: -r["trust_score"])
        shortlist = evidence_confirmed[:MAX_LLM_CANDIDATES]

        llm_verdicts = {}
        if shortlist:
            candidates_for_llm = []
            for r in shortlist:
                hotel_text = hotel_text_cache.get(r["hotel_id"], "")
                evidence_text = _get_evidence_sentences(hotel_text, keyword_patterns)
                candidates_for_llm.append({
                    "hotel_id": r["hotel_id"],
                    "hotel_name": r["hotel_name"],
                    "evidence_text": evidence_text or "(no exact sentence extracted)",
                })
            llm_verdicts = _llm_verify_candidates(candidates_for_llm, priority_text)

        for r in results:
            verdict = llm_verdicts.get(r["hotel_id"])
            if verdict is not None:
                r["has_direct_evidence"] = verdict["confirmed"]
                r["llm_strength"] = verdict["strength"]
            else:
                r["llm_strength"] = None

        def sort_key(r):
            if r["llm_strength"] is not None:
                # LLM-verified: sort confirmed-true first, by strength
                return (0 if r["has_direct_evidence"] else 1, -r["llm_strength"])
            # No LLM verdict available (outside the cap, or call failed):
            # confirmed-by-keyword-only candidates next, then trust score
            return (2 if not r["has_direct_evidence"] else 1, -r["trust_score"])

        results.sort(key=sort_key)
    else:
        results.sort(key=lambda r: r["match_pct"], reverse=True)

    return {"detected_priorities": weights, "is_fallback": False, "results": results[:top_n]}