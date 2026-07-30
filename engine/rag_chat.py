"""
InnSight — RAG Chat (Layer 3: Generation)
---------------------------------------------
Combines the retrieval layer (retrieval.py) with an LLM call (Google Gemini,
free tier) to answer natural-language questions about a hotel, grounded in
its actual reviews. Every answer comes with a Confidence Score derived from
retrieval quality, not just handed back from the model — this is important:
we don't ask the LLM "how confident are you", we compute confidence
ourselves from how much real evidence backed the retrieval.

Confidence Score formula:
    - 0 retrieved reviews          -> confidence = 0  (honestly "don't know")
    - few reviews, low similarity  -> low confidence
    - many reviews, high similarity, consistent ratings -> high confidence
"""

import os
import time
import random
import hashlib
import logging
import concurrent.futures
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from .retrieval import retrieve_relevant_reviews

load_dotenv()

logger = logging.getLogger("innsight.rag_chat")

# IMPORTANT: don't crash the whole API process just because the key is
# missing/misconfigured. Previously `genai.configure` ran at import time
# with a bare `os.environ[...]` lookup, so a missing/blank GEMINI_API_KEY
# would raise a KeyError the moment api.py imported this module — taking
# down every endpoint, not just chat. Configure defensively instead, and
# let ask_hotel_question() fail soft per-request.
_API_KEY = os.environ.get("GEMINI_API_KEY")
if _API_KEY:
    genai.configure(api_key=_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set — chat will return a fallback answer instead of calling Gemini.")

MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_TIMEOUT_SECONDS = 20

FALLBACK_ANSWER = (
    "I'm having trouble reaching the AI service right now, so I can't generate "
    "an answer for this question at the moment. Please try again in a moment."
)

# ---- Simple in-memory response cache ----
# The free Gemini tier has a tight requests-per-minute/day quota, and demo
# sessions tend to repeat similar questions across hotels/testers. Caching
# identical (hotel, question) pairs cuts real API calls dramatically without
# touching any of the actual retrieval/generation logic. In-memory is fine
# for a single-process hackathon deploy; it just resets on restart.
_response_cache = {}
CACHE_TTL_SECONDS = 60 * 60  # 1 hour


def _cache_key(hotel_name: str, question: str) -> str:
    normalized = f"{hotel_name.strip().lower()}::{question.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cache_get(key: str):
    entry = _response_cache.get(key)
    if entry is None:
        return None
    result, saved_at = entry
    if time.time() - saved_at > CACHE_TTL_SECONDS:
        _response_cache.pop(key, None)
        return None
    return result


def _cache_set(key: str, result: dict):
    _response_cache[key] = (result, time.time())


def _compute_confidence(retrieved: list) -> float:
    """
    Confidence is NOT the LLM's self-reported certainty — it's computed from
    retrieval evidence, which is a more honest signal:
      - more supporting reviews  -> higher confidence
      - higher average similarity -> higher confidence
      - consistent ratings among supporting reviews -> higher confidence
        (if retrieved reviews wildly disagree, the answer is less reliable)
    """
    if not retrieved:
        return 0.0

    n = len(retrieved)
    avg_similarity = np.mean([r["similarity"] for r in retrieved])
    ratings = [r["rating"] for r in retrieved]
    rating_std = np.std(ratings) if len(ratings) > 1 else 0.0
    consistency = max(0.0, 1.0 - rating_std / 4.5)

    volume_factor = min(1.0, n / 5.0)

    confidence = (0.4 * avg_similarity + 0.3 * volume_factor + 0.3 * consistency)
    return round(float(min(0.97, max(0.05, confidence))), 3)


def _build_prompt(question: str, retrieved: list, hotel_name: str) -> str:
    review_block = "\n".join(
        f"- (rating {r['rating']}/10): {r['review_text']}" for r in retrieved
    )
    return f"""You are answering a traveler's question about "{hotel_name}" using ONLY
the guest reviews provided below. Do not use outside knowledge about this
hotel or hotels in general.

Guest reviews:
{review_block}

Question: {question}

Instructions:
- Answer in 2-3 sentences, directly and honestly based on the reviews above.
- If the reviews don't clearly address the question, say so explicitly.
- Do not make up details not present in the reviews.
"""


def _extract_answer_text(response) -> str:
    """
    Safely pulls text out of a Gemini response. `response.text` raises
    ValueError if Gemini's safety filters blocked every candidate (no parts
    to read) — this happens silently and looks identical to a normal call
    from the outside, so it's a common source of unhandled crashes. We check
    explicitly instead of letting that attribute access throw.
    """
    if not getattr(response, "candidates", None):
        block_reason = None
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None:
            block_reason = getattr(feedback, "block_reason", None)
        raise ValueError(f"Gemini returned no candidates (block_reason={block_reason})")

    candidate = response.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    # finish_reason 3 == SAFETY in the genai SDK's enum
    if finish_reason == 3:
        raise ValueError("Gemini blocked the response for safety reasons")

    return response.text.strip()


def _call_gemini(prompt: str, max_retries: int = 2) -> str:
    """
    Calls Gemini with a hard timeout (the SDK's own timeout handling is
    unreliable — same issue noted in priority_match.py). Retries on 429
    ResourceExhausted specifically, with a short backoff plus jitter — a
    single retry often succeeds since free-tier RPM limits are a rolling
    window, not a hard daily block. Any other exception (or repeated 429s)
    is raised so the caller can fail soft.
    """
    from google.api_core.exceptions import ResourceExhausted

    model = genai.GenerativeModel(MODEL_NAME)
    attempt = 0
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.generate_content, prompt)
                response = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
            return _extract_answer_text(response)
        except ResourceExhausted:
            attempt += 1
            if attempt > max_retries:
                raise
            backoff = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("Gemini rate-limited (429) — retry %s/%s in %.1fs",
                            attempt, max_retries, backoff)
            time.sleep(backoff)


def ask_hotel_question(question: str, hotel_reviews, hotel_name: str, top_k: int = 8):
    """
    Main entry point. Returns:
    {
        "answer": str,
        "confidence": float (0-1),
        "supporting_review_ids": [...],
        "based_on_count": int,
    }
    Fails soft: if retrieval finds nothing, or the Gemini call fails for any
    reason (missing key, rate limit, timeout, safety block, network error),
    this returns a normal 200-shaped dict with an honest fallback answer
    instead of raising — callers should never need a try/except around this.
    """
    retrieved = retrieve_relevant_reviews(question, hotel_reviews, top_k=top_k)
    confidence = _compute_confidence(retrieved)

    if not retrieved:
        return {
            "answer": "I couldn't find any reviews discussing this — not enough information to answer confidently.",
            "confidence": 0.0,
            "supporting_review_ids": [],
            "based_on_count": 0,
        }

    if not _API_KEY:
        return {
            "answer": FALLBACK_ANSWER,
            "confidence": 0.0,
            "supporting_review_ids": [r["review_id"] for r in retrieved],
            "based_on_count": len(retrieved),
        }

    cache_key = _cache_key(hotel_name, question)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    prompt = _build_prompt(question, retrieved, hotel_name)

    try:
        answer_text = _call_gemini(prompt)
    except concurrent.futures.TimeoutError:
        logger.warning("Gemini call timed out after %ss for question=%r hotel=%r",
                        GEMINI_TIMEOUT_SECONDS, question, hotel_name)
        answer_text = FALLBACK_ANSWER
        confidence = 0.0
    except Exception as e:
        # Covers rate limits (after retries exhausted), invalid/expired API
        # key, network errors, safety blocks, and anything else the SDK can
        # throw. Logged with the real exception so it's diagnosable
        # server-side, but the person using the app just sees an honest,
        # non-crashing answer.
        logger.exception("Gemini call failed for question=%r hotel=%r: %s",
                          question, hotel_name, e)
        answer_text = FALLBACK_ANSWER
        confidence = 0.0

    result = {
        "answer": answer_text,
        "confidence": confidence,
        "supporting_review_ids": [r["review_id"] for r in retrieved],
        "based_on_count": len(retrieved),
    }

    # Only cache genuine successful answers — never cache the fallback, or
    # a transient rate-limit/outage would get "stuck" as the cached answer
    # for the rest of the TTL window.
    if answer_text != FALLBACK_ANSWER:
        _cache_set(cache_key, result)

    return result


if __name__ == "__main__":
    from .ingest import load_raw_reviews

    df = load_raw_reviews("data/hotel_reviews.csv")
    top_hotel_id = df.groupby("hotel_id").size().idxmax()
    hotel_reviews = df[df["hotel_id"] == top_hotel_id]
    hotel_name = hotel_reviews["hotel_name"].iloc[0]
    print(f"Chatting about: {hotel_name} ({len(hotel_reviews)} reviews)\n")

    test_questions = [
        "Is the WiFi good?",
        "Is parking available?",
        "Is breakfast included and is it good?",
    ]
    for q in test_questions:
        print("=" * 70)
        print(f"Q: {q}")
        result = ask_hotel_question(q, hotel_reviews, hotel_name)
        print(f"A: {result['answer']}")
        print(f"   Confidence: {result['confidence']*100:.0f}% "
              f"(based on {result['based_on_count']} reviews)")