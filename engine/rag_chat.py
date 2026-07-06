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
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from .retrieval import retrieve_relevant_reviews

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"


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


def ask_hotel_question(question: str, hotel_reviews, hotel_name: str, top_k: int = 8):
    """
    Main entry point. Returns:
    {
        "answer": str,
        "confidence": float (0-1),
        "supporting_review_ids": [...],
        "based_on_count": int,
    }
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

    prompt = _build_prompt(question, retrieved, hotel_name)
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)

    return {
        "answer": response.text.strip(),
        "confidence": confidence,
        "supporting_review_ids": [r["review_id"] for r in retrieved],
        "based_on_count": len(retrieved),
    }


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