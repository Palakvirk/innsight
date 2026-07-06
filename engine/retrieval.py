"""
InnSight — Retrieval Layer (RAG: the "R")
--------------------------------------------
Given a hotel and a natural-language question, finds the most relevant
reviews to ground an LLM's answer in. Uses TF-IDF + cosine similarity —
the same technique as the Trust Engine's template-similarity detector, so
there's no new dependency and the mechanism is easy to explain to judges:
"we measure word-overlap similarity between the question and each review,
and hand the LLM only the reviews that are actually relevant."

This module is intentionally LLM-agnostic: it just returns ranked reviews.
The generation step (calling Gemini) lives in rag_chat.py.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def retrieve_relevant_reviews(question: str, hotel_reviews: pd.DataFrame, top_k: int = 8):
    """
    Ranks this hotel's reviews by relevance to `question` using TF-IDF
    cosine similarity. Returns a list of dicts:
    [{review_id, review_text, similarity, rating}, ...]
    sorted by similarity descending, limited to top_k.

    Reviews with near-zero similarity (score <= 0.05) are excluded — if
    nothing in the corpus actually discusses the question, we want the
    confidence score (built in rag_chat.py) to reflect that honestly,
    rather than force-feeding the LLM unrelated reviews.
    """
    texts = hotel_reviews["review_text"].tolist()
    if len(texts) == 0:
        return []

    # Fit TF-IDF over [question] + all reviews for this hotel, so the
    # question and reviews share the same vocabulary/vector space.
    corpus = [question] + texts
    vec = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        tfidf = vec.fit_transform(corpus)
    except ValueError:
        return []  # e.g. question is only stopwords

    question_vec = tfidf[0:1]
    review_vecs = tfidf[1:]
    similarities = cosine_similarity(question_vec, review_vecs)[0]

    ranked = sorted(
        zip(hotel_reviews["review_id"], texts, similarities, hotel_reviews["rating"]),
        key=lambda x: x[2],
        reverse=True,
    )

    results = []
    for review_id, text, sim, rating in ranked[:top_k]:
        if sim <= 0.05:
            continue
        results.append({
            "review_id": int(review_id),
            "review_text": text,
            "similarity": round(float(sim), 3),
            "rating": float(rating),
        })
    return results


if __name__ == "__main__":
    from .ingest import load_raw_reviews

    df = load_raw_reviews("data/hotel_reviews.csv")
    # pick a hotel with a decent number of reviews to test against
    top_hotel_id = df.groupby("hotel_id").size().idxmax()
    hotel_reviews = df[df["hotel_id"] == top_hotel_id]
    print(f"Testing retrieval on: {hotel_reviews['hotel_name'].iloc[0]} "
          f"({len(hotel_reviews)} reviews)\n")

    test_questions = [
        "Is the WiFi good?",
        "Is parking available?",
        "Is it safe for solo female travelers?",
    ]
    for q in test_questions:
        print("=" * 70)
        print(f"Q: {q}")
        results = retrieve_relevant_reviews(q, hotel_reviews, top_k=3)
        if not results:
            print("  No relevant reviews found.")
        for r in results:
            print(f"  [sim={r['similarity']:.2f}, rating={r['rating']}] {r['review_text'][:120]}")