"""
InnSight — Trust Engine (Layer 2)
------------------------------------
Computes per-review suspicion signals and rolls them into a per-hotel
AI Trust Score. This dataset has no reviewer metadata (no reviewer_id,
account age, or verified badge), so every signal here is derived purely
from review TEXT and POSTING PATTERNS. Rule-based and explainable — every
score traces back to a stated reason, no black box.

Trust Score formula (reweighted for available signals):
    Trust Score = Consistency (35%)      -- rating vs text-sentiment agreement
                + Non-Template (25%)     -- low near-duplicate/template rate
                + Non-Burst (20%)        -- no suspicious posting spikes
                + Freshness (20%)        -- recency-weighted review activity
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_rating_text_agreement(hotel_reviews: pd.DataFrame, avg_aspect_sentiment: float) -> float:
    """
    Checks if the hotel's average star rating agrees with the average
    text-derived sentiment. Large disagreement (e.g. 9/10 rating but very
    negative text) is a red flag for manipulated or mismatched reviews.
    Returns a 0-1 score (1 = fully consistent).
    """
    if hotel_reviews.empty:
        return 0.5

    avg_rating_normalized = (hotel_reviews["rating"].mean() - 1) / 9.0 * 2 - 1  # map 1-10 -> -1..1
    disagreement = abs(avg_rating_normalized - avg_aspect_sentiment)
    return max(0.0, 1.0 - disagreement)


def compute_template_similarity_flag(hotel_reviews: pd.DataFrame, threshold: float = 0.85):
    """
    Uses TF-IDF + cosine similarity to detect near-duplicate/templated
    reviews within the same hotel — a classic fake-review signal (copy-paste
    campaigns). Returns (non_template_score 0-1, list of suspicious review_ids).
    """
    texts = hotel_reviews["review_text"].tolist()
    ids = hotel_reviews["review_id"].tolist()

    if len(texts) < 2:
        return 1.0, []

    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
    except ValueError:
        return 1.0, []

    n = len(texts)
    suspicious_ids = set()
    high_sim_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if sim_matrix[i, j] >= threshold:
                high_sim_pairs += 1
                suspicious_ids.add(ids[i])
                suspicious_ids.add(ids[j])

    template_rate = high_sim_pairs / total_pairs if total_pairs > 0 else 0.0
    non_template_score = max(0.0, 1.0 - template_rate * 3)  # amplify penalty
    return non_template_score, list(suspicious_ids)


def compute_burst_score(hotel_reviews: pd.DataFrame, burst_threshold: int = 5):
    """
    Flags months where review volume spikes abnormally vs the hotel's own
    baseline — a signal of coordinated/incentivized review campaigns.
    Returns (non_burst_score 0-1, list of {month, count} burst events).
    """
    if hotel_reviews.empty:
        return 1.0, []

    monthly_counts = hotel_reviews.groupby(
        hotel_reviews["review_date"].dt.to_period("M")
    ).size()

    if len(monthly_counts) < 2:
        return 1.0, []

    mean_c = monthly_counts.mean()
    std_c = monthly_counts.std(ddof=0) or 1.0

    bursts = []
    for month, count in monthly_counts.items():
        z = (count - mean_c) / std_c
        if z > 2.0 and count >= burst_threshold:
            bursts.append({"month": str(month), "count": int(count)})

    non_burst_score = max(0.0, 1.0 - 0.25 * len(bursts))
    return non_burst_score, bursts


def compute_freshness_score(hotel_reviews: pd.DataFrame, reference_date=None):
    """
    Rewards hotels with a healthy stream of RECENT reviews (stale review
    history = harder to trust as a current signal). Returns 0-1.
    """
    if hotel_reviews.empty:
        return 0.0

    reference_date = reference_date or hotel_reviews["review_date"].max()
    days_old = (reference_date - hotel_reviews["review_date"]).dt.days.clip(lower=0)

    weights = np.exp(-days_old / 180.0)  # exponential decay half-life ~180 days
    recency_ratio = weights.sum() / len(hotel_reviews)
    return float(min(1.0, recency_ratio * 2))


def compute_extremity_flags(hotel_reviews: pd.DataFrame):
    """
    Flags reviews that are extreme rating (1/2 or 9/10) + very short text —
    a classic low-effort fake/incentivized review pattern.
    """
    flagged = hotel_reviews[
        (hotel_reviews["rating"].isin([1, 2, 9, 10])) &
        (hotel_reviews["review_text"].str.len() < 25)
    ]
    return flagged["review_id"].tolist()


def compute_trust_score(hotel_reviews: pd.DataFrame, avg_aspect_sentiment: float):
    """Main entry point: computes the full Trust Score breakdown for one hotel."""
    consistency = compute_rating_text_agreement(hotel_reviews, avg_aspect_sentiment)
    non_template, template_suspects = compute_template_similarity_flag(hotel_reviews)
    non_burst, bursts = compute_burst_score(hotel_reviews)
    freshness = compute_freshness_score(hotel_reviews)
    extremity_suspects = compute_extremity_flags(hotel_reviews)

    weights = {
        "consistency": 0.35,
        "non_template": 0.25,
        "non_burst": 0.20,
        "freshness": 0.20,
    }
    components = {
        "consistency": consistency,
        "non_template": non_template,
        "non_burst": non_burst,
        "freshness": freshness,
    }
    trust_score = sum(components[k] * weights[k] for k in weights) * 100

    reliability = (
        "High" if trust_score >= 80 else
        "Medium" if trust_score >= 55 else
        "Low"
    )

    all_suspects = set(template_suspects) | set(extremity_suspects)
    suspicious_rate = len(all_suspects) / len(hotel_reviews) if len(hotel_reviews) else 0.0

    return {
        "trust_score": round(trust_score, 1),
        "reliability": reliability,
        "breakdown": {
            "consistency": round(components["consistency"] * weights["consistency"] * 100, 1),
            "non_template": round(components["non_template"] * weights["non_template"] * 100, 1),
            "non_burst": round(components["non_burst"] * weights["non_burst"] * 100, 1),
            "freshness": round(components["freshness"] * weights["freshness"] * 100, 1),
        },
        "suspicious_review_rate": round(suspicious_rate * 100, 1),
        "burst_events": bursts,
        "flagged_review_ids": list(all_suspects),
    }