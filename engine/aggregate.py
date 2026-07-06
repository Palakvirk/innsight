"""
InnSight — Aggregation Layer
------------------------------
Combines per-review NLP output + Trust Engine output into ONE object per
hotel: the Hotel Intelligence Profile. Every downstream feature (Trust Score
card, Radar Chart, Timeline, Booking Risk, Deal Breakers, Comparative Score,
"Should I Book?") reads ONLY from this object — nothing re-touches raw
reviews. That single-source-of-truth design is the core architecture claim.
"""

import pandas as pd
import numpy as np
from .nlp_engine import extract_aspects
from .trust_engine import compute_trust_score

ASPECT_LIST = [
    "cleanliness", "staff", "food", "rooms",
    "amenities", "noise", "location", "value", "check_in",
]


def process_all_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the NLP engine over every review once, storing results as a column
    of aspect-dicts. This is the expensive step — do it once, reuse everywhere.
    """
    df = df.copy()
    df["aspects"] = df["review_text"].apply(extract_aspects)
    return df


def _aspect_scores_for_hotel(hotel_reviews: pd.DataFrame):
    """
    Builds per-aspect aggregate stats: avg sentiment, review count, and
    trend deltas (approximated at month granularity since the dataset only
    has month-level dates).
    """
    rows = []
    for _, r in hotel_reviews.iterrows():
        for a in r["aspects"]:
            rows.append({
                "review_date": r["review_date"],
                "aspect": a["aspect"],
                "sentiment": a["sentiment"],
            })
    if not rows:
        return {a: {"avg_sentiment": 0.0, "review_count": 0, "trend": "flat", "trend_pct": 0.0}
                for a in ASPECT_LIST}

    adf = pd.DataFrame(rows)
    latest_date = hotel_reviews["review_date"].max()

    result = {}
    for aspect in ASPECT_LIST:
        sub = adf[adf["aspect"] == aspect]
        if sub.empty:
            result[aspect] = {"avg_sentiment": 0.0, "review_count": 0, "trend": "flat", "trend_pct": 0.0}
            continue

        avg_sentiment = sub["sentiment"].mean()

        recent = sub[sub["review_date"] > latest_date - pd.Timedelta(days=90)]
        prior = sub[
            (sub["review_date"] <= latest_date - pd.Timedelta(days=90)) &
            (sub["review_date"] > latest_date - pd.Timedelta(days=180))
        ]

        trend_pct = 0.0
        trend = "flat"
        if len(recent) >= 2 and len(prior) >= 2:
            recent_avg = recent["sentiment"].mean()
            prior_avg = prior["sentiment"].mean()
            recent_score = (recent_avg + 1) * 50
            prior_score = (prior_avg + 1) * 50
            if prior_score != 0:
                trend_pct = ((recent_score - prior_score) / abs(prior_score)) * 100
            trend = "up" if trend_pct > 5 else "down" if trend_pct < -5 else "flat"

        result[aspect] = {
            "avg_sentiment": round(float(avg_sentiment), 3),
            "review_count": int(len(sub)),
            "trend": trend,
            "trend_pct": round(float(trend_pct), 1),
        }
    return result


def _deal_breakers(aspect_scores: dict, top_n: int = 4):
    """Rank aspects by (negative sentiment * mention volume) to surface the
    most-cited real complaints, not just the lowest score on thin data."""
    scored = []
    for aspect, stats in aspect_scores.items():
        if stats["avg_sentiment"] < -0.1 and stats["review_count"] >= 2:
            severity = abs(stats["avg_sentiment"]) * np.log1p(stats["review_count"])
            scored.append((aspect, stats["review_count"], severity))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [{"aspect": a, "mention_count": c} for a, c, _ in scored[:top_n]]


def _booking_risk(aspect_scores: dict, trust: dict, avg_rating: float, deal_breakers: list):
    """
    Rules-based risk level from FOUR independent signal types:
      1. Trend risk    — an aspect actively getting worse recently
      2. Absolute risk — an aspect is bad right now, regardless of trend
      3. Trust risk    — low trust score / suspicious posting patterns
      4. Deal-breaker risk — an aspect already surfaced as a Deal Breaker
         (kept in sync with _deal_breakers() so the two features never
         contradict each other, e.g. "no risk" while listing a deal breaker)
    """
    reasons = []

    down_trends = [a for a, s in aspect_scores.items() if s["trend"] == "down" and s["review_count"] >= 2]
    for a in down_trends:
        reasons.append(f"{a.replace('_', ' ').title()} sentiment declining")

    poor_aspects = [
        a for a, s in aspect_scores.items()
        if s["avg_sentiment"] <= -0.3 and s["review_count"] >= 2
    ]
    for a in poor_aspects:
        if a not in down_trends:
            reasons.append(f"{a.replace('_', ' ').title()} rated consistently poor")

    deal_breaker_aspects = [d["aspect"] for d in deal_breakers]
    for a in deal_breaker_aspects:
        if a not in down_trends and a not in poor_aspects:
            reasons.append(f"{a.replace('_', ' ').title()} frequently mentioned as a concern")

    if avg_rating <= 4.0:
        reasons.append(f"Overall guest rating is low ({avg_rating}/10)")
    if trust["trust_score"] < 55:
        reasons.append("Low review trust score")
    if trust["burst_events"]:
        reasons.append("Unusual review posting pattern detected")

    severity_signals = (
        len(down_trends)
        + len(poor_aspects)
        + (1 if deal_breaker_aspects else 0)
        + (1 if avg_rating <= 4.0 else 0)
        + (1 if trust["trust_score"] < 45 else 0)
        + (1 if trust["burst_events"] else 0)
        + (1 if trust.get("suspicious_review_rate", 0) >= 30 else 0)
    )

    if severity_signals >= 2 or avg_rating <= 3.0:
        level = "High"
    elif severity_signals == 1 or trust["trust_score"] < 65:
        level = "Medium"
    else:
        level = "Low"

    if not reasons:
        reasons.append("No significant negative patterns detected")

    return {"level": level, "reasons": reasons}

def _should_i_book(aspect_scores: dict, trust: dict, risk: dict, comparative_pct: dict,
                    avg_rating: float, deal_breakers: list):
    """The synthesis layer: one final recommendation from everything above."""
    avg_percentile = np.mean(list(comparative_pct.values())) if comparative_pct else 50

    if risk["level"] == "High" or trust["trust_score"] < 45 or avg_rating <= 4.0:
        return {"decision": "Wait", "reasons": risk["reasons"][:3] + ["Consider nearby alternatives"]}

    if risk["level"] == "Low" and trust["trust_score"] >= 75 and avg_rating >= 7.0 and not deal_breakers:
        decision = "Book"
        reasons = ["Strong trust score", "No significant risk factors"]
        if avg_percentile >= 70:
            reasons.append(f"Ranks in top {100 - int(avg_percentile)}% among nearby hotels")
        return {"decision": decision, "reasons": reasons}

    return {"decision": "Book with Caution", "reasons": risk["reasons"][:3]}


def build_hotel_profile(hotel_id, all_reviews_df: pd.DataFrame, city_cohort_df: pd.DataFrame = None):
    """
    Builds the full Hotel Intelligence Profile for one hotel.
    `city_cohort_df` (optional): pre-aggregated aspect scores for hotels in
    the same area, used for comparative percentile ranking.
    """
    hotel_reviews = all_reviews_df[all_reviews_df["hotel_id"] == hotel_id]
    if hotel_reviews.empty:
        return None

    aspect_scores = _aspect_scores_for_hotel(hotel_reviews)
    overall_sentiment = np.mean([
        s["avg_sentiment"] for s in aspect_scores.values() if s["review_count"] > 0
    ]) if any(s["review_count"] > 0 for s in aspect_scores.values()) else 0.0

    avg_rating = round(float(hotel_reviews["rating"].mean()), 2)

    trust = compute_trust_score(hotel_reviews, overall_sentiment)
    deal_breakers = _deal_breakers(aspect_scores)
    risk = _booking_risk(aspect_scores, trust, avg_rating, deal_breakers)

    comparative_pct = {}
    if city_cohort_df is not None:
        for aspect in ASPECT_LIST:
            hotel_val = aspect_scores[aspect]["avg_sentiment"]
            cohort_vals = city_cohort_df.get(aspect)
            if cohort_vals is not None and len(cohort_vals) > 1:
                percentile = float((cohort_vals < hotel_val).mean() * 100)
                comparative_pct[aspect] = round(percentile, 1)

    should_book = _should_i_book(aspect_scores, trust, risk, comparative_pct, avg_rating, deal_breakers)

    return {
        "hotel_id": int(hotel_id),
        "hotel_name": hotel_reviews["hotel_name"].iloc[0],
        "area": hotel_reviews["area"].iloc[0],
        "review_count": int(len(hotel_reviews)),
        "avg_rating": avg_rating,
        "aspect_scores": aspect_scores,
        "trust": trust,
        "deal_breakers": deal_breakers,
        "booking_risk": risk,
        "comparative_percentile": comparative_pct,
        "should_i_book": should_book,
        "last_updated": str(hotel_reviews["review_date"].max().date()),
    }