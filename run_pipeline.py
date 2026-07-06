"""
InnSight — Full Pipeline Runner
----------------------------------
Ingestion -> NLP extraction (once, cached) -> per-hotel aggregation
(with area-cohort comparative scoring) -> Hotel Intelligence Profiles.

Run: python3 run_pipeline.py
"""

import json
import time
import pandas as pd
from engine.ingest import load_raw_reviews
from engine.aggregate import process_all_reviews, build_hotel_profile, ASPECT_LIST


def build_area_cohorts(processed_df: pd.DataFrame):
    """
    Precompute per-area aspect sentiment distributions so each hotel can be
    percentile-ranked against other hotels in the same locality (our stand-in
    for 'price band', since this dataset has no price field).
    """
    from engine.aggregate import _aspect_scores_for_hotel

    hotel_ids = processed_df["hotel_id"].unique()
    area_of_hotel = processed_df.groupby("hotel_id")["area"].first()

    area_aspect_values = {}
    hotel_aspect_cache = {}

    for hid in hotel_ids:
        hotel_reviews = processed_df[processed_df["hotel_id"] == hid]
        scores = _aspect_scores_for_hotel(hotel_reviews)
        hotel_aspect_cache[hid] = scores
        area = area_of_hotel[hid]
        area_aspect_values.setdefault(area, {a: [] for a in ASPECT_LIST})
        for a in ASPECT_LIST:
            if scores[a]["review_count"] > 0:
                area_aspect_values[area][a].append(scores[a]["avg_sentiment"])

    area_cohorts = {}
    for area, aspects in area_aspect_values.items():
        area_cohorts[area] = {a: pd.Series(v) for a, v in aspects.items()}

    return area_cohorts, hotel_aspect_cache


def main():
    t0 = time.time()
    print("Loading raw reviews...")
    df = load_raw_reviews("data/hotel_reviews.csv")
    print(f"  {len(df)} reviews, {df['hotel_id'].nunique()} hotels ({time.time()-t0:.1f}s)")

    print("Running NLP aspect extraction (one pass over all reviews)...")
    processed_df = process_all_reviews(df)
    print(f"  done ({time.time()-t0:.1f}s elapsed)")

    print("Building area cohorts for comparative scoring...")
    area_cohorts, _ = build_area_cohorts(processed_df)
    print(f"  {len(area_cohorts)} areas ({time.time()-t0:.1f}s elapsed)")

    print("Building hotel profiles for a sample of hotels...")
    review_counts = processed_df.groupby("hotel_id").size().sort_values(ascending=False)
    sample_hotel_ids = review_counts.head(5).index.tolist()

    profiles = []
    for hid in sample_hotel_ids:
        area = processed_df[processed_df["hotel_id"] == hid]["area"].iloc[0]
        cohort_df = area_cohorts.get(area, {})
        profile = build_hotel_profile(hid, processed_df, city_cohort_df=cohort_df)
        profiles.append(profile)

    print(f"  done ({time.time()-t0:.1f}s elapsed)\n")

    with open("output/sample_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    for p in profiles:
        print("=" * 70)
        print(f"{p['hotel_name']} ({p['area']})")
        print(f"Reviews: {p['review_count']} | Avg Rating: {p['avg_rating']}/10")
        print(f"Trust Score: {p['trust']['trust_score']}/100 ({p['trust']['reliability']})")
        print(f"Booking Risk: {p['booking_risk']['level']} — {p['booking_risk']['reasons'][0]}")
        print(f"Should I Book?: {p['should_i_book']['decision']}")
        print("Deal Breakers:", [d["aspect"] for d in p["deal_breakers"]])

    print(f"\nFull profiles written to output/sample_profiles.json")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()