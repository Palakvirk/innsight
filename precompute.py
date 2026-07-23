"""
InnSight — Precompute Script
--------------------------------
Runs the full pipeline ONCE (ingestion -> NLP -> trust -> aggregation) and
saves every hotel's profile to a single JSON file. The API then just loads
this file at startup instead of reprocessing ~4,400 reviews every time it
restarts — which matters a lot on free hosting tiers that spin down after
inactivity and cold-start on the next visit.

Run this manually whenever the underlying data/logic changes:
    python3 precompute.py

Then commit the resulting output/all_profiles.json — the deployed API reads
it directly instead of rebuilding everything from raw reviews.
"""

import json
import time
import pandas as pd
from engine.ingest import load_raw_reviews
from engine.aggregate import process_all_reviews, build_hotel_profile, ASPECT_LIST, _aspect_scores_for_hotel


def main():
    t0 = time.time()
    print("Loading and processing all reviews...")
    df = load_raw_reviews("data/hotel_reviews.csv")
    processed_df = process_all_reviews(df)
    print(f"  {len(processed_df)} reviews processed ({time.time()-t0:.1f}s)")

    hotel_ids = processed_df["hotel_id"].unique()
    area_of_hotel = processed_df.groupby("hotel_id")["area"].first()

    print("Building area cohorts for comparative scoring...")
    area_aspect_values = {}
    for hid in hotel_ids:
        hotel_reviews = processed_df[processed_df["hotel_id"] == hid]
        scores = _aspect_scores_for_hotel(hotel_reviews)
        area = area_of_hotel[hid]
        area_aspect_values.setdefault(area, {a: [] for a in ASPECT_LIST})
        for a in ASPECT_LIST:
            if scores[a]["review_count"] > 0:
                area_aspect_values[area][a].append(scores[a]["avg_sentiment"])
    area_cohorts = {
        area: {a: pd.Series(v) for a, v in aspects.items()}
        for area, aspects in area_aspect_values.items()
    }

    print(f"Building {len(hotel_ids)} hotel profiles...")
    profiles = {}
    for hid in hotel_ids:
        area = area_of_hotel[hid]
        cohort = area_cohorts.get(area, {})
        profile = build_hotel_profile(hid, processed_df, city_cohort_df=cohort)
        profiles[int(hid)] = profile

    with open("output/all_profiles.json", "w") as f:
        json.dump(profiles, f)

    print(f"Done — {len(profiles)} profiles saved to output/all_profiles.json "
          f"({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()