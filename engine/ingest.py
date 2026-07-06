"""
InnSight — Ingestion Layer
----------------------------
Loads the raw Kaggle CSV, cleans it, and parses the Month-Year date string
into an actual datetime so trend/timeline features can group by month.

Real dataset columns: Index, Name, Area, Review_Date, Rating_attribute,
Rating(Out of 10), Review_Text

Note: Review_Date is month-granularity only (e.g. "Jul-23"), not a full date.
We parse it to the 15th of that month as a stand-in "mid-month" timestamp —
good enough for monthly trend aggregation, not meant to imply day-level
precision. Any "this week" style feature is reframed as "this month" to
stay honest about what the data actually supports.
"""

import pandas as pd


def load_raw_reviews(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop rows with no review text — nothing to analyze
    df = df.dropna(subset=["Review_Text"]).copy()

    # Parse "Jul-23" -> Timestamp(2023-07-15)
    df["review_date"] = pd.to_datetime(
        df["Review_Date"], format="%b-%y", errors="coerce"
    ) + pd.Timedelta(days=14)

    # Drop rows where date parsing failed
    df = df.dropna(subset=["review_date"]).copy()

    df = df.rename(columns={
        "Name": "hotel_name",
        "Area": "area",
        "Rating_attribute": "review_title",
        "Rating(Out of 10)": "rating",
        "Review_Text": "review_text",
    })

    # Stable hotel_id from hotel name + area (dataset has no explicit ID)
    df["hotel_id"] = (df["hotel_name"] + "|" + df["area"]).astype("category").cat.codes

    #Deduplication logic
    before = len(df)
    df = df.drop_duplicates(subset=["hotel_id", "review_text"], keep="first").copy()
    removed = before - len(df)
    if removed > 0:
        print(f"[ingest] Removed {removed} exact-duplicate reviews "
              f"({removed/before*100:.1f}% of raw rows)")

    # A synthetic-but-stable review_id for downstream referencing
    df["review_id"] = df.index

    return df[[
        "review_id", "hotel_id", "hotel_name", "area",
        "review_date", "rating", "review_title", "review_text",
    ]].reset_index(drop=True)


if __name__ == "__main__":
    df = load_raw_reviews("data/hotel_reviews.csv")
    print(f"Loaded {len(df)} reviews across {df['hotel_id'].nunique()} hotels")
    print(df.head(3).to_string())
    print("\nDate range:", df["review_date"].min(), "to", df["review_date"].max())