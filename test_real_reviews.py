"""
Quick sanity check: run the NLP engine on real sampled reviews from the
actual dataset to confirm it holds up outside of hand-crafted examples.
This file is just for manual testing — not part of the pipeline itself.
"""

import pandas as pd
from engine.nlp_engine import extract_aspects

df = pd.read_csv('data/hotel_reviews.csv')
df = df.dropna(subset=['Review_Text'])

sample = df.sample(5, random_state=42)
for idx, row in sample.iterrows():
    print("=" * 70)
    print(f"Hotel: {row['Name']} | Rating: {row['Rating(Out of 10)']}/10")
    print(f"Review: {row['Review_Text'][:200]}")
    print("-" * 70)
    result = extract_aspects(row['Review_Text'])
    for r in result:
        print(f"  {r['aspect']:15s} sentiment={r['sentiment']:+.2f}  conf={r['confidence']:.2f}  ({r['emotion_label']})")
    if not result:
        print("  (no aspects matched)")