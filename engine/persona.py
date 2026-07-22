"""
InnSight — Persona Recommendation Engine (Layer 3)
------------------------------------------------------
Takes an ALREADY-BUILT Hotel Intelligence Profile and re-weights its
aspect_scores according to what different traveler personas care about.
No new NLP, no new trust computation — this is pure re-weighting of data
that already exists, which is why it's cheap to add as a feature.

Each persona defines which aspects matter most to them (weights sum to 1.0
per persona, but don't have to — they're relative importance, not a
probability distribution).
"""

PERSONA_WEIGHTS = {
    "business_traveler": {
        "location": 0.25, "amenities": 0.20, "noise": 0.20,
        "check_in": 0.15, "staff": 0.10, "cleanliness": 0.05,
        "rooms": 0.05, "food": 0.0, "value": 0.0,
    },
    "family": {
        "rooms": 0.25, "amenities": 0.20, "food": 0.15,
        "cleanliness": 0.15, "staff": 0.10, "value": 0.10,
        "noise": 0.05, "location": 0.0, "check_in": 0.0,
    },
    "solo_female": {
        "staff": 0.30, "location": 0.25, "cleanliness": 0.15,
        "check_in": 0.15, "noise": 0.10, "rooms": 0.05,
        "amenities": 0.0, "food": 0.0, "value": 0.0,
    },
    "budget_traveler": {
        "value": 0.35, "cleanliness": 0.20, "location": 0.15,
        "staff": 0.15, "rooms": 0.10, "food": 0.05,
        "amenities": 0.0, "noise": 0.0, "check_in": 0.0,
    },
}

PERSONA_LABELS = {
    "business_traveler": "Business Traveler",
    "family": "Family Traveler",
    "solo_female": "Solo Female Traveler",
    "budget_traveler": "Budget Traveler",
}


def compute_persona_match(profile: dict, persona: str):
    """
    Returns a persona-weighted match for a hotel profile:
    {
        "persona": "Family Traveler",
        "match_stars": 4.2,       # out of 5
        "pros": [...],            # aspects this persona cares about, scoring well
        "cons": [...],            # aspects this persona cares about, scoring poorly
    }
    """
    weights = PERSONA_WEIGHTS.get(persona)
    if weights is None:
        raise ValueError(f"Unknown persona: {persona}. Options: {list(PERSONA_WEIGHTS)}")

    aspect_scores = profile["aspect_scores"]

    weighted_sum = 0.0
    weight_total = 0.0
    pros, cons = [], []

    for aspect, weight in weights.items():
        if weight == 0:
            continue
        stats = aspect_scores.get(aspect)
        if not stats or stats["review_count"] == 0:
            continue  # can't judge what nobody mentioned

        sentiment = stats["avg_sentiment"]
        weighted_sum += sentiment * weight
        weight_total += weight

        label = aspect.replace("_", " ").title()
        if sentiment >= 0.25:
            pros.append(label)
        elif sentiment <= -0.15:
            cons.append(label)

    if weight_total == 0:
        match_stars = 2.5  # neutral default — not enough data for this persona's priorities
    else:
        avg_weighted_sentiment = weighted_sum / weight_total  # back to -1..1 scale
        match_stars = round((avg_weighted_sentiment + 1) * 2.5, 1)  # -1..1 -> 0..5

    return {
        "persona": PERSONA_LABELS.get(persona, persona),
        "match_stars": match_stars,
        "pros": pros,
        "cons": cons,
    }


def compute_all_persona_matches(profile: dict):
    """Convenience: returns a match dict for every defined persona."""
    return {p: compute_persona_match(profile, p) for p in PERSONA_WEIGHTS}


if __name__ == "__main__":
    import json
    with open("output/sample_profiles.json") as f:
        profiles = json.load(f)

    profile = profiles[0]
    print(f"Hotel: {profile['hotel_name']}\n")

    for persona in PERSONA_WEIGHTS:
        match = compute_persona_match(profile, persona)
        print(f"--- {match['persona']} ---")
        print(f"  Match: {match['match_stars']}/5 stars")
        print(f"  Pros: {match['pros']}")
        print(f"  Cons: {match['cons']}")
        print()