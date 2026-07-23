"""
InnSight — Aspect & Sentiment Lexicon
---------------------------------------
Rule-based lexicon powering the offline ABSA (Aspect-Based Sentiment Analysis)
engine. This is intentionally explainable: every score can be traced back to
specific matched words/phrases, which is a feature, not a limitation, for a
trust-focused product.

To upgrade to an LLM-based extractor later (recommended for production/demo
day if API access is available), replace `nlp_engine.extract_aspects()` with
a call to an LLM using ASPECT_KEYWORDS.keys() as the schema — the rest of the
pipeline (aggregation, trust scoring, dashboard) doesn't need to change.
"""

# Each aspect maps to keywords/phrases that signal a review is discussing it.
ASPECT_KEYWORDS = {
    "cleanliness": [
        "clean", "cleanliness", "dirty", "dust", "dusty", "stain", "stained",
        "smell", "smelly", "hygiene", "hygienic", "unclean", "spotless",
        "housekeeping", "mold", "mould", "cockroach", "bed bug", "bedbug",
    ],
    "staff": [
        "staff", "service", "receptionist", "reception", "manager",
        "helpful", "rude", "friendly", "courteous", "hospitality",
        "front desk", "employee", "behaviour", "behavior", "attitude",
    ],
    "food": [
        "food", "breakfast", "restaurant", "buffet", "meal", "menu",
        "dinner", "lunch", "taste", "tasty", "cuisine", "kitchen",
    ],
    "rooms": [
        "room", "bed", "bathroom", "furniture", "interior", "mattress",
        "pillow", "space", "spacious", "small room", "washroom", "toilet",
    ],
    "amenities": [
        "amenities", "gym", "pool", "spa", "elevator", "lift", "parking",
        "ac", "air condition", "tv", "television", "facilities", "wifi",
        "wi-fi", "internet", "hot water", "geyser",
        "pet", "pets", "pet friendly", "pet-friendly", "petfriendly",
        "workout", "dumbbells", "dumbbell", "exercise", "fitness",
        "treadmill", "fitness center", "fitness centre", "weights",
    ],
    "noise": [
        "noise", "noisy", "loud", "quiet", "peaceful", "silent", "traffic sound",
        "construction", "disturbance",
    ],
    "location": [
        "location", "located", "nearby", "distance", "walking distance",
        "metro", "airport", "connectivity", "accessible", "central",
    ],
    "value": [
        "price", "value", "worth", "expensive", "cheap", "budget",
        "affordable", "overpriced", "money", "cost",
    ],
    "check_in": [
        "check-in", "check in", "checkin", "check-out", "check out",
        "checkout", "waiting time", "queue",
    ],
}

# Simple polarity lexicon with intensity weights (-2..+2).
POSITIVE_WORDS = {
    "excellent": 2, "amazing": 2, "wonderful": 2, "fantastic": 2, "best": 2,
    "perfect": 2, "outstanding": 2, "lovely": 1.5, "great": 1.5, "good": 1,
    "nice": 1, "clean": 1, "spacious": 1, "comfortable": 1, "friendly": 1,
    "helpful": 1.5, "convenient": 1, "spotless": 2, "recommend": 1.5,
    "recommended": 1.5, "courteous": 1.5, "peaceful": 1, "quiet": 1,
    "tasty": 1, "affordable": 1, "worth": 1, "impressed": 1.5, "enjoyed": 1,
    "satisfied": 1, "pleasant": 1, "polite": 1,
}

NEGATIVE_WORDS = {
    "worst": -2, "terrible": -2, "horrible": -2, "awful": -2, "disgusting": -2,
    "dirty": -1.5, "rude": -1.5, "poor": -1, "bad": -1, "small": -0.5,
    "noisy": -1, "loud": -1, "smell": -1, "smelly": -1.5, "expensive": -1,
    "overpriced": -1.5, "disappointing": -1.5, "disappointed": -1.5,
    "unclean": -1.5, "broken": -1, "slow": -0.75, "waiting": -0.5,
    "cockroach": -2, "mold": -2, "mould": -2, "uncomfortable": -1,
    "unhelpful": -1.5, "worse": -1.5, "avoid": -1.5, "waste": -1.5,
    "disturbance": -1, "construction": -0.5, "stained": -1,
}

NEGATION_WORDS = {"not", "no", "never", "n't", "without", "hardly", "barely"}

# Words that boost the following sentiment word's polarity
INTENSIFIERS = {"very": 1.4, "extremely": 1.6, "really": 1.2, "so": 1.2, "too": 1.2}