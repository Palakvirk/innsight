"""
InnSight — API Layer
------------------------
Wraps the existing engine (ingestion -> NLP -> trust -> aggregation ->
persona -> RAG) as REST endpoints so the React frontend has something to
fetch from. IMPORTANT: this file adds ZERO new intelligence — it's a thin
HTTP wrapper around functions you already built and tested in Phases 1-7.
That's the whole point of the "one engine, many views" architecture: the
frontend is just another view.

All hotel profiles are computed ONCE at startup and cached in memory
(reasonable at this dataset's scale — ~570 hotels processes in a couple
seconds). In a real production system this would instead be a scheduled
job writing to a database, but for a hackathon demo, in-memory is honest
and fast.

Run: python3 api.py
Then visit http://localhost:8000/docs for interactive API docs (FastAPI
gives you this for free — worth showing judges, it doubles as live
documentation of your own system).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from engine.ingest import load_raw_reviews
from engine.aggregate import process_all_reviews, build_hotel_profile, ASPECT_LIST
from engine.persona import compute_persona_match, compute_all_persona_matches, PERSONA_WEIGHTS
from engine.rag_chat import ask_hotel_question
from engine.priority_match import rank_hotels_by_priorities
from engine.geo import rank_hotels_by_distance

app = FastAPI(title="InnSight API")

# Allow the React dev server (usually localhost:5173 or :3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a hackathon demo; tighten for real production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Startup: build every hotel's profile once, cache in memory ----
_state = {}


@app.on_event("startup")
def build_all_profiles():
    print("Loading and processing all reviews...")
    df = load_raw_reviews("data/hotel_reviews.csv")
    processed_df = process_all_reviews(df)

    from engine.aggregate import _aspect_scores_for_hotel
    hotel_ids = processed_df["hotel_id"].unique()
    area_of_hotel = processed_df.groupby("hotel_id")["area"].first()

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

    profiles = {}
    for hid in hotel_ids:
        area = area_of_hotel[hid]
        cohort = area_cohorts.get(area, {})
        profile = build_hotel_profile(hid, processed_df, city_cohort_df=cohort)
        profiles[int(hid)] = profile

    _state["processed_df"] = processed_df
    _state["profiles"] = profiles
    print(f"Ready — {len(profiles)} hotel profiles cached in memory.")


# ---- Endpoints ----

@app.get("/hotels")
def list_hotels():
    """List all hotels with basic info — powers a hotel picker in the frontend."""
    profiles = _state["profiles"]
    return [
        {
            "hotel_id": p["hotel_id"],
            "hotel_name": p["hotel_name"],
            "area": p["area"],
            "avg_rating": p["avg_rating"],
            "review_count": p["review_count"],
            "trust_score": p["trust"]["trust_score"],
        }
        for p in profiles.values()
    ]

@app.get("/meta")
def get_meta():
    """Dataset scope info — powers the 'coverage' banner and stat row in the frontend."""
    profiles = _state["profiles"]
    total_reviews = sum(p["review_count"] for p in profiles.values())
    return {
        "city": "New Delhi",
        "hotel_count": len(profiles),
        "review_count": total_reviews,
        "note": "InnSight currently covers hotels in New Delhi only.",
    }

@app.get("/search")
def search_hotels(q: str):
    """
    Simple substring search over hotel names/areas. Returns matches plus
    dataset scope info, so the frontend can distinguish "no matches for your
    search" from "this city/hotel isn't in our dataset at all."
    """
    profiles = _state["profiles"]
    q_lower = q.lower().strip()
    matches = [
        {
            "hotel_id": p["hotel_id"],
            "hotel_name": p["hotel_name"],
            "area": p["area"],
            "avg_rating": p["avg_rating"],
            "trust_score": p["trust"]["trust_score"],
        }
        for p in profiles.values()
        if q_lower in p["hotel_name"].lower() or q_lower in p["area"].lower()
    ]
    return {
        "query": q,
        "results": matches[:20],
        "total_hotels_in_dataset": len(profiles),
        "city_scope": "New Delhi",
    }


class PriorityMatchRequest(BaseModel):
    text: str


@app.post("/match")
def match_by_priorities(req: PriorityMatchRequest):
    """Freeform 'what do you want' search -> ranked hotel suggestions."""
    profiles = _state["profiles"]
    return rank_hotels_by_priorities(profiles, req.text, top_n=5)


@app.get("/hotels/{hotel_id}")
def get_hotel_profile(hotel_id: int):
    """Full Hotel Intelligence Profile — powers the main dashboard view."""
    profile = _state["profiles"].get(hotel_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Hotel not found")
    return profile


@app.get("/hotels/{hotel_id}/personas")
def get_persona_matches(hotel_id: int):
    """All 4 persona matches for this hotel."""
    profile = _state["profiles"].get(hotel_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Hotel not found")
    return compute_all_persona_matches(profile)


class ChatRequest(BaseModel):
    question: str


@app.post("/hotels/{hotel_id}/chat")
def chat_about_hotel(hotel_id: int, req: ChatRequest):
    """RAG chat endpoint — grounds the answer in this hotel's real reviews."""
    profile = _state["profiles"].get(hotel_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Hotel not found")

    processed_df = _state["processed_df"]
    hotel_reviews = processed_df[processed_df["hotel_id"] == hotel_id]
    result = ask_hotel_question(req.question, hotel_reviews, profile["hotel_name"])
    return result

class NearbyRequest(BaseModel):
    latitude: float
    longitude: float


@app.post("/nearby")
def hotels_near_me(req: NearbyRequest):
    """
    'Hotels near me' — ranks by distance from the user's coordinates to
    each hotel's LOCALITY center (not exact hotel GPS, which this dataset
    doesn't have). Honest about that limitation in the response itself.
    """
    profiles = _state["profiles"]
    results = rank_hotels_by_distance(profiles, req.latitude, req.longitude, top_n=10)
    return {
        "results": results,
        "note": "Distances are approximate — based on the hotel's locality, not its exact address.",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)