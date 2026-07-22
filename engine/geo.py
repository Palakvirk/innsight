"""
InnSight — Geo Module ("Hotels Near Me")
--------------------------------------------
IMPORTANT HONESTY NOTE: this dataset has no per-hotel GPS coordinates, only
a locality name (e.g. "Paharganj, New Delhi"). So "distance to a hotel"
here really means "distance to that hotel's locality center" — every hotel
in the same area gets the same distance. This is a legitimate, useful
approximation given what the data actually contains, but it is NOT
hotel-level precision, and the frontend should say so rather than implying
exact distances.

Coordinates below are approximate locality centroids for the 31 areas
present in the dataset, based on general Delhi geography.
"""

import math

AREA_COORDS = {
    "Central Delhi, New Delhi": (28.6519, 77.2315),
    "Chanakyapuri, New Delhi": (28.5935, 77.1885),
    "Chattarpur, New Delhi": (28.4930, 77.1770),
    "Connaught Place, New Delhi": (28.6315, 77.2167),
    "Dariyaganj, New Delhi": (28.6465, 77.2410),
    "Dwarka, New Delhi": (28.5921, 77.0460),
    "East Delhi, New Delhi": (28.6280, 77.2950),
    "Greater Kailash 1, New Delhi": (28.5480, 77.2410),
    "Hauz Khas, New Delhi": (28.5494, 77.2001),
    "Janakpuri, New Delhi": (28.6219, 77.0878),
    "Jasola, New Delhi": (28.5490, 77.2900),
    "Kailash Colony, New Delhi": (28.5560, 77.2430),
    "Karol bagh, New Delhi": (28.6519, 77.1909),
    "Mahipalpur, New Delhi": (28.5470, 77.1220),
    "Malviya Nagar, New Delhi": (28.5290, 77.2070),
    "Mayur Vihar Phase 1, New Delhi": (28.6090, 77.2950),
    "Nehru Place, New Delhi": (28.5490, 77.2510),
    "New Delhi": (28.6139, 77.2090),
    "New Friends Colony, New Delhi": (28.5620, 77.2740),
    "North Delhi, New Delhi": (28.7040, 77.2010),
    "Paharganj, New Delhi": (28.6440, 77.2160),
    "Pashim Vihar, New Delhi": (28.6730, 77.1010),
    "Patparganj, New Delhi": (28.6270, 77.2910),
    "Rohini, New Delhi": (28.7495, 77.0640),
    "Safdarjung Enclave, New Delhi": (28.5610, 77.1930),
    "Saket, New Delhi": (28.5245, 77.2066),
    "South Delhi, New Delhi": (28.5245, 77.2066),
    "South West, New Delhi": (28.5921, 77.0460),
    "Sundar Nagar, New Delhi": (28.6020, 77.2410),
    "Vasant Vihar, New Delhi": (28.5680, 77.1590),
    "West Delhi, New Delhi": (28.6650, 77.1000),
}

DELHI_FALLBACK_CENTER = (28.6139, 77.2090)  # used if an area isn't in the map above


def get_area_coords(area: str):
    """Returns (lat, lon) for a known area, or the Delhi-center fallback."""
    return AREA_COORDS.get(area, DELHI_FALLBACK_CENTER)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points, in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def rank_hotels_by_distance(profiles: dict, user_lat: float, user_lon: float, top_n: int = 10):
    """
    Ranks hotels by distance from (user_lat, user_lon) to their AREA's
    centroid (not the individual hotel — see module docstring). Returns a
    list sorted nearest-first, each with an approx_distance_km field.
    """
    results = []
    for profile in profiles.values():
        area = profile["area"]
        area_lat, area_lon = get_area_coords(area)
        dist_km = haversine_km(user_lat, user_lon, area_lat, area_lon)
        results.append({
            "hotel_id": profile["hotel_id"],
            "hotel_name": profile["hotel_name"],
            "area": area,
            "avg_rating": profile["avg_rating"],
            "trust_score": profile["trust"]["trust_score"],
            "approx_distance_km": round(dist_km, 1),
        })
    results.sort(key=lambda r: r["approx_distance_km"])
    return results[:top_n]