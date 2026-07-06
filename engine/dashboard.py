"""
InnSight — Dashboard Formatter (Layer 3 views)
--------------------------------------------------
IMPORTANT DESIGN NOTE: every function here takes an already-built Hotel
Intelligence Profile (from aggregate.build_hotel_profile) and only FORMATS
it for display. No new computation happens here. This is the payoff of the
"one engine, many views" architecture — if you added a 10th dashboard view
tomorrow, it would just be another formatter function reading the same
profile object, never touching raw reviews again.
"""


def render_trust_card(profile: dict) -> str:
    t = profile["trust"]
    lines = [
        f"┌─────────────────────────────────────┐",
        f"│  {profile['hotel_name'][:35]:<35}│",
        f"│  Rating: {profile['avg_rating']}/10   Trust Score: {t['trust_score']}/100  │",
        f"│  Reliability: {t['reliability']:<10}                │",
        f"├─────────────────────────────────────┤",
        f"│  Trust Score Breakdown               │",
    ]
    for component, value in t["breakdown"].items():
        label = component.replace("_", " ").title()
        lines.append(f"│   {label:<20} {value:>5}          │")
    lines.append("└─────────────────────────────────────┘")
    return "\n".join(lines)


def render_radar(profile: dict) -> str:
    """Text-bar version of the radar chart — sentiment mapped from -1..1 to a 0-10 bar."""
    lines = ["Aspect Scores (out of 10)", ""]
    for aspect, stats in profile["aspect_scores"].items():
        if stats["review_count"] == 0:
            continue
        score_10 = round((stats["avg_sentiment"] + 1) * 5, 1)  # -1..1 -> 0..10
        bar_len = int(score_10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        flag = "  ⚠ low" if score_10 < 4.5 else ""
        lines.append(f"{aspect.replace('_',' ').title():15s} {bar}  {score_10}{flag}")
    return "\n".join(lines)


def render_booking_risk(profile: dict) -> str:
    r = profile["booking_risk"]
    icon = "⚠" if r["level"] != "Low" else "✓"
    lines = [
        f"┌─────────────────────────────────────┐",
        f"│  {icon} Booking Risk : {r['level']:<10}          │",
        f"├─────────────────────────────────────┤",
        f"│  Reasons:                            │",
    ]
    for reason in r["reasons"]:
        lines.append(f"│   • {reason[:33]:<33}│")
    lines.append("└─────────────────────────────────────┘")
    return "\n".join(lines)


def render_deal_breakers(profile: dict) -> str:
    if not profile["deal_breakers"]:
        return "No significant deal breakers found."
    lines = [
        "┌─────────────────────────────────────┐",
        "│  ⚠ Potential Deal Breakers           │",
        "├─────────────────────────────────────┤",
    ]
    for i, d in enumerate(profile["deal_breakers"], 1):
        label = d["aspect"].replace("_", " ").title()
        lines.append(f"│  {i}. {label:<20} ({d['mention_count']} mentions)│")
    lines.append("└─────────────────────────────────────┘")
    return "\n".join(lines)


def render_should_i_book(profile: dict) -> str:
    s = profile["should_i_book"]
    icon = "✅" if s["decision"] == "Book" else "⚠"
    lines = [
        "┌─────────────────────────────────────┐",
        f"│  {icon} Recommendation: {s['decision']:<18}│",
        "├─────────────────────────────────────┤",
        "│  Why:                                │",
    ]
    for reason in s["reasons"]:
        lines.append(f"│   • {reason[:33]:<33}│")
    lines.append("└─────────────────────────────────────┘")
    return "\n".join(lines)


def render_full_dashboard(profile: dict) -> str:
    """Combine every view into one full-page render, in demo order."""
    sections = [
        render_trust_card(profile),
        "",
        render_radar(profile),
        "",
        render_booking_risk(profile),
        "",
        render_deal_breakers(profile),
        "",
        render_should_i_book(profile),
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    import json
    with open("output/sample_profiles.json") as f:
        profiles = json.load(f)
    print(render_full_dashboard(profiles[0]))