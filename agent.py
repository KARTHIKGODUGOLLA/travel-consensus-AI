"""
agent.py - Group Travel Consensus AI

Everything that is not UI lives here. Three LLM calls, one deterministic
validator, one deterministic scorer.

DESIGN RULE
-----------
There is exactly ONE planning function: generate_plan(). Round 1, iteration
rounds, and mid-trip disruption replanning are the SAME call with different
arguments populated. Do not add a second planner.

DATA SHAPES
-----------
PROFILE = {
    "id": str, "name": str, "avatar": str, "can_drive": bool,
    "hard": [str],                                  # non-negotiable
    "soft": [{"pref": str, "weight": float}],       # 0..1
    "learned": [{"pref": str, "old": float, "new": float,
                 "evidence": str, "trip_id": str}],
}

PLAN = {
    "round": int, "summary": str, "reasoning": str,
    "days": [{"date": str, "label": str, "items": [ITEM]}],
    "tradeoffs": [{"description": str, "favored": [id], "disfavored": [id]}],
}

ITEM = {
    "id": str, "time": "HH:MM", "type": "drive"|"meal"|"sight"|"stay"|"work"|"free",
    "title": str, "location": {"name": str, "lat": float, "lng": float},
    "duration_min": int, "driver": id|None, "cost_per_person": float,
    "notes": str, "serves_for": [id],
}

SCORE = {"member_id": str, "score": float,
         "breakdown": [{"pref": str, "status": "met"|"partial"|"unmet", "why": str}]}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).parent
SEED = ROOT / "seed"
FIXTURES = ROOT / "fixtures"

MODEL = os.getenv("TCAI_MODEL", "claude-sonnet-5")
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

STATUS_VALUE = {"met": 1.0, "partial": 0.5, "unmet": 0.0}
MAX_ROUNDS = 3
CONVERGENCE_DELTA = 0.3


# --------------------------------------------------------------------------
# seed loading
# --------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_profiles() -> list[dict]:
    return _read_json(SEED / "profiles.json")


def load_trip_preset() -> dict:
    return _read_json(SEED / "trip.json")


def load_trip1_history() -> dict:
    return _read_json(SEED / "trip1_history.json")


# --------------------------------------------------------------------------
# JSON schemas for tool-use (guarantees well-formed output)
# --------------------------------------------------------------------------

_LOCATION = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "lat": {"type": "number"},
        "lng": {"type": "number"},
    },
    "required": ["name", "lat", "lng"],
}

_ITEM = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "stable short id, e.g. d1i3"},
        "time": {"type": "string", "description": "24h HH:MM start time"},
        "type": {
            "type": "string",
            "enum": ["drive", "meal", "sight", "stay", "work", "free"],
        },
        "title": {"type": "string"},
        "location": _LOCATION,
        "duration_min": {"type": "integer"},
        "driver": {
            "type": "string",
            "description": "member id driving this leg, or empty string if not a drive",
        },
        "cost_per_person": {"type": "number"},
        "notes": {
            "type": "string",
            "description": (
                "Why this item is here and which hard constraints it satisfies. "
                "MUST explicitly name dietary provisions (vegetarian, gluten-free) "
                "and accessibility (step-free / wheelchair accessible) when relevant."
            ),
        },
        "serves_for": {
            "type": "array",
            "items": {"type": "string"},
            "description": "member ids whose preferences this item serves",
        },
    },
    "required": [
        "id", "time", "type", "title", "location",
        "duration_min", "driver", "cost_per_person", "notes", "serves_for",
    ],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "one line, plain language"},
        "reasoning": {
            "type": "string",
            "description": "2-4 sentences on why this plan is a good compromise",
        },
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "label": {"type": "string", "description": "e.g. 'Day 1 - SF to Monterey'"},
                    "items": {"type": "array", "items": _ITEM},
                },
                "required": ["date", "label", "items"],
            },
        },
        "tradeoffs": {
            "type": "array",
            "description": (
                "Every decision where you chose one member's preference over "
                "another's. Be honest and specific."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "favored": {"type": "array", "items": {"type": "string"}},
                    "disfavored": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["description", "favored", "disfavored"],
            },
        },
    },
    "required": ["summary", "reasoning", "days", "tradeoffs"],
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "member_id": {"type": "string"},
                    "breakdown": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "pref": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["met", "partial", "unmet"],
                                },
                                "why": {"type": "string"},
                            },
                            "required": ["pref", "status", "why"],
                        },
                    },
                },
                "required": ["member_id", "breakdown"],
            },
        }
    },
    "required": ["evaluations"],
}

LEARN_SCHEMA = {
    "type": "object",
    "properties": {
        "updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "member_id": {"type": "string"},
                    "pref": {"type": "string", "description": "must match an existing soft pref exactly"},
                    "old": {"type": "number"},
                    "new": {"type": "number"},
                    "evidence": {
                        "type": "string",
                        "description": "cite the specific behaviour and rating that justifies this",
                    },
                },
                "required": ["member_id", "pref", "old", "new", "evidence"],
            },
        }
    },
    "required": ["updates"],
}


# --------------------------------------------------------------------------
# Anthropic plumbing
# --------------------------------------------------------------------------

class AgentError(RuntimeError):
    pass


def api_key_present() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _client():
    from anthropic import Anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise AgentError(
            "ANTHROPIC_API_KEY is not set. Put it in .env, or run with DEMO_MODE=1."
        )
    return Anthropic(api_key=key)


def _tool_call(prompt: str, tool_name: str, schema: dict, max_tokens: int = 8000) -> dict:
    """Force the model through a JSON schema. Returns the tool input dict."""
    resp = _client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        tools=[{"name": tool_name, "description": "Return the structured result.",
                "input_schema": schema}],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    raise AgentError(f"model returned no tool_use block for {tool_name}")


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------

def _effective_soft(member: dict) -> list[dict]:
    """Soft prefs with learned weight adjustments applied."""
    learned = {l["pref"]: l["new"] for l in member.get("learned", [])}
    return [
        {"pref": s["pref"], "weight": learned.get(s["pref"], s["weight"])}
        for s in member["soft"]
    ]


def _member_block(member: dict) -> str:
    lines = [f"### {member['name']} (id: {member['id']})"]
    lines.append(f"- can drive: {'yes' if member['can_drive'] else 'NO'}")
    lines.append("- HARD constraints (non-negotiable):")
    for h in member["hard"]:
        lines.append(f"    * {h}")
    lines.append("- SOFT preferences (weight 0-1, higher = matters more):")
    for s in _effective_soft(member):
        lines.append(f"    * [{s['weight']:.1f}] {s['pref']}")
    if member.get("learned"):
        lines.append("- LEARNED from past trips (already applied above):")
        for l in member["learned"]:
            lines.append(
                f"    * '{l['pref']}' {l['old']:.1f} -> {l['new']:.1f} ({l['evidence']})"
            )
    return "\n".join(lines)


def _all_hard(members: list[dict]) -> list[str]:
    out = []
    for m in members:
        for h in m["hard"]:
            out.append(f"{m['name']}: {h}")
    return out


def build_plan_prompt(
    trip: dict,
    members: list[dict],
    trip_constraints: dict[str, str] | None = None,
    feedback_history: list[dict] | None = None,
    ledger: dict[str, int] | None = None,
    disruption: str | None = None,
    locked_items: list[dict] | None = None,
    violations: list[str] | None = None,
    round_no: int = 1,
) -> str:
    p: list[str] = []
    p.append(
        "You are planning a multi-day group road trip. Produce ONE complete "
        "itinerary. Be concrete: real places, real coordinates, realistic drive "
        "times. Never invent a place that does not exist."
    )
    p.append(
        f"\n## TRIP\n"
        f"- {trip['origin']} to {trip['destination']}\n"
        f"- Route: {trip.get('route_hint', 'open')}\n"
        f"- Dates: {trip['start_date']} to {trip['end_date']} ({trip['days']} days)\n"
        f"- Budget: ${trip['budget_per_person']} per person, all in\n"
        f"- Party: {len(members)} people\n"
        f"- Notes: {trip.get('notes', '')}"
    )

    p.append("\n## MEMBERS\n" + "\n\n".join(_member_block(m) for m in members))

    hard = _all_hard(members)
    p.append(
        "\n## HARD CONSTRAINTS - violating ANY of these invalidates the entire plan\n"
        + "\n".join(f"- {h}" for h in hard)
        + "\n\nEvery meal item's `notes` MUST state how each dietary restriction is "
          "handled. Every item's `notes` MUST state accessibility when a member "
          "requires it. Never assign a driver who cannot drive."
    )

    if trip_constraints:
        named = [
            f"- {c}"
            for c in trip_constraints.values()
            if c and c.strip()
        ]
        if named:
            p.append("\n## EXTRA NON-NEGOTIABLES FOR THIS TRIP\n" + "\n".join(named))

    if feedback_history:
        blocks = []
        for rnd in feedback_history:
            lines = [f"### Round {rnd['round']} scored {rnd['group_score']:.1f}/10"]
            for fb in rnd["feedback"]:
                tagged = ", ".join(fb.get("tags", [])) or "nothing specific"
                lines.append(
                    f"- {fb['name']} rated it {fb['score']}/10. "
                    f"Disliked: {tagged}. \"{fb.get('note', '')}\""
                )
            blocks.append("\n".join(lines))
        p.append(
            "\n## PREVIOUS ROUNDS - fix these complaints\n"
            + "\n\n".join(blocks)
            + "\n\nChange what was criticised. KEEP what scored well - do not "
              "rewrite the whole trip, revise it."
        )

    if ledger and any(v for v in ledger.values()):
        overruled = [
            f"- {mid} has been overruled {abs(v)} time(s)"
            for mid, v in sorted(ledger.items(), key=lambda kv: kv[1])
            if v < 0
        ]
        if overruled:
            p.append(
                "\n## FAIRNESS LEDGER\n"
                + "\n".join(overruled)
                + "\nWhen this round forces a tradeoff, break the tie in favour of "
                  "whoever has been overruled most. Nobody should lose twice running."
            )

    if disruption:
        p.append(f"\n## MID-TRIP DISRUPTION\n{disruption}")
        if locked_items:
            done = ", ".join(f"{i['title']} ({i['time']})" for i in locked_items)
            p.append(f"\nALREADY HAPPENED - do not move or remove these: {done}")
        p.append(
            "\nReturn ONE revised plan covering the REMAINDER of the trip only. "
            "Do not restart from the beginning. Do not propose alternatives - the "
            "group is on the road and needs a single answer. Keep as much of the "
            "original plan intact as the disruption allows."
        )

    if violations:
        p.append(
            "\n## YOUR PREVIOUS ATTEMPT WAS REJECTED\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\nFix every one of these. Do not repeat them."
        )

    p.append(
        f"\nThis is round {round_no}. Emit the plan via the emit_plan tool. "
        "Populate `tradeoffs` honestly - every time you favoured one member over "
        "another, say so and name them by id."
    )
    return "\n".join(p)


# --------------------------------------------------------------------------
# 1. PLANNER - the one function
# --------------------------------------------------------------------------

def generate_plan(
    trip: dict,
    members: list[dict],
    trip_constraints: dict[str, str] | None = None,
    feedback_history: list[dict] | None = None,
    ledger: dict[str, int] | None = None,
    disruption: str | None = None,
    locked_items: list[dict] | None = None,
    round_no: int = 1,
    max_retries: int = 2,
) -> tuple[dict, list[str]]:
    """
    Returns (plan, remaining_violations).

    Round 1        -> feedback_history empty
    Rounds 2, 3    -> feedback_history populated
    Mid-trip       -> disruption + locked_items populated
    """
    if DEMO_MODE:
        # Live, the planner reads each member's learned weights straight out of
        # the prompt and its first attempt is already better. The canned
        # fixtures cannot adapt, so we skip the naive first draft instead --
        # same effect, same story, no API call.
        smarter = any(m.get("learned") for m in members)
        plan = _demo_plan(round_no, disruption, skip_ahead=1 if smarter else 0)
        return plan, validate_hard(plan, members)

    violations: list[str] = []
    plan: dict = {}

    for attempt in range(max_retries + 1):
        prompt = build_plan_prompt(
            trip=trip,
            members=members,
            trip_constraints=trip_constraints,
            feedback_history=feedback_history,
            ledger=ledger,
            disruption=disruption,
            locked_items=locked_items,
            violations=violations if attempt else None,
            round_no=round_no,
        )
        plan = _tool_call(prompt, "emit_plan", PLAN_SCHEMA)
        plan["round"] = round_no
        violations = validate_hard(plan, members)
        if not violations:
            break

    return plan, violations


# --------------------------------------------------------------------------
# 2. VALIDATOR - pure Python, no LLM. This is what protects hard constraints.
# --------------------------------------------------------------------------

def _minutes(hhmm: str) -> int:
    try:
        h, m = hhmm.strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 0


def iter_items(plan: dict):
    for day in plan.get("days", []):
        for item in day.get("items", []):
            yield day, item


def validate_hard(plan: dict, members: list[dict]) -> list[str]:
    """Deterministic hard-constraint check. Empty list == plan is legal."""
    v: list[str] = []
    by_id = {m["id"]: m for m in members}

    non_drivers = {m["id"] for m in members if not m["can_drive"]}
    veg = [m for m in members if any("vegetarian" in h.lower() for h in m["hard"])]
    gf = [m for m in members if any("gluten" in h.lower() for h in m["hard"])]
    access = [
        m for m in members
        if any(("wheelchair" in h.lower() or "step-free" in h.lower()) for h in m["hard"])
    ]
    night_limited = {
        m["id"]: m for m in members
        if any(("after sunset" in h.lower() or "night-vision" in h.lower())
               for h in m["hard"])
    }

    meals_seen = 0
    total_cost = 0.0

    for day, item in iter_items(plan):
        notes = (item.get("notes") or "").lower()
        title = item.get("title", "?")
        driver = (item.get("driver") or "").strip()
        total_cost += float(item.get("cost_per_person") or 0)

        # driver legality
        if driver:
            if driver in non_drivers:
                name = by_id.get(driver, {}).get("name", driver)
                v.append(f"{name} is assigned to drive '{title}' but cannot drive.")
            if driver not in by_id:
                v.append(f"Unknown driver id '{driver}' on '{title}'.")
            if driver in night_limited and _minutes(item.get("time", "00:00")) >= 19 * 60 + 30:
                name = night_limited[driver]["name"]
                v.append(
                    f"{name} is driving '{title}' at {item.get('time')} - "
                    f"after their 19:30 night-driving limit."
                )

        # dietary
        if item.get("type") == "meal":
            meals_seen += 1
            for m in veg:
                if "vegetarian" not in notes and "vegan" not in notes:
                    v.append(
                        f"Meal '{title}' does not state a vegetarian option "
                        f"({m['name']} is vegetarian)."
                    )
                    break
            for m in gf:
                if "gluten" not in notes:
                    v.append(
                        f"Meal '{title}' does not state a gluten-free option "
                        f"({m['name']} is gluten-free)."
                    )
                    break

        # accessibility
        if access and item.get("type") in {"meal", "sight", "stay", "work"}:
            if not any(k in notes for k in ("accessible", "step-free", "step free", "wheelchair")):
                v.append(
                    f"'{title}' does not state wheelchair accessibility "
                    f"({access[0]['name']} uses a wheelchair)."
                )

    if meals_seen == 0 and (veg or gf):
        v.append("Plan contains no meal stops at all, but members have dietary needs.")

    return v


# --------------------------------------------------------------------------
# 3. SCORER - LLM judges each pref, Python does the arithmetic
# --------------------------------------------------------------------------

_STOP = {
    "wants", "want", "prefers", "prefer", "likes", "like", "enjoys", "enjoy",
    "and", "the", "a", "an", "or", "to", "in", "at", "of", "for", "with",
    "least", "one", "each", "per", "than", "not", "no", "be", "is", "are",
    "rather", "quick", "more", "any", "single", "hour", "good", "just",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in _STOP}


def _doc_freq(members: list[dict]) -> dict[str, int]:
    """How many preferences across the whole group contain each word.

    Words that show up in everybody's preferences ('stops', 'local') carry
    almost no signal, so we weight each keyword by 1/frequency.
    """
    df: dict[str, int] = {}
    for m in members:
        for s in m["soft"]:
            for w in _keywords(s["pref"]):
                df[w] = df.get(w, 0) + 1
    return df


def _item_text(item: dict) -> set[str]:
    return _keywords(
        f"{item.get('title','')} {item.get('notes','')} {item.get('type','')} "
        f"{item.get('location',{}).get('name','')}"
    )


def _heuristic_breakdown(plan: dict, member: dict, df: dict[str, int] | None = None) -> list[dict]:
    """Offline scorer.

    For each preference, find the single best-matching itinerary item using
    frequency-weighted keyword overlap. An item that explicitly lists this
    member in `serves_for` counts for more. Deliberately strict: 'met' needs a
    genuinely strong match, not an incidental word collision.
    """
    df = df or {}
    items = [i for _, i in iter_items(plan)]
    out = []

    for s in _effective_soft(member):
        kws = _keywords(s["pref"])
        total = sum(1.0 / max(df.get(w, 1), 1) for w in kws) or 1.0

        best, best_item, best_hits = 0.0, None, set()
        for i in items:
            hits = kws & _item_text(i)
            if not hits:
                continue
            weight = sum(1.0 / max(df.get(w, 1), 1) for w in hits)
            if member["id"] in (i.get("serves_for") or []):
                weight *= 1.6
            ratio = weight / total
            if ratio > best:
                best, best_item, best_hits = ratio, i, hits

        if best >= 0.55:
            status = "met"
        elif best >= 0.22:
            status = "partial"
        else:
            status = "unmet"

        if best_item is not None:
            why = (
                f"offline scorer: closest match was '{best_item['title']}' "
                f"({', '.join(sorted(best_hits)[:3])})"
            )
        else:
            why = "offline scorer: nothing in the itinerary matches this preference"

        out.append({"pref": s["pref"], "status": status, "why": why})
    return out


def score_plan(plan: dict, members: list[dict]) -> dict[str, dict]:
    """Returns {member_id: SCORE}."""
    df = _doc_freq(members)
    if DEMO_MODE or not api_key_present():
        evals = _demo_evals(plan)
        if not evals:
            evals = [
                {"member_id": m["id"], "breakdown": _heuristic_breakdown(plan, m, df)}
                for m in members
            ]
    else:
        compact = {
            "days": [
                {
                    "label": d.get("label"),
                    "items": [
                        {
                            "time": i.get("time"),
                            "type": i.get("type"),
                            "title": i.get("title"),
                            "notes": i.get("notes"),
                            "cost_per_person": i.get("cost_per_person"),
                            "driver": i.get("driver"),
                        }
                        for i in d.get("items", [])
                    ],
                }
                for d in plan.get("days", [])
            ]
        }
        member_block = "\n\n".join(
            f"### {m['name']} (id: {m['id']})\n"
            + "\n".join(f"- {s['pref']}" for s in _effective_soft(m))
            for m in members
        )
        prompt = (
            "Judge how well this itinerary serves each member's SOFT preferences.\n\n"
            "For EACH member and EACH of their preferences return one of:\n"
            "  met     - the plan clearly and fully serves it\n"
            "  partial - partly served, or served once when they wanted it throughout\n"
            "  unmet   - not served, or actively contradicted\n\n"
            "Give a one-sentence reason citing a specific item. Be strict: 'met' "
            "means genuinely satisfied, not merely 'not violated'.\n"
            "Do NOT compute an overall number - that is calculated elsewhere.\n\n"
            f"## ITINERARY\n{json.dumps(compact, indent=1)}\n\n"
            f"## MEMBERS AND THEIR PREFERENCES\n{member_block}\n\n"
            "Return the `pref` text exactly as given."
        )
        try:
            result = _tool_call(prompt, "emit_scores", SCORE_SCHEMA, max_tokens=6000)
            evals = result.get("evaluations", [])
        except Exception:
            evals = [
                {"member_id": m["id"], "breakdown": _heuristic_breakdown(plan, m)}
                for m in members
            ]

    by_id = {m["id"]: m for m in members}
    scores: dict[str, dict] = {}
    for ev in evals:
        mid = ev["member_id"]
        member = by_id.get(mid)
        if not member:
            continue
        scores[mid] = {
            "member_id": mid,
            "name": member["name"],
            "score": person_score(ev.get("breakdown", []), _effective_soft(member)),
            "breakdown": ev.get("breakdown", []),
        }

    # any member the model skipped falls back to the heuristic
    for m in members:
        if m["id"] not in scores:
            bd = _heuristic_breakdown(plan, m, df)
            scores[m["id"]] = {
                "member_id": m["id"],
                "name": m["name"],
                "score": person_score(bd, _effective_soft(m)),
                "breakdown": bd,
            }
    return scores


def person_score(breakdown: list[dict], soft: list[dict]) -> float:
    """10 * sum(weight * status) / sum(weight). Computed here, never by the LLM."""
    weights = {s["pref"]: s["weight"] for s in soft}
    num = den = 0.0
    for b in breakdown:
        w = weights.get(b["pref"])
        if w is None:  # tolerate slight rewording from the model
            match = next((k for k in weights if k[:25].lower() == b["pref"][:25].lower()), None)
            w = weights.get(match, 0.5)
        num += w * STATUS_VALUE.get(b.get("status", "unmet"), 0.0)
        den += w
    return round(10 * num / den, 1) if den else 0.0


def summarise(scores: dict[str, dict]) -> dict:
    vals = [s["score"] for s in scores.values()]
    if not vals:
        return {"group": 0.0, "floor": 0.0, "floor_name": "-"}
    lowest = min(scores.values(), key=lambda s: s["score"])
    return {
        "group": round(sum(vals) / len(vals), 1),
        "floor": lowest["score"],
        "floor_name": lowest["name"],
        "floor_id": lowest["member_id"],
    }


# --------------------------------------------------------------------------
# 4. FAIRNESS LEDGER
# --------------------------------------------------------------------------

def update_ledger(ledger: dict[str, int], plan: dict) -> dict[str, int]:
    out = dict(ledger)
    for t in plan.get("tradeoffs", []):
        for mid in t.get("favored", []):
            out[mid] = out.get(mid, 0) + 1
        for mid in t.get("disfavored", []):
            out[mid] = out.get(mid, 0) - 1
    return out


# --------------------------------------------------------------------------
# 5. LEARNER - the hero. Stated preference vs revealed behaviour.
# --------------------------------------------------------------------------

def update_profiles(members: list[dict], history: dict) -> list[dict]:
    """Returns a list of weight updates. Does not mutate `members`."""
    if DEMO_MODE or not api_key_present():
        return _demo_updates(members)

    ev_by_member: dict[str, list[str]] = {}
    for e in history.get("events", []):
        ev_by_member.setdefault(e["member"], []).append(
            f"{e['action'].upper()}: {e['item']} - {e['note']}"
        )
    for r in history.get("ratings", []):
        ev_by_member.setdefault(r["member"], []).append(
            f"RATED {r['score']}/10: {r['item']} - \"{r['note']}\""
        )

    blocks = []
    for m in members:
        lines = [f"### {m['name']} (id: {m['id']})", "Stated soft preferences:"]
        for s in m["soft"]:
            lines.append(f"  - [{s['weight']:.1f}] {s['pref']}")
        lines.append("What actually happened:")
        for line in ev_by_member.get(m["id"], ["(no data)"]):
            lines.append(f"  - {line}")
        blocks.append("\n".join(lines))

    prompt = (
        "A group finished a trip. For each member, compare what they SAID they "
        "wanted against what they ACTUALLY did and how they rated it.\n\n"
        "Where stated preference diverges from revealed behaviour, adjust the "
        "preference WEIGHT (0.0-1.0). Raise it when they sought something out or "
        "rated it highly; lower it when they skipped it or rated it poorly.\n\n"
        "RULES:\n"
        "- Only emit an update when the evidence is clear and specific.\n"
        "- Cite the actual behaviour and rating in `evidence`.\n"
        "- `pref` must match the stated preference text EXACTLY.\n"
        "- At most 2 updates per member.\n"
        "- If a member's behaviour matched their stated preferences, emit NOTHING "
        "for them. An empty result for a consistent member is the correct answer.\n\n"
        f"## TRIP\n{history.get('title')} ({history.get('dates')})\n\n"
        f"## MEMBERS\n" + "\n\n".join(blocks)
    )

    try:
        result = _tool_call(prompt, "emit_updates", LEARN_SCHEMA, max_tokens=4000)
        return result.get("updates", [])
    except Exception:
        return _demo_updates(members)


def apply_updates(members: list[dict], updates: list[dict], trip_id: str = "trip1") -> list[dict]:
    """Returns new member dicts with `learned` populated."""
    out = [json.loads(json.dumps(m)) for m in members]
    by_id = {m["id"]: m for m in out}
    for u in updates:
        m = by_id.get(u["member_id"])
        if not m:
            continue
        m.setdefault("learned", []).append({
            "pref": u["pref"],
            "old": float(u["old"]),
            "new": float(u["new"]),
            "evidence": u["evidence"],
            "trip_id": trip_id,
        })
    return out


# --------------------------------------------------------------------------
# 6. CONVERGENCE
# --------------------------------------------------------------------------

def convergence(round_no: int, group: float, prev_group: float | None) -> tuple[bool, str]:
    if round_no >= MAX_ROUNDS:
        return True, (
            f"Round {round_no} is the cap. Further rounds trade one person's "
            "satisfaction for another's without raising the group. This is the "
            "best available compromise."
        )
    if prev_group is not None and abs(group - prev_group) < CONVERGENCE_DELTA:
        return True, (
            f"The group score moved only {abs(group - prev_group):.1f} points since "
            "the last round. The plan has converged - more rounds will not help."
        )
    return False, ""


# --------------------------------------------------------------------------
# 7. EXTERNAL DATA - map, facts, weather. All optional, all fail soft.
# --------------------------------------------------------------------------

def static_map_url(plan: dict, size: str = "640x420") -> str | None:
    pts = [
        (i["location"]["lat"], i["location"]["lng"])
        for _, i in iter_items(plan)
        if i.get("location", {}).get("lat")
    ]
    if not pts or not GOOGLE_MAPS_KEY:
        return None
    seen, uniq = set(), []
    for p in pts:
        k = (round(p[0], 3), round(p[1], 3))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    uniq = uniq[:20]
    markers = "&".join(
        f"markers=color:0xE8613C%7Clabel:{i+1}%7C{lat},{lng}"
        for i, (lat, lng) in enumerate(uniq)
    )
    path = "path=color:0x2E5FE8AA%7Cweight:4%7C" + "%7C".join(
        f"{lat},{lng}" for lat, lng in uniq
    )
    return (
        f"https://maps.googleapis.com/maps/api/staticmap?size={size}&scale=2"
        f"&{markers}&{path}&key={GOOGLE_MAPS_KEY}"
    )


_FACT_CACHE: dict[tuple, str] = {}


def fun_fact(lat: float, lng: float) -> str | None:
    """Nearest Wikipedia article to a coordinate, first sentence. Keyless."""
    key = (round(lat, 3), round(lng, 3))
    if key in _FACT_CACHE:
        return _FACT_CACHE[key]
    try:
        geo = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "geosearch", "format": "json",
                "gscoord": f"{lat}|{lng}", "gsradius": 10000, "gslimit": 5,
            },
            headers={"User-Agent": "travel-consensus-ai/0.1"},
            timeout=6,
        ).json()
        pages = geo.get("query", {}).get("geosearch", [])
        if not pages:
            return None
        title = pages[0]["title"]
        summary = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            headers={"User-Agent": "travel-consensus-ai/0.1"},
            timeout=6,
        ).json()
        extract = (summary.get("extract") or "").strip()
        if not extract:
            return None
        first = re.split(r"(?<=[.!?])\s", extract)[0]
        fact = f"{first}"
        _FACT_CACHE[key] = fact
        return fact
    except Exception:
        return None


WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "heavy showers", 82: "violent showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "severe thunderstorm",
}


def _fetch_weather(lat: float, lng: float, date: str) -> dict | None:
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lng,
                "daily": "precipitation_probability_max,temperature_2m_max,weathercode,windspeed_10m_max",
                "timezone": "auto", "start_date": date, "end_date": date,
            },
            timeout=6,
        ).json()
        d = r.get("daily", {})
        if not d.get("time"):
            return None
        return {
            "date": d["time"][0],
            "rain_chance": d["precipitation_probability_max"][0],
            "temp_max": d["temperature_2m_max"][0],
            "wind_max": (d.get("windspeed_10m_max") or [None])[0],
            "code": d["weathercode"][0],
            "desc": WEATHER_CODES.get(d["weathercode"][0], "unsettled"),
            "approx": False,
        }
    except Exception:
        return None


def weather_for(lat: float, lng: float, date: str) -> dict | None:
    """Open-Meteo. Keyless.

    The free forecast horizon is ~15 days. If the requested date is beyond it
    we clamp to the furthest available day and flag the result as approximate,
    rather than returning nothing.
    """
    import datetime as _dt

    got = _fetch_weather(lat, lng, date)
    if got:
        return got

    try:
        want = _dt.date.fromisoformat(date)
    except Exception:
        return None

    today = _dt.date.today()
    for horizon in (15, 14, 10, 7, 3, 0):
        cand = today + _dt.timedelta(days=horizon)
        if cand > want:
            continue
        got = _fetch_weather(lat, lng, cand.isoformat())
        if got:
            got["approx"] = True
            got["requested"] = date
            return got
    return None


# --------------------------------------------------------------------------
# 8. DEMO MODE - offline fallback so the demo can never hard-fail
# --------------------------------------------------------------------------

def _demo_plan(round_no: int, disruption: str | None = None, skip_ahead: int = 0) -> dict:
    try:
        plans = _read_json(FIXTURES / "demo_plans.json")
    except Exception:
        return {"round": round_no, "summary": "demo fixture missing",
                "reasoning": "", "days": [], "tradeoffs": []}
    key = "disrupted" if disruption else f"round{min(round_no + skip_ahead, MAX_ROUNDS)}"
    plan = plans.get(key) or plans.get("round1", {})
    plan = json.loads(json.dumps(plan))
    plan["round"] = round_no
    plan["_demo_key"] = key
    return plan


def _demo_evals(plan: dict) -> list[dict]:
    """Pre-written breakdowns for the canned demo plans, so the offline curve
    matches the story exactly. Returns [] for any plan we have no fixture for,
    which sends the caller to the keyword heuristic instead."""
    try:
        scores = _read_json(FIXTURES / "demo_scores.json")
    except Exception:
        return []

    key = plan.get("_demo_key")
    if not key:
        # Plan came from somewhere other than _demo_plan() (a REPL, a test).
        # Recover the key by matching the summary against the fixtures so the
        # numbers stay consistent with what the app shows.
        try:
            plans = _read_json(FIXTURES / "demo_plans.json")
        except Exception:
            return []
        summary = (plan.get("summary") or "").strip()
        key = next(
            (k for k, v in plans.items() if (v.get("summary") or "").strip() == summary and summary),
            None,
        )
        if not key:
            return []

    return scores.get(key, {}).get("evaluations", [])


def _demo_updates(members: list[dict]) -> list[dict]:
    try:
        return _read_json(FIXTURES / "demo_updates.json")
    except Exception:
        return []
