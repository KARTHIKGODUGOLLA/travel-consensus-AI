# Group Travel Consensus AI — 6-Hour Build Plan (Python / Streamlit)

**Team:** Karthik (all code, Python) + teammate (seed data, fixtures, demo)
**Time:** 6 hours, cold start, no prep
**Heroes:** cross-trip profile learning · live disruption replanning

---

## 1. Stack

| Concern | Choice | Why |
|---|---|---|
| UI | **Streamlit** | Pure Python. Sliders, progress bars, tabs are one-liners. Zero HTML/CSS/JS. |
| LLM | `anthropic` SDK, `claude-sonnet-5` | Tool-use for structured JSON |
| State | `st.session_state` | This *is* your database. Nothing else needed. |
| Map | Google **Static** Maps | One URL string → `st.image(url)`. No JS. |
| Weather | Open-Meteo | Keyless, free |
| Facts | Wikipedia geosearch | Keyless, free |
| Hosting | **None — run on localhost** | Judges watch your screen. Deployment is pure risk. |

```bash
pip install streamlit anthropic requests
streamlit run app.py
```

### Files — only four

```
app.py              # entire UI, 3 tabs
agent.py            # the 3 LLM calls + validator + scorer
seed/profiles.json          # teammate writes
seed/trip1_history.json     # teammate writes
fixtures/route.json         # teammate writes
```

No `pages/` directory, no multipage routing. **Use `st.tabs()`** — simpler, and it demos better because you never navigate away.

---

## 2. Streamlit gotchas — read before writing code

**Streamlit reruns your entire script top to bottom on every widget interaction.** Every button click, every slider drag. This is the #1 way people lose an hour.

Consequences:

```python
# WRONG — fires an LLM call on every single click
plan = generate_plan(ctx)

# RIGHT — computed once, survives reruns
if "plan" not in st.session_state:
    st.session_state.plan = None

if st.button("Generate Plan"):
    with st.spinner("Planning..."):
        st.session_state.plan = generate_plan(ctx)

if st.session_state.plan:
    render_plan(st.session_state.plan)
```

Rules:
- Anything expensive goes behind `if st.button(...)` and writes its result into `st.session_state`
- Anything read from disk gets `@st.cache_data`
- Never mutate `st.session_state` inside a loop that also renders widgets
- Widget keys must be unique — use `key=f"slider_{member_id}"` when rendering per-member controls

Initialize everything once at the top:

```python
DEFAULTS = {
    "profiles": load_seed(), "trip": None, "plan": None,
    "round": 0, "scores": {}, "feedback": {}, "history": [],
    "ledger": {}, "active_user": None, "locked_items": [],
    "learned": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
```

---

## 3. Architecture — the one rule

**One function handles all three planning modes.** Initial, iteration, and mid-trip disruption differ only by which arguments are populated.

```python
def generate_plan(
    trip,                    # dict: origin, dest, dates, budget
    members,                 # list of profile dicts (incl. learned weights)
    trip_constraints,        # non-negotiables for THIS trip
    feedback_history=None,   # [] on round 1
    ledger=None,             # fairness counters
    disruption=None,         # set => mid-trip mode, return ONE plan
    locked_items=None,       # stops that already happened
) -> dict:
```

Build three separate planners and you will not finish. One prompt template with conditional sections.

### Flow

```
seed profiles ─┐
trip basics ───┼─> generate_plan ─> plan ─> validate_hard()
constraints ───┘                             │ fail (max 2 retries)
                                             ↓ pass
                                        score_plan() ─> per-person 0–10
                                             ↓
                                    user feedback (score + tagged items)
                                             ↓
                              round >= 3 or delta < 0.3 ? ──no──> loop
                                             ↓ yes
                                        FINAL PLAN
                                             ↓
                                     [disruption during trip]
                                             ↓
                                     update_profiles() ─> weight diffs
```

---

## 4. Data shapes

Plain dicts. Document the schema at the top of `agent.py` — faster than dataclasses and AI codegen reads it fine.

```python
PROFILE = {
    "id": "priya", "name": "Priya", "avatar": "🧭",
    "hard": ["vegetarian", "cannot drive"],
    "soft": [{"pref": "historical sites", "weight": 0.9},
             {"pref": "in by 10pm", "weight": 0.6}],
    "can_drive": False,
    "learned": [{"pref": "historical sites", "old": 0.9, "new": 0.4,
                 "evidence": "skipped 2 of 3 museums on Trip 1"}],
}

PLAN = {
    "round": 1,
    "days": [{"date": "2026-08-14", "items": [{
        "id": "d1i3", "time": "14:30", "type": "meal",
        "title": "Lunch at Saffron Kitchen",
        "location": {"name": "...", "lat": 37.7, "lng": -122.4},
        "duration_min": 60, "driver": "sam",
        "notes": "full vegetarian menu — covers Priya",
        "fun_fact": "...", "serves_for": ["priya", "ana"],
    }]}],
    "reasoning": "why this plan is good",
    "tradeoffs": [{"description": "chose brewery over temple",
                   "favored": ["sam"], "disfavored": ["priya"]}],
}
```

### The weights, and why they matter

`weight` is 0–1: how much that preference matters *to that person*. Intake uses a 3-way picker — **Must have / Nice to have / Mild** → 0.9 / 0.6 / 0.3. Never make users type decimals.

Scoring, computed in **Python, not by the LLM**:

```python
def person_score(breakdown, soft):
    val = {"met": 1.0, "partial": 0.5, "unmet": 0.0}
    num = sum(s["weight"] * val[b["status"]] for s, b in zip(soft, breakdown))
    den = sum(s["weight"] for s in soft)
    return round(10 * num / den, 1)

group = mean(all_scores)
floor = min(all_scores)     # display both
```

**The weight is what learns.** Step 7 doesn't rewrite preference text — it moves the number. `0.9 → 0.4`. Your entire cross-trip learning story is one float changing, and it renders beautifully as a diff.

---

## 5. The three LLM calls

Use tool-use for guaranteed JSON:

```python
resp = client.messages.create(
    model="claude-sonnet-5", max_tokens=4000,
    tools=[{"name": "emit_plan", "description": "Return the itinerary",
            "input_schema": PLAN_SCHEMA}],
    tool_choice={"type": "tool", "name": "emit_plan"},
    messages=[{"role": "user", "content": prompt}],
)
plan = resp.content[0].input
```

### A. `generate_plan`

```
Plan a group trip. TRIP: {origin} → {dest}, {dates}, {n} people, ${budget}/person

MEMBERS:
{name, HARD constraints, SOFT prefs with weights, can_drive, LEARNED weight adjustments}

HARD CONSTRAINTS — violating any invalidates the plan: {list}

{if feedback_history:}
PREVIOUS ROUNDS:
Round {n} scored {x}/10. {name} rated {score}, disliked "{item}" because "{note}"
Change what was criticized. Keep what scored well.

{if ledger:}
FAIRNESS: {name} has been overruled {n} times. Favor them this round.

{if disruption:}
MID-TRIP CHANGE: {description}
LOCKED (already happened, cannot move): {locked_items}
Return ONE revised plan for the remainder. Do not restart the trip.

For every decision where you chose one member's preference over another's,
record it in `tradeoffs` with who was favored and disfavored.
```

### B. `score_plan` — separate call, never let the planner grade itself

```
For each member, evaluate each of their soft preferences against this plan.
Return met | partial | unmet plus a one-sentence reason.
Do NOT compute an overall number.
```

Python does the arithmetic. Say this out loud in the demo — it's a real credibility point.

### C. `update_profiles` — the hero

```
{name}'s profile before the trip: {profile}
What they actually did: {attendance, skips, extra stops}
How they rated it: {ratings + notes}

Where does stated preference diverge from revealed behavior?
Output 1–3 weight adjustments: {pref, old_weight, new_weight, evidence}
Only if evidence is clear. Empty list is valid.
```

---

## 6. Hard constraint validator — pure Python, no LLM

```python
def validate_hard(plan, members) -> list[str]:
    v = []
    for day in plan["days"]:
        for item in day["items"]:
            if item.get("driver"):
                d = by_id(members, item["driver"])
                if not d["can_drive"]:
                    v.append(f"{d['name']} assigned to drive but cannot drive")
            if item["type"] == "meal":
                for m in members:
                    if "vegetarian" in m["hard"] and "vegetarian" not in item["notes"].lower():
                        v.append(f"{item['title']} may not serve {m['name']}")
    return v
```

Violations → re-call `generate_plan` with `VIOLATIONS: {...}` appended. **Max 2 retries**, then display them rather than looping forever.

Demo this: deliberately trigger a violation and show the agent self-correcting.

---

## 7. UI — three tabs in `app.py`

```python
tab_setup, tab_plan, tab_trip = st.tabs(["Setup", "Plan", "Live Trip"])
```

Sidebar on every tab — the user switcher:

```python
st.sidebar.selectbox("Acting as", [p["name"] for p in profiles], key="active_user")
```

**Setup** — trip basics (`st.text_input`, `st.date_input`, `st.number_input`), then an expander per member showing their stored profile plus a box for this trip's non-negotiables. One `st.button("Generate Plan")`.

**Plan** — the main screen.
- `st.metric("Group", "8.4")` and `st.metric("Lowest", "6.8 (Sam)")` side by side
- Per-member `st.progress(score/10)` bars
- Itinerary as `st.container(border=True)` per item — time, title, driver, fun fact
- The map: `st.image(static_maps_url)`
- `st.expander("Why this plan")` → reasoning + tradeoffs
- Feedback for the active user: `st.slider(1, 10)`, `st.multiselect` of item titles they disliked, `st.text_input` for a note
- When all members have submitted → "Generate Round N+1", or if converged, "Lock this plan"

**Live Trip** — locked past items greyed with `st.caption`. Leader's "What changed?" text box + a "⛈️ Weather Alert" button. Replan shows the **impact preview** before committing:

> `Priya 8.2 → 6.1 — loses her only vegetarian dinner`

Then acknowledge buttons. No re-voting mid-trip; the leader has override authority.

Second sub-tab: **"What the agent learned"** → the weight diffs.

### Convergence

```python
converged = st.session_state.round >= 3 or abs(group - prev_group) < 0.3
```

Then show: *"Further rounds won't meaningfully improve this. Round 3 is the best compromise. Sam is lowest at 6.8 because his brewery stop conflicts with the Sunday return time."*

**Capped at 3.** Five rounds is unwatchable in a 3-minute demo.

### The map — five minutes

```python
def map_url(items):
    markers = "&".join(
        f"markers=color:red%7Clabel:{i+1}%7C{it['location']['lat']},{it['location']['lng']}"
        for i, it in enumerate(items))
    path = "path=color:0x0000ff80|weight:4|" + "|".join(
        f"{it['location']['lat']},{it['location']['lng']}" for it in items)
    return f"https://maps.googleapis.com/maps/api/staticmap?size=640x480&{markers}&{path}&key={KEY}"

st.image(map_url(all_items))
```

---

## 8. Hour by hour — you

| Time | Task |
|---|---|
| **0:00–0:15** | `pip install streamlit anthropic requests`. `app.py` with 3 empty tabs + sidebar user switcher. Confirm `streamlit run` works. |
| **0:15–0:40** | Schema comments at top of `agent.py`. `session_state` defaults. **2 throwaway profiles you write yourself** so you're not blocked waiting on your teammate. |
| **0:40–1:40** | `generate_plan` — real Anthropic call with tool-use. **Test from a plain Python script, print JSON. No Streamlit yet.** |
| **1:40–2:10** | `validate_hard` + retry loop. `score_plan` + the Python arithmetic. Still script-only. |
| **2:10–3:10** | Plan tab: metrics, progress bars, itinerary, feedback widgets, round loop, convergence |
| **3:10–3:40** | Static map + fun facts (Wikipedia geosearch → LLM one-liner, cached) |
| **3:40–4:20** | Live Trip tab: disruption input, replan, **impact preview**, acknowledge |
| **4:20–4:55** | `update_profiles` + weight-diff view. Wire trip #1 history so the diff is real. |
| **4:55–6:00** | Spinners on every LLM call, cache all demo responses, **rehearse twice** |

**Two rules that decide whether you finish:**

1. **Hours 0:40–2:10 you write zero UI.** Get the agent producing correct JSON in a script first. Debugging LLM output through Streamlit reruns is miserable.
2. **Lock the `generate_plan` signature at 0:40 and never change it.** Everything downstream depends on it.

---

## 9. Teammate — never touches your files

| Deadline | Task |
|---|---|
| **First 30 min** | Get the Google Maps API key with billing enabled. Most common hackathon failure. If it stalls past 45 min, abandon it and use a screenshot. |
| **By 1:30** | `seed/profiles.json` — 5 members with **deliberate conflicts**: one vegetarian, one who can't drive, one with a Tuesday 2pm call needing wifi, one budget-tight, one who wants nightlife. Conflict is what makes the agent look smart. |
| **By 2:30** | `seed/trip1_history.json` — a completed trip where **stated preference contradicts behavior**. Sam says budget-conscious, rated the $220 hotel 9/10 and the hostel 4/10. Priya says she loves museums, skipped 2 of 3. **This file is the learning demo.** The contradiction must be obvious at a glance. |
| **By 3:00** | `fixtures/route.json` — call Directions + Places once, save the raw response |
| **3:00–5:00** | Run the app every 20 min. Write bugs in a shared list. Don't fix, just report. |
| **By 5:00** | 3-minute demo script, timed out loud |
| **5:00–6:00** | Screen-record the working demo as backup |

---

## 10. Cut lines — in this order, no hesitation

1. Fairness ledger → mention in the pitch, don't build the UI
2. Fun facts → hard-code five
3. Weather API → a button with canned storm text
4. Map → a screenshot with `st.image`
5. Round 3 → cap at 2
6. Hard-constraint auto-retry → display violations, skip the fix loop

**Never cut:** plan → score → feedback → replan, the disruption replan, the weight diff. Those three are the entire pitch.

---

## 11. Demo script — 3 minutes

| Time | Beat |
|---|---|
| 0:00 | "Five friends, one road trip, irreconcilable preferences." Show the profiles and their conflicts. |
| 0:20 | Round 1 → **Group 7.1, lowest 4.2 (Priya).** "It tells us who's unhappy and why, before anyone argues." |
| 0:45 | Two members leave tagged feedback. Round 2 → **8.4, lowest 6.8.** Point at the tradeoffs. |
| 1:10 | Round 3 → **8.9.** Convergence fires. Lock it. |
| 1:30 | **Trip is live.** Storm alert → one-shot replan with impact preview: "costs Priya 2 points, here's the mitigation." |
| 2:10 | **Learning tab.** "Sam *said* budget-conscious. He *behaved* premium. His weight moved 0.8 → 0.3." |
| 2:35 | "Next trip round 1 starts at 8.8, not 7.1. The group gets easier to plan for every time they travel." |

Lead with conflict, land on learning. The score is supporting evidence, not the story.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Streamlit reruns firing repeat LLM calls | Everything expensive behind `st.button` + `session_state`. Section 2. |
| Maps billing fails | Cached fixture by hour 3, or a screenshot |
| Malformed JSON | Tool-use schema + one retry |
| 20s LLM latency on stage | Pre-generate every demo plan; a `DEMO_MODE` flag serves cached dicts instantly |
| Venue wifi dies | Backup screen recording, rehearse offline |
| Scope creep into auth/DB | You have neither. Keep it that way. |
| Three separate planners | One `generate_plan`. Re-read section 3. |

---

## First 15 minutes

```bash
mkdir travel-consensus && cd travel-consensus
python -m venv .venv && source .venv/bin/activate
pip install streamlit anthropic requests
export ANTHROPIC_API_KEY=sk-...
touch app.py agent.py && mkdir seed fixtures
streamlit run app.py
```

Then paste the `session_state` defaults and the three empty tabs, and confirm it loads in the browser before writing anything else.
