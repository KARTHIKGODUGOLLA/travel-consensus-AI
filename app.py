"""
app.py - Group Travel Consensus AI (Streamlit)

Run:  streamlit run app.py
Offline demo:  DEMO_MODE=1 streamlit run app.py

STREAMLIT RULE
--------------
This script re-runs top to bottom on EVERY widget interaction. Anything
expensive (an LLM call, a network fetch) must sit behind a button and write its
result into st.session_state. Never call the agent at module level.
"""

from __future__ import annotations

import datetime as dt
import os

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import agent as A

st.set_page_config(page_title="Group Travel Consensus AI", page_icon="🧭", layout="wide")

TYPE_ICON = {
    "drive": "🚐", "meal": "🍽️", "sight": "📍",
    "stay": "🛏️", "work": "💻", "free": "🫧",
}


# ==========================================================================
# state
# ==========================================================================

def init_state() -> None:
    if st.session_state.get("_ready"):
        return

    base = A.load_profiles()
    preset = A.load_trip_preset()
    start = dt.date.today() + dt.timedelta(days=7)

    defaults = {
        "base_profiles": base,
        "profiles": base,
        "using_learned": False,
        "learned_updates": [],
        "trip": {
            **preset,
            "start_date": start.isoformat(),
            "end_date": (start + dt.timedelta(days=preset.get("days", 3) - 1)).isoformat(),
        },
        "trip_constraints": {m["id"]: "" for m in base},
        "round": 0,
        "plan": None,
        "scores": {},
        "summary": {},
        "violations": [],
        "history": [],
        "ledger": {},
        "feedback": {},
        "converged": False,
        "converge_msg": "",
        "locked": False,
        "progress_idx": 0,
        "disruption_text": "",
        "preview": None,
        "acks": {},
        "facts": {},
        "error": "",
        "_ready": True,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_rounds() -> None:
    for k, v in {
        "round": 0, "plan": None, "scores": {}, "summary": {}, "violations": [],
        "history": [], "ledger": {}, "feedback": {}, "converged": False,
        "converge_msg": "", "locked": False, "progress_idx": 0,
        "preview": None, "acks": {}, "disruption_text": "",
    }.items():
        st.session_state[k] = v


init_state()
P = st.session_state


def members() -> list[dict]:
    return P["profiles"]


def member_by_id(mid: str) -> dict:
    return next((m for m in members() if m["id"] == mid), {})


# ==========================================================================
# agent wrappers - the ONLY places the agent is called
# ==========================================================================

def run_round(disruption: str | None = None, locked_items: list[dict] | None = None) -> dict | None:
    """Generate + validate + score. Returns a bundle, does not mutate state."""
    rnd = P["round"] + 1 if disruption is None else P["round"]
    try:
        plan, viol = A.generate_plan(
            trip=P["trip"],
            members=members(),
            trip_constraints=P["trip_constraints"],
            feedback_history=P["history"] or None,
            ledger=P["ledger"] or None,
            disruption=disruption,
            locked_items=locked_items,
            round_no=rnd,
        )
        scores = A.score_plan(plan, members())
        return {
            "plan": plan,
            "violations": viol,
            "scores": scores,
            "summary": A.summarise(scores),
            "round": rnd,
        }
    except A.AgentError as e:
        P["error"] = str(e)
    except Exception as e:  # noqa: BLE001
        P["error"] = f"{type(e).__name__}: {e}"
    return None


def commit_round(bundle: dict) -> None:
    P["plan"] = bundle["plan"]
    P["scores"] = bundle["scores"]
    P["summary"] = bundle["summary"]
    P["violations"] = bundle["violations"]
    P["round"] = bundle["round"]
    P["ledger"] = A.update_ledger(P["ledger"], bundle["plan"])
    P["feedback"] = {}

    prev = P["history"][-1]["group_score"] if P["history"] else None
    P["converged"], P["converge_msg"] = A.convergence(
        bundle["round"], bundle["summary"]["group"], prev
    )


# ==========================================================================
# rendering helpers
# ==========================================================================

def score_colour(v: float) -> str:
    return "#2E9E5B" if v >= 8 else "#C8901E" if v >= 6 else "#C2452D"


def render_scoreboard(scores: dict, summary: dict, key: str) -> None:
    c1, c2, c3 = st.columns([1, 1, 3])
    c1.metric("Group score", f"{summary.get('group', 0)} / 10")
    c2.metric(
        "Lowest member",
        f"{summary.get('floor', 0)} / 10",
        help="Optimising the average alone can leave one person miserable.",
    )
    c3.caption(f"Carrying the most compromise right now: **{summary.get('floor_name','-')}**")

    cols = st.columns(len(scores))
    for col, mid in zip(cols, scores):
        s = scores[mid]
        m = member_by_id(mid)
        with col:
            st.markdown(
                f"<div style='font-size:0.9rem'>{m.get('avatar','')} <b>{s['name']}</b> "
                f"<span style='color:{score_colour(s['score'])}'>{s['score']}</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(s["score"] / 10, 1.0))
            with st.expander("why", expanded=False):
                for b in s["breakdown"]:
                    icon = {"met": "🟢", "partial": "🟡", "unmet": "🔴"}.get(b["status"], "⚪")
                    st.markdown(
                        f"{icon} **{b['pref']}**<br>"
                        f"<span style='font-size:0.8rem;opacity:0.75'>{b['why']}</span>",
                        unsafe_allow_html=True,
                    )


def render_item(item: dict, greyed: bool = False, show_fact: bool = True) -> None:
    icon = TYPE_ICON.get(item.get("type"), "•")
    title = item.get("title", "")
    dur = item.get("duration_min") or 0
    cost = item.get("cost_per_person") or 0
    driver = item.get("driver") or ""
    dim = "opacity:0.42;" if greyed else ""

    with st.container(border=True):
        head = f"<div style='{dim}'><b>{item.get('time','')}</b> &nbsp; {icon} &nbsp; {title}"
        if greyed:
            head += " &nbsp;<span style='font-size:0.75rem'>· done</span>"
        head += "</div>"
        st.markdown(head, unsafe_allow_html=True)

        bits = []
        if dur:
            bits.append(f"{dur} min")
        if driver:
            bits.append(f"driver: {member_by_id(driver).get('name', driver)}")
        if cost:
            bits.append(f"${cost:.0f} pp")
        loc = item.get("location", {}).get("name")
        if loc:
            bits.append(loc)
        if bits:
            st.markdown(
                f"<div style='{dim}font-size:0.8rem;opacity:0.7'>{' · '.join(bits)}</div>",
                unsafe_allow_html=True,
            )

        if item.get("notes"):
            st.markdown(
                f"<div style='{dim}font-size:0.82rem'>{item['notes']}</div>",
                unsafe_allow_html=True,
            )

        served = [member_by_id(x).get("avatar", "") for x in (item.get("serves_for") or [])]
        if served:
            st.markdown(
                f"<div style='{dim}font-size:0.8rem'>serves {' '.join(served)}</div>",
                unsafe_allow_html=True,
            )

        if show_fact and not greyed:
            fact = P["facts"].get(item.get("id"))
            if fact:
                st.markdown(
                    f"<div style='font-size:0.78rem;opacity:0.7'>💡 {fact}</div>",
                    unsafe_allow_html=True,
                )


def render_itinerary(plan: dict, upto: int | None = None) -> None:
    n = 0
    for day in plan.get("days", []):
        st.markdown(f"##### {day.get('label') or day.get('date')}")
        for item in day.get("items", []):
            render_item(item, greyed=(upto is not None and n < upto))
            n += 1


def render_map(plan: dict) -> None:
    url = A.static_map_url(plan)
    if url:
        st.image(url, use_container_width=True)
    else:
        st.info(
            "Map hidden - set `GOOGLE_MAPS_API_KEY` in `.env` to render the route.",
            icon="🗺️",
        )
        pts = [
            {"lat": i["location"]["lat"], "lon": i["location"]["lng"]}
            for _, i in A.iter_items(plan)
            if i.get("location", {}).get("lat")
        ]
        if pts:
            st.map(pts, size=200)


def all_items(plan: dict) -> list[dict]:
    return [i for _, i in A.iter_items(plan)]


# ==========================================================================
# sidebar
# ==========================================================================

with st.sidebar:
    st.markdown("### 🧭 Consensus AI")

    names = [f"{m['avatar']} {m['name']}" for m in members()]
    picked = st.selectbox("Acting as", names, key="active_user_label")
    active = members()[names.index(picked)]

    st.divider()
    if A.DEMO_MODE:
        st.warning("DEMO_MODE - canned plans, no API calls", icon="🎬")
    elif not A.api_key_present():
        st.error("No ANTHROPIC_API_KEY - falling back to offline fixtures", icon="🔑")
    else:
        st.success(f"Live · {A.MODEL}", icon="🤖")

    if P["using_learned"]:
        st.info("Profiles include Trip 1 learning", icon="🧠")

    st.divider()
    st.caption(f"Round {P['round']} of {A.MAX_ROUNDS}")
    if P["ledger"]:
        st.caption("**Fairness ledger**")
        for mid, v in sorted(P["ledger"].items(), key=lambda kv: kv[1]):
            if v:
                st.caption(f"{member_by_id(mid).get('name', mid)}: {v:+d}")

    st.divider()
    if st.button("Reset rounds", use_container_width=True):
        reset_rounds()
        st.rerun()

if P["error"]:
    st.error(P["error"], icon="⚠️")
    if st.button("dismiss"):
        P["error"] = ""
        st.rerun()


# ==========================================================================
# tabs
# ==========================================================================

tab_setup, tab_plan, tab_trip, tab_learn = st.tabs(
    ["1 · Setup", "2 · Plan", "3 · Live trip", "4 · What it learned"]
)


# --------------------------------------------------------------------------
# TAB 1 - setup
# --------------------------------------------------------------------------
with tab_setup:
    st.subheader("Trip basics")
    t = P["trip"]
    c1, c2, c3 = st.columns(3)
    t["origin"] = c1.text_input("From", t["origin"])
    t["destination"] = c2.text_input("To", t["destination"])
    t["budget_per_person"] = c3.number_input(
        "Budget per person ($)", 100, 5000, int(t["budget_per_person"]), 50
    )
    c4, c5 = st.columns(2)
    sd = c4.date_input("Start", dt.date.fromisoformat(t["start_date"]))
    ed = c5.date_input("End", dt.date.fromisoformat(t["end_date"]))
    t["start_date"], t["end_date"] = sd.isoformat(), ed.isoformat()
    t["days"] = max((ed - sd).days + 1, 1)
    t["route_hint"] = st.text_input("Route preference", t.get("route_hint", ""))

    st.divider()
    st.subheader("The group")
    st.caption(
        "Hard constraints are validated in code and never traded away. "
        "Soft preferences are weighted and scored."
    )

    for m in members():
        with st.expander(f"{m['avatar']} {m['name']}", expanded=False):
            st.markdown("**Hard — non-negotiable**")
            for h in m["hard"]:
                st.markdown(f"- {h}")
            st.markdown("**Soft — weighted**")
            for s in A._effective_soft(m):
                base = next((x["weight"] for x in m["soft"] if x["pref"] == s["pref"]), s["weight"])
                moved = abs(base - s["weight"]) > 0.01
                tag = f"  ~~{base:.1f}~~ → **{s['weight']:.1f}**" if moved else f"  **{s['weight']:.1f}**"
                st.markdown(f"- {s['pref']} —{tag}")
            P["trip_constraints"][m["id"]] = st.text_input(
                "Anything non-negotiable for THIS trip?",
                value=P["trip_constraints"].get(m["id"], ""),
                key=f"tc_{m['id']}",
                placeholder="e.g. I have to be at a wedding rehearsal Saturday 4pm",
            )

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        if st.button("Generate plan", type="primary", use_container_width=True):
            reset_rounds()
            with st.spinner("Planning around 5 people's constraints..."):
                b = run_round()
            if b:
                commit_round(b)
                st.rerun()
    with right:
        st.caption(
            "Round 1 uses profiles + this trip's non-negotiables. "
            "Nobody has given feedback yet."
        )


# --------------------------------------------------------------------------
# TAB 2 - plan and iteration
# --------------------------------------------------------------------------
with tab_plan:
    if not P["plan"]:
        st.info("No plan yet. Head to **Setup** and hit Generate plan.", icon="👈")
    else:
        plan = P["plan"]
        st.subheader(f"Round {P['round']} — {plan.get('summary','')}")

        if P["violations"]:
            st.error(
                "Hard-constraint violations the planner could not fix:\n\n"
                + "\n".join(f"- {v}" for v in P["violations"]),
                icon="🚫",
            )
        else:
            st.success(
                f"All {len(A._all_hard(members()))} hard constraints verified in code.",
                icon="✅",
            )

        render_scoreboard(P["scores"], P["summary"], key="plan")
        st.divider()

        left, right = st.columns([3, 2])
        with left:
            render_itinerary(plan)
        with right:
            render_map(plan)
            with st.expander("Why this plan", expanded=True):
                st.write(plan.get("reasoning", ""))
            if plan.get("tradeoffs"):
                with st.expander("Tradeoffs the agent made", expanded=True):
                    for tr in plan["tradeoffs"]:
                        fav = ", ".join(member_by_id(x).get("name", x) for x in tr.get("favored", []))
                        dis = ", ".join(member_by_id(x).get("name", x) for x in tr.get("disfavored", []))
                        st.markdown(
                            f"- {tr['description']}<br>"
                            f"<span style='font-size:0.78rem;opacity:0.75'>"
                            f"favoured {fav or '—'} · cost to {dis or '—'}</span>",
                            unsafe_allow_html=True,
                        )
            if st.button("Load local facts for these stops"):
                with st.spinner("Wikipedia geosearch..."):
                    for i in all_items(plan):
                        loc = i.get("location", {})
                        if loc.get("lat") and i["id"] not in P["facts"]:
                            f = A.fun_fact(loc["lat"], loc["lng"])
                            if f:
                                P["facts"][i["id"]] = f
                st.rerun()

        st.divider()

        if P["locked"]:
            st.success("Plan locked. Move to **Live trip**.", icon="🔒")

        elif P["converged"]:
            st.info(P["converge_msg"], icon="🎯")
            low = P["summary"].get("floor_name")
            st.caption(
                f"{low} is carrying the most compromise at "
                f"{P['summary'].get('floor')}/10. That is the honest cost of this plan."
            )
            if st.button("Lock this plan", type="primary"):
                P["locked"] = True
                st.rerun()

        else:
            st.markdown(f"##### Feedback — you are **{active['name']}**")
            st.caption(
                "The score alone tells the agent nothing. Tag the specific stops "
                "you did not like — that is what drives the next round."
            )
            items = all_items(plan)
            labels = {f"{i['time']} · {i['title']}": i["id"] for i in items}

            existing = P["feedback"].get(active["id"], {})
            # Widget keys MUST be scoped by round. The itinerary changes between
            # rounds, so a persisted multiselect value from round N would no
            # longer be a valid option in round N+1 and Streamlit would raise.
            wk = f"{P['round']}_{active['id']}"
            fc1, fc2 = st.columns([1, 2])
            sc_val = fc1.slider(
                "Your score", 1, 10, int(existing.get("score", 7)), key=f"fb_s_{wk}"
            )
            tags = fc2.multiselect(
                "What did not work?",
                list(labels),
                default=[k for k, v in labels.items() if v in existing.get("tag_ids", [])],
                key=f"fb_t_{wk}",
            )
            note = st.text_input(
                "Why? (one line)", value=existing.get("note", ""), key=f"fb_n_{wk}"
            )

            if st.button("Submit feedback", type="primary"):
                P["feedback"][active["id"]] = {
                    "name": active["name"],
                    "score": sc_val,
                    "tags": tags,
                    "tag_ids": [labels[t] for t in tags],
                    "note": note,
                }
                st.rerun()

            done = len(P["feedback"])
            st.progress(done / max(len(members()), 1))
            waiting = [m["name"] for m in members() if m["id"] not in P["feedback"]]
            st.caption(
                f"{done}/{len(members())} submitted."
                + (f" Waiting on: {', '.join(waiting)}" if waiting else " Everyone is in.")
            )

            if not waiting:
                if st.button(f"Generate round {P['round'] + 1}", type="primary"):
                    P["history"].append({
                        "round": P["round"],
                        "group_score": P["summary"]["group"],
                        "feedback": list(P["feedback"].values()),
                    })
                    with st.spinner("Revising around the complaints..."):
                        b = run_round()
                    if b:
                        commit_round(b)
                        st.rerun()

        if P["history"]:
            st.divider()
            st.caption("**Score by round**")
            pts = [h["group_score"] for h in P["history"]] + [P["summary"]["group"]]
            st.line_chart({"group score": pts}, height=160)


# --------------------------------------------------------------------------
# TAB 3 - live trip
# --------------------------------------------------------------------------
with tab_trip:
    if not P["locked"]:
        st.info("Lock a plan on the **Plan** tab first.", icon="🔒")
    else:
        plan = P["plan"]
        items = all_items(plan)

        st.subheader("Live trip")
        P["progress_idx"] = st.slider(
            "Where is the group right now?", 0, len(items),
            min(P["progress_idx"], len(items)),
            help="Items before this point are locked - the agent cannot move them.",
        )
        done = items[: P["progress_idx"]]

        st.divider()
        st.markdown("##### Something changed")
        dc1, dc2 = st.columns([3, 1])
        P["disruption_text"] = dc1.text_input(
            "What happened?",
            value=P["disruption_text"],
            placeholder="Highway 1 is closed south of Big Sur / we are running 2 hours late",
        )
        with dc2:
            st.write("")
            if st.button("⛈️ Check weather", use_container_width=True):
                nxt = items[P["progress_idx"]] if P["progress_idx"] < len(items) else items[-1]
                loc = nxt.get("location", {})
                day = plan["days"][0]["date"]
                w = A.weather_for(loc.get("lat", 0), loc.get("lng", 0), day)
                if w:
                    approx = " (nearest forecast day)" if w.get("approx") else ""
                    P["disruption_text"] = (
                        f"Weather at {loc.get('name')}: {w['desc']}, "
                        f"{w['rain_chance']}% chance of rain, max {w['temp_max']}°C"
                        f"{approx}. Outdoor stops are at risk - reroute to keep the "
                        f"group dry and safe."
                    )
                else:
                    P["disruption_text"] = (
                        "Storm warning on the coastal stretch. Outdoor stops unsafe."
                    )
                st.rerun()

        if st.button("Replan from here", type="primary", disabled=not P["disruption_text"]):
            with st.spinner("One-shot replan, no voting rounds..."):
                b = run_round(disruption=P["disruption_text"], locked_items=done)
            if b:
                P["preview"] = {**b, "base_scores": P["scores"]}
                P["acks"] = {}
                st.rerun()

        if P["preview"]:
            pv = P["preview"]
            st.divider()
            st.markdown("##### Impact preview")
            st.caption("Nothing is committed yet. This is what the change costs each person.")

            cols = st.columns(len(pv["scores"]))
            for col, mid in zip(cols, pv["scores"]):
                before = pv["base_scores"].get(mid, {}).get("score", 0)
                after = pv["scores"][mid]["score"]
                col.metric(
                    pv["scores"][mid]["name"], f"{after}",
                    f"{after - before:+.1f}", delta_color="normal",
                )

            hurt = sorted(
                (
                    (pv["scores"][mid]["name"], pv["base_scores"].get(mid, {}).get("score", 0),
                     pv["scores"][mid]["score"])
                    for mid in pv["scores"]
                ),
                key=lambda r: r[2] - r[1],
            )
            worst = hurt[0]
            if worst[2] < worst[1]:
                st.warning(
                    f"**{worst[0]}** takes the hit: {worst[1]} → {worst[2]}. "
                    f"{pv['plan'].get('reasoning','')}",
                    icon="⚠️",
                )

            if pv["violations"]:
                st.error("\n".join(f"- {v}" for v in pv["violations"]), icon="🚫")

            with st.expander("Revised itinerary", expanded=False):
                render_itinerary(pv["plan"], upto=P["progress_idx"])

            st.markdown("**Acknowledge** — the leader decides, but everyone should see it.")
            acols = st.columns(len(members()))
            for col, m in zip(acols, members()):
                with col:
                    if P["acks"].get(m["id"]):
                        st.success(f"{m['avatar']} ✓", icon="✅")
                    elif st.button(f"{m['avatar']} {m['name']}", key=f"ack_{m['id']}"):
                        P["acks"][m["id"]] = True
                        st.rerun()

            c1, c2 = st.columns(2)
            if c1.button("Apply this plan", type="primary"):
                P["plan"] = pv["plan"]
                P["scores"] = pv["scores"]
                P["summary"] = pv["summary"]
                P["violations"] = pv["violations"]
                P["preview"] = None
                P["disruption_text"] = ""
                st.rerun()
            if c2.button("Discard"):
                P["preview"] = None
                st.rerun()

        st.divider()
        render_scoreboard(P["scores"], P["summary"], key="trip")
        st.divider()
        render_itinerary(plan, upto=P["progress_idx"])


# --------------------------------------------------------------------------
# TAB 4 - learning
# --------------------------------------------------------------------------
with tab_learn:
    st.subheader("Post-trip: stated preference vs revealed behaviour")
    hist = A.load_trip1_history()
    st.caption(
        f"Source: **{hist['title']}, {hist['dates']}** — "
        f"{len(hist['events'])} logged behaviours, {len(hist['ratings'])} ratings."
    )

    if st.button("Run post-mortem", type="primary"):
        with st.spinner("Comparing what they said against what they did..."):
            P["learned_updates"] = A.update_profiles(P["base_profiles"], hist)
        st.rerun()

    if not P["learned_updates"]:
        st.info(
            "Run the post-mortem to see which preference weights the agent would "
            "change, and why.",
            icon="🧠",
        )
    else:
        by_member: dict[str, list[dict]] = {}
        for u in P["learned_updates"]:
            by_member.setdefault(u["member_id"], []).append(u)

        for m in P["base_profiles"]:
            ups = by_member.get(m["id"], [])
            with st.container(border=True):
                if not ups:
                    st.markdown(
                        f"{m['avatar']} **{m['name']}** — no changes. "
                        "<span style='opacity:0.7;font-size:0.85rem'>Behaviour matched "
                        "what they said they wanted.</span>",
                        unsafe_allow_html=True,
                    )
                    continue
                st.markdown(f"{m['avatar']} **{m['name']}**")
                for u in ups:
                    arrow = "▲" if u["new"] > u["old"] else "▼"
                    colour = "#2E9E5B" if u["new"] > u["old"] else "#C2452D"
                    st.markdown(
                        f"<div style='font-size:0.9rem'>{u['pref']}<br>"
                        f"<span style='color:{colour};font-weight:600'>"
                        f"{u['old']:.2f} {arrow} {u['new']:.2f}</span></div>"
                        f"<div style='font-size:0.8rem;opacity:0.75'>{u['evidence']}</div>",
                        unsafe_allow_html=True,
                    )

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            if not P["using_learned"]:
                if st.button("Apply to profiles", type="primary", use_container_width=True):
                    P["profiles"] = A.apply_updates(
                        P["base_profiles"], P["learned_updates"], hist["trip_id"]
                    )
                    P["using_learned"] = True
                    reset_rounds()
                    st.rerun()
            else:
                if st.button("Revert to stated profiles", use_container_width=True):
                    P["profiles"] = P["base_profiles"]
                    P["using_learned"] = False
                    reset_rounds()
                    st.rerun()
        with c2:
            st.caption(
                "Applying these resets the rounds so you can re-plan the same trip "
                "with the learned weights. Round 1 should land higher than it did "
                "before — that is the whole point."
            )
