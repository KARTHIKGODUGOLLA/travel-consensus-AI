# travel-consensus-AI-Agent

An agent that plans a group trip around everyone's constraints at once, shows
who is compromising and by how much, replans in one shot when reality
interferes, and gets measurably better at each traveller after every trip.

## Quickstart

```bash
git clone https://github.com/KARTHIKGODUGOLLA/travel-consensus-AI.git
cd travel-consensus-AI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add your ANTHROPIC_API_KEY
streamlit run app.py
```

Offline, no API calls, no key needed:

```bash
DEMO_MODE=1 streamlit run app.py
```

`DEMO_MODE` serves hand-built fixtures instantly. Use it to rehearse, and as
the safety net if the venue wifi dies.

## What it does

1. **Profiles already exist.** Each person has *hard* constraints (vegetarian,
   cannot drive, wheelchair access) and *soft* preferences with weights 0–1.
2. **Plan.** One LLM call produces a full itinerary plus an honest list of the
   tradeoffs it made and who lost each one.
3. **Validate.** A pure-Python checker rejects any plan that breaks a hard
   constraint and sends it back with the specific violations. Max 2 retries.
4. **Score.** A second LLM call judges each preference met / partial / unmet.
   **Python does the arithmetic** — the model never invents the number.
5. **Iterate.** Members rate the plan *and tag the specific stops they disliked*.
   Capped at 3 rounds, or earlier if the score stops moving.
6. **Live trip.** A disruption triggers a one-shot replan with an **impact
   preview** — what it costs each person — before anything is committed.
7. **Learn.** After the trip, the agent compares what people *said* they wanted
   against what they *did*, and moves the weights.

## Architecture

There is exactly **one** planning function:

```python
generate_plan(trip, members, trip_constraints, feedback_history,
              ledger, disruption, locked_items, round_no)
```

Round 1, iteration rounds, and mid-trip replanning are the same call with
different arguments populated. Do not add a second planner.

```
agent.py    all logic: 3 LLM calls, validator, scorer, ledger, external APIs
app.py      the entire UI, 4 tabs
seed/       profiles, trip preset, Trip 1 history
fixtures/   canned plans + scores for DEMO_MODE
```

State lives in `st.session_state`. There is no database.

### Scoring

```
person = 10 × Σ(weight × status) / Σ(weight)     met=1.0  partial=0.5  unmet=0.0
group  = mean(person)
floor  = min(person)          <- shown alongside the average, deliberately
```

The floor matters. A group average of 8.4 can hide someone at 3.

### The weights are what learn

Post-trip, the agent does not rewrite preference text — it moves the number.
`historical sites: 0.9 → 0.4`, evidenced by "left the museum after 20 minutes
and rated it 4/10". That single float is the entire cross-trip learning story.

## Demo path

| Beat | Action |
|---|---|
| The problem | **Setup** — expand the five profiles. Vegetarian, gluten-free, wheelchair, two non-drivers, a Tuesday client call. |
| Round 1 | Generate. **5.7 group, Mei at 2.5** — the plan is unusable for her. Sam is at 10. |
| Feedback | Switch users in the sidebar, tag specific stops, submit. |
| Round 2 | **7.8. Sam drops to 5.5** — he is now carrying the compromise. Show the fairness ledger. |
| Round 3 | **8.5.** Convergence fires: further rounds only move cost between people. Lock it. |
| Live trip | Set the progress slider, hit **Check weather**, Replan. **Ana 9.1 → 4.8** in the impact preview. |
| Learning | **Run post-mortem.** Sam said budget-conscious, rated the $210 hotel 9/10. Ana gets **no changes** — she was consistent. |
| Payoff | Apply to profiles, replan. **Round 1 opens at 8.2 instead of 5.7.** |

Ana having no updates is the point: the agent does not invent changes it cannot
evidence.

## Verification

```bash
DEMO_MODE=1 python3 -c "
import agent as A
m = A.load_profiles()
p = A._read_json(A.FIXTURES/'demo_plans.json')
print('violations found in broken plan:', len(A.validate_hard(p['broken'], m)))
print('violations in real plans:', [len(A.validate_hard(p[k], m)) for k in ['round1','round2','round3']])
"
```

Expected: `5` and `[0, 0, 0]`.

## Notes and limits

- The Google Maps key is optional. Without it the app falls back to Streamlit's
  built-in point map, which is uglier but works.
- Open-Meteo forecasts ~15 days out. Beyond that `weather_for` clamps to the
  furthest available day and flags the result `approx`.
- Fun facts come from Wikipedia geosearch (keyless) — nearest article to a
  coordinate, first sentence.
- In `DEMO_MODE` the plans are canned, so they cannot genuinely adapt to learned
  weights. The fixture skips the naive first draft instead, which produces the
  same round-1 uplift you get live. With a real key the planner reads the
  learned weights directly.
