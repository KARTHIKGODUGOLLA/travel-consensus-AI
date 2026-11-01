# Travel Consensus AI

A Python and Streamlit prototype for planning a group trip around competing
preferences. It shows itinerary tradeoffs, collects feedback, previews a
disruption replan, and demonstrates preference-weight updates after a trip.
It is intended for developers exploring collaborative travel-planning workflows.

## Run the offline demo

Python 3.11 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
DEMO_MODE=1 python -m streamlit run app.py --server.address 127.0.0.1
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`, set
`$env:DEMO_MODE = "1"`, then run `python -m streamlit run app.py`.

Click **Generate plan**. The offline demo serves authored itinerary fixtures.
It does not adapt those fixtures to arbitrary new destinations or dates.
External AI, weather, facts, and maps are disabled in this mode, including when
API credentials are already present in the environment. Usage reporting is
disabled in the included Streamlit configuration.

## Optional hosted API mode

```bash
cp .env.example .env
# Set DEMO_MODE=0 and your own ANTHROPIC_API_KEY in .env.
python -m streamlit run app.py --server.address 127.0.0.1
```

The live planner uses Anthropic's hosted API; requests can incur provider
charges. `TCAI_MODEL` selects the model ID. Check the provider's available
[model IDs](https://platform.claude.com/docs/en/models/overview) for your account.
The optional `GOOGLE_MAPS_API_KEY` enables static maps. Weather and local facts
use Open-Meteo and Wikipedia. No credentials are included in the repository.

The app passes supplied trip, profile, feedback, and history text to the hosted
provider in live mode. Use demo data for development and review. See
[DATA.md](DATA.md) before using live integrations.

## What works and what remains limited

- One planning function handles initial planning, feedback rounds, and replanning.
- Prototype Python checks inspect driver assignments and selected dietary and
  accessibility wording. They do not verify venue facts or every free-text rule.
- A preference score is computed in Python from weighted met/partial/unmet
  statuses. These are application scores, not model accuracy benchmarks.
- The feedback loop has a round limit and a score-change stopping rule. Neither
  establishes that an itinerary is optimal.
- The post-trip view changes application preference weights. It does not train
  or fine-tune model weights.
- State is held in Streamlit's session. There is no durable database, user
  authentication, booking, payment, or guaranteed itinerary availability.
- Offline updates and the next-trip improvement display are scripted fixtures;
  they are not evidence of measured model improvement.

See [ROADMAP.md](ROADMAP.md) for a proposed 26–34 hours of application work.

## Checks

```bash
DEMO_MODE=1 python -m unittest discover -s tests -v
python -m compileall -q app.py agent.py tests
```

The checks use committed demo fixtures and block external requests. They verify
application behavior, not the quality of a hosted model. Live API calls and
provider account access are not part of these checks.

## Files and history

- `app.py`: Streamlit UI and session state.
- `agent.py`: prompts, hosted API calls, fixture selection, domain checks, and scoring.
- `seed/` and `fixtures/`: authored demo profiles, trip history, itineraries, and scores.
- `BUILD_PLAN.md`: original hackathon design document; some details differ from the implementation.
- `tests/`: offline application regression checks.

The existing July 2026 commits are retained, including the README contribution
by `surbhisguha`. Authorship and ownership must not be inferred solely from the
account that hosts the repository.

## License status

The original repository has no root LICENSE file. A release license still needs
maintainer confirmation covering all code and demo fixtures, including other
contributors' work. This preparation does not claim the project is already
licensed under MIT or approved for a submission program.
