# Data and integrations

The build plan describes authored seed profiles with deliberate conflicts and
an authored prior-trip history. The maintainer must confirm that every record
in `seed/` and `fixtures/` is fictional or otherwise appropriate for public
release before claiming that the repository contains no real personal data.

## Offline demo

With `DEMO_MODE=1`, application AI calls, Wikipedia facts, weather requests,
and external maps are disabled. Plans, preference assessments, and updates
come from local fixtures or deterministic application code. Streamlit usage
reporting is disabled in the supplied configuration. State is in memory for
the current Streamlit session; the app does not write profiles back to disk.

The fixed fixtures retain their authored destinations and dates. The display
is a scenario demonstration, not a booking service or evidence that a model
improved after a real trip.

## Optional live integrations

- Anthropic receives trip parameters, profiles, constraints, feedback, and
  prior-trip text included in the application's prompts.
- Wikipedia receives coordinates and article requests when facts are requested.
- Open-Meteo receives coordinates and requested dates for forecasts.
- Google Maps receives itinerary coordinates when a configured static map is
  displayed. The no-key live fallback also displays a browser map.

Use only fictional demo records while preparing or reviewing this project.
The submission program's restrictions on other people's personal or regulated
data remain relevant; this document does not certify production privacy.

Keys belong in local environment variables or an ignored `.env` file. The MIT
license, if later selected, would cover only material the maintainer has the
right to license; external services retain their own terms.
