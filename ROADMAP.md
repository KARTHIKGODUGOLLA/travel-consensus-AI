# Proposed application roadmap

These are candidate development items beyond submission preparation. The
maintainer should confirm which work they independently intend to pursue.
Estimates are not work already completed or authorization for paid hours.

| Work | Acceptance criteria | Estimate |
| --- | --- | --- |
| Durable demo sessions | Save and resume synthetic trip state with schema versions; add reset/export behavior and recovery tests. | 5–6 h |
| Structured response handling | Validate hosted tool-output shapes and references; provide actionable errors and bounded retries using mocked responses. | 4–6 h |
| Editable demo scenarios | Add clear editors for fictional members, constraints, and itinerary items; handle conflicts and empty inputs. | 4–5 h |
| Honest offline adaptation | Make the offline path respond to changed demo dates and selected constraints rather than serving unrelated fixed fixtures; label unsupported requests. | 5–6 h |
| Accessible review and exports | Provide keyboard-friendly feedback, plain tables for chart information, and itinerary/decision exports. | 4–5 h |
| Integration resilience | Add session-recovery and cancellation tests, request timeout handling, redacted diagnostics, and a fresh-install walkthrough. | 4–6 h |
| **Total proposed scope** | **If all six items are selected by the maintainer.** | **26–34 h** |

This roadmap concerns application behavior and synthetic scenarios. It does
not include model training, fine-tuning, model benchmarking, or collecting real
traveller profiles for development or evaluation.
