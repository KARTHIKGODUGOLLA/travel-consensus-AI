import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["DEMO_MODE"] = "1"

import agent as A
from streamlit.testing.v1 import AppTest


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.mode = patch.object(A, "DEMO_MODE", True)
        self.mode.start()
        self.addCleanup(self.mode.stop)
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("External request in offline test"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_external_helpers_stay_offline_even_with_a_map_key(self):
        plan = A._demo_plan(1)
        with patch.object(A, "GOOGLE_MAPS_KEY", "test-only-not-a-real-key"):
            self.assertIsNone(A.static_map_url(plan))
        self.assertIsNone(A.fun_fact(37.7, -122.4))
        self.assertIsNone(A.weather_for(37.7, -122.4, "2026-08-14"))
        self.assertIsNone(A._fetch_weather(37.7, -122.4, "2026-08-14"))

    def test_direct_hosted_calls_are_blocked(self):
        with self.assertRaisesRegex(A.AgentError, "disabled"):
            A._client()
        with self.assertRaisesRegex(A.AgentError, "disabled"):
            A._tool_call("test", "emit_test", {})

    def test_fixture_plans_pass_prototype_checks_and_broken_fixture_fails(self):
        members = A.load_profiles()
        fixtures = A._read_json(A.FIXTURES / "demo_plans.json")
        for key in ("round1", "round2", "round3"):
            with self.subTest(key=key):
                self.assertEqual(A.validate_hard(fixtures[key], members), [])
        self.assertTrue(A.validate_hard(fixtures["broken"], members))

    def test_generated_demo_plan_is_an_independent_copy(self):
        members = A.load_profiles()
        first, _ = A.generate_plan(A.load_trip_preset(), members, round_no=1)
        first["summary"] = "changed locally"
        second, _ = A.generate_plan(A.load_trip_preset(), members, round_no=1)
        self.assertNotEqual(second["summary"], "changed locally")

    def test_preference_update_does_not_mutate_input_profiles(self):
        members = A.load_profiles()
        before = copy.deepcopy(members)
        updates = A.update_profiles(members, A.load_trip1_history())
        changed = A.apply_updates(members, updates)
        self.assertEqual(members, before)
        self.assertIsNot(changed, members)
        self.assertTrue(any(m.get("learned") for m in changed))

    def test_preference_score_is_weighted_application_arithmetic(self):
        soft = [{"pref": "museums", "weight": 0.6}, {"pref": "parks", "weight": 0.4}]
        values = [{"pref": "museums", "status": "met"}, {"pref": "parks", "status": "unmet"}]
        self.assertEqual(A.person_score(values, soft), 6.0)
        self.assertEqual(A.person_score([], []), 0.0)

    def test_round_cap_is_enforced(self):
        self.assertTrue(A.convergence(A.MAX_ROUNDS, 1.0, 9.0)[0])
        self.assertFalse(A.convergence(1, 6.0, None)[0])

    def test_dashboard_starts_and_generates_a_demo_plan_without_requests(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with patch.dict(os.environ, {"DEMO_MODE": "1"}):
            app = AppTest.from_file(str(app_path), default_timeout=20).run()
            self.assertFalse(app.exception)
            generate = next(button for button in app.button if button.label == "Generate plan")
            generate.click().run()
            self.assertFalse(app.exception)
            self.assertIsNotNone(app.session_state["plan"])
            self.assertTrue(any("External maps are disabled" in entry.value for entry in app.info))


if __name__ == "__main__":
    unittest.main()
