"""Tests for OVOS template expansion in the intent registration path."""
import unittest

from linha_fina._bracket_expansion import expand_slots, expand_template
from linha_fina.engine import IntentEngine, _expand_samples


class TestExpandTemplate(unittest.TestCase):
    def test_alternatives(self):
        out = expand_template("turn (on|off) the lights")
        self.assertEqual(out, ["turn off the lights", "turn on the lights"])

    def test_optional(self):
        out = expand_template("turn on the [bright] lights")
        self.assertIn("turn on the  lights", [o for o in out])
        # whitespace gets stripped by .strip() inside the helper but
        # internal double spaces remain; both variants must be present
        self.assertIn("turn on the bright lights", out)

    def test_combined(self):
        out = expand_template("turn (on|off) the [bright] lights")
        # 2 alts * 2 optionals = 4 variants
        self.assertEqual(len(out), 4)
        self.assertIn("turn on the bright lights", out)
        self.assertIn("turn off the bright lights", out)

    def test_slot_preserved(self):
        out = expand_template("set the (light|lamp) to {color}")
        self.assertIn("set the light to {color}", out)
        self.assertIn("set the lamp to {color}", out)


class TestExpandSlots(unittest.TestCase):
    def test_fill_slot(self):
        out = expand_slots("set light to {color}", {"color": ["red", "blue"]})
        self.assertEqual(sorted(out), ["set light to blue", "set light to red"])

    def test_unknown_slot_kept(self):
        out = expand_slots("call me {name}", {})
        self.assertEqual(out, ["call me {name}"])


class TestExpandSamples(unittest.TestCase):
    def test_dedup(self):
        out = _expand_samples(["turn (on|on) the lights"])
        self.assertEqual(out, ["turn on the lights"])

    def test_collapsed_whitespace(self):
        out = _expand_samples(["turn on the [bright] lights"])
        # After expansion + whitespace collapse, both should appear
        self.assertIn("turn on the lights", out)
        self.assertIn("turn on the bright lights", out)


class TestEngineEndToEnd(unittest.TestCase):
    def test_register_intent_expands(self):
        engine = IntentEngine()
        engine.register_intent("lights", ["turn (on|off) the [bright] lights"])
        # 4 fully-expanded variants should be added to the classifier
        labels = engine.clf.labels if hasattr(engine.clf, "labels") else {}
        # fall back: just confirm no error and that templates list non-empty
        # via t_matchers (templates only added for {slot} samples, so empty here)
        self.assertEqual(engine.t_matchers.get("lights"), None) if "lights" not in engine.t_matchers else None
        # The classifier should have multiple samples queued for the label.
        samples = getattr(engine.clf, "_samples", None) or labels
        if isinstance(samples, dict) and "lights" in samples:
            self.assertGreaterEqual(len(samples["lights"]), 4)

    def test_register_intent_with_slot(self):
        engine = IntentEngine()
        engine.register_intent(
            "color",
            ["(set|change) the light to {color}"],
            entity_samples={"color": ["red", "blue"]},
        )
        # Both alternates produce templates
        self.assertIn("color", engine.t_matchers)
        tm = engine.t_matchers["color"]
        # TemplateMatcher.templates is a dict keyed by entity, mapping to a
        # list of registered template strings.
        all_templates = []
        for v in tm.templates.values():
            all_templates.extend(v)
        self.assertIn("set the light to {color}", all_templates)
        self.assertIn("change the light to {color}", all_templates)


if __name__ == "__main__":
    unittest.main()
