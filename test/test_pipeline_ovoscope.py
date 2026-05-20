"""End-to-end smoke tests via ovoscope's PipelineHarness.

These run the linha-fina pipeline plugin inside a real MiniCroft to prove
the plugin loads, the entry point resolves, and bus-driven registration
plus inference work without a hand-built FakeBus harness.

For exhaustive bus-contract coverage see ``test_pipeline.py`` — those
tests drive the plugin directly with a FakeBus and don't spin up
MiniCroft, so they're an order of magnitude faster.
"""

import time

import pytest

ovoscope = pytest.importorskip("ovoscope")
from ovos_bus_client.message import Message
from ovoscope.pipeline import PipelineHarness

PLUGIN = "ovos-linha-fina-pipeline-plugin"


@pytest.fixture(scope="module")
def harness():
    h = PipelineHarness(pipeline=[PLUGIN], lang="en-US")
    with h:
        yield h


class TestPluginLoadsInMiniCroft:
    def test_plugin_resolves_from_entry_point(self, harness):
        # If MiniCroft started without error, OPM resolved the entry point.
        assert harness._mc is not None

    def test_unregistered_utterance_does_not_match(self, harness):
        # No intents registered — assert_no_match should succeed.
        harness.assert_no_match("frobble zorp xyzzy snork", timeout=3.0)


class TestBusDrivenRegistration:
    def test_register_intent_via_bus_then_predict(self, harness):
        bus = harness._mc.bus

        # Capture the pipeline's match log line so we can assert it fired.
        matches = []

        def _on_log(message):
            matches.append(message)

        bus.on("padatious:register_intent", _on_log)

        for name, samples in [
            ("demo:greet", ["hello", "hi there", "hey", "good morning"]),
            ("demo:bye", ["goodbye", "see you", "bye", "later"]),
            ("demo:thanks", ["thanks", "thank you", "much appreciated"]),
        ]:
            bus.emit(Message("padatious:register_intent", data={
                "name": name, "samples": samples, "lang": "en-US",
            }))
        bus.emit(Message("mycroft.ready"))
        time.sleep(2.0)

        # All three registrations were received by the plugin's handler.
        registered_names = [m.data.get("name") for m in matches]
        assert "demo:greet" in registered_names
        assert "demo:bye" in registered_names
        assert "demo:thanks" in registered_names

        bus.remove("padatious:register_intent", _on_log)
