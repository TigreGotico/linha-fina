"""End-to-end pipeline tests via ovoscope's PipelineHarness.

Loads the linha-fina pipeline plugin inside a real MiniCroft, drives it
with bus events, and asserts state on the **live plugin instance** —
``mc.intents.pipeline_plugins['ovos-linha-fina-pipeline-plugin']`` —
to confirm the plugin actually processed the events, not just that some
listener received them.

Exhaustive bus-contract coverage stays in ``test_pipeline.py`` via
direct FakeBus instantiation; these tests verify the OPM entry-point
integration and real-MiniCroft bus routing.
"""

import time

import pytest

ovoscope = pytest.importorskip("ovoscope")
from ovos_bus_client.message import Message
from ovoscope.pipeline import PipelineHarness

from linha_fina.opm import LinhaFinaPipeline

PLUGIN = "ovos-linha-fina-pipeline-plugin"


@pytest.fixture(scope="module")
def harness():
    h = PipelineHarness(pipeline=[PLUGIN], lang="en-US")
    with h:
        yield h


@pytest.fixture
def pipeline(harness):
    """The live LinhaFinaPipeline instance loaded inside MiniCroft."""
    plugin = harness._mc.intents.pipeline_plugins[PLUGIN]
    assert isinstance(plugin, LinhaFinaPipeline)
    # Reset state between tests so module-scoped harness can be reused
    plugin.registered_intents.clear()
    plugin.registered_entities.clear()
    for lang in plugin.containers:
        # Wipe each per-language engine without rebuilding the container
        plugin.containers[lang].clf.clfs.clear()
        plugin.containers[lang].t_matchers.clear()
        plugin.containers[lang].k_matchers.clear()
    return plugin


class TestPluginIntegration:
    def test_plugin_loaded_via_entry_point(self, harness):
        # If MiniCroft started and the entry point resolved, the plugin
        # registry contains a live LinhaFinaPipeline instance.
        plugins = harness._mc.intents.pipeline_plugins
        assert PLUGIN in plugins
        assert isinstance(plugins[PLUGIN], LinhaFinaPipeline)

    def test_plugin_uses_minicrofts_bus(self, harness, pipeline):
        # The plugin should be bound to the harness's MiniCroft bus, not
        # a private one.
        assert pipeline.bus is harness._mc.bus


class TestBusDrivenRegistration:
    def test_register_intent_lands_in_plugin_container(self, harness, pipeline):
        harness._mc.bus.emit(Message("padatious:register_intent", data={
            "name": "demo:greet",
            "samples": ["hello", "hi there", "hey"],
            "lang": "en-US",
        }))
        # FakeBus emit is synchronous, but give a tiny window for any
        # threaded handlers in the path.
        time.sleep(0.1)

        # Assert state on the live plugin, not on a side-channel listener.
        assert "demo:greet" in pipeline.registered_intents
        container = pipeline.containers[pipeline.lang]
        assert "demo:greet" in container.clf.clfs
        positives = container.clf.clfs["demo:greet"].positives
        assert "hello" in positives
        assert "hi there" in positives

    def test_detach_intent_via_bus_removes_from_plugin(self, harness, pipeline):
        bus = harness._mc.bus
        for name, samples in [
            ("demo:greet", ["hello", "hi", "hey"]),
            ("demo:bye", ["goodbye", "bye", "see you"]),
            ("demo:thanks", ["thanks", "thank you", "ty"]),
        ]:
            bus.emit(Message("padatious:register_intent", data={
                "name": name, "samples": samples, "lang": "en-US",
            }))
        time.sleep(0.1)
        assert "demo:greet" in pipeline.registered_intents

        bus.emit(Message("detach_intent", data={"intent_name": "demo:greet"}))
        time.sleep(0.1)

        assert "demo:greet" not in pipeline.registered_intents
        assert "demo:greet" not in pipeline.containers[pipeline.lang].clf.clfs
        # Sibling intents untouched
        assert "demo:bye" in pipeline.registered_intents
        assert "demo:thanks" in pipeline.registered_intents

    def test_detach_skill_via_bus_removes_all_skill_intents(self, harness, pipeline):
        bus = harness._mc.bus
        for name, samples in [
            ("media:play", ["play music", "start music", "begin playback"]),
            ("media:stop", ["stop music", "halt", "cease"]),
            ("weather:forecast", ["weather", "forecast", "temperature"]),
        ]:
            bus.emit(Message("padatious:register_intent", data={
                "name": name, "samples": samples, "lang": "en-US",
            }))
        time.sleep(0.1)

        bus.emit(Message("detach_skill", data={"skill_id": "media"}))
        time.sleep(0.1)

        # All media intents gone, weather untouched
        assert "media:play" not in pipeline.registered_intents
        assert "media:stop" not in pipeline.registered_intents
        assert "weather:forecast" in pipeline.registered_intents
        container = pipeline.containers[pipeline.lang]
        assert "media:play" not in container.clf.clfs
        assert "media:stop" not in container.clf.clfs
        assert "weather:forecast" in container.clf.clfs

    def test_unregistered_utterance_does_not_match(self, harness):
        # PipelineHarness.assert_no_match is a meaningful end-to-end check:
        # an empty plugin should not fabricate matches on gibberish.
        harness.assert_no_match("frobble zorp xyzzy snork", timeout=3.0)
