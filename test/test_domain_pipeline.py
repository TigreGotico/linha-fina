"""E2E tests for the DomainLinhaFinaPipeline."""

import pytest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from linha_fina.domain_opm import DomainLinhaFinaPipeline, _split_intent_label


def _make_register_msg(name: str, samples, lang="en-US"):
    return Message("padatious:register_intent",
                   {"name": name, "samples": samples, "lang": lang})


@pytest.fixture
def pipe():
    p = DomainLinhaFinaPipeline(bus=FakeBus(), config={})
    yield p
    p.shutdown()


def test_split_intent_label():
    assert _split_intent_label("media.skill:play") == ("media.skill", "play")
    assert _split_intent_label("noskill") == ("noskill", "noskill")


def test_parallel_argmax_routes_and_matches(pipe):
    # linha-fina's per-intent SVM needs samples from other intents in the
    # same engine to build negatives, but with parallel-argmax routing
    # there is no per-domain *router* minimum any more.
    pipe.register_intent(_make_register_msg(
        "media.skill:play",
        ["play music", "put on a song", "play africa", "i want to hear something"]))
    pipe.register_intent(_make_register_msg(
        "media.skill:stop",
        ["stop", "stop the music", "halt playback", "silence please"]))
    pipe.register_intent(_make_register_msg(
        "media.skill:next",
        ["next track", "skip song", "next please", "skip this one"]))
    pipe.register_intent(_make_register_msg(
        "home.skill:lights_on",
        ["turn on the lights", "lights on", "switch on the lamp", "illuminate the room"]))
    pipe.register_intent(_make_register_msg(
        "home.skill:lights_off",
        ["turn off the lights", "lights off", "switch off the lamp", "darken the room"]))
    pipe.register_intent(_make_register_msg(
        "home.skill:thermostat",
        ["set temperature to twenty", "make it warmer", "make it cooler", "thermostat to high"]))

    match = pipe.calc_intent(["play music"], lang="en-US")
    assert match is not None
    skill_id, _ = _split_intent_label(match.name)
    assert skill_id == "media.skill"

    match2 = pipe.calc_intent(["turn on the lights"], lang="en-US")
    assert match2 is not None
    skill_id2, _ = _split_intent_label(match2.name)
    assert skill_id2 == "home.skill"


def test_detach_skill_removes_domain(pipe):
    for name, samples in [
        ("media.skill:play", ["play music", "play a song", "put on tunes", "play africa"]),
        ("media.skill:stop", ["stop", "halt", "pause it", "silence"]),
        ("media.skill:next", ["next", "skip", "next track", "skip this"]),
    ]:
        pipe.register_intent(_make_register_msg(name, samples))

    engine = pipe.containers["en-US"]
    assert "media.skill" in engine.domains

    pipe.handle_detach_skill(Message("detach_skill", {"skill_id": "media.skill"}))
    assert "media.skill" not in engine.domains
    assert not any(i.startswith("media.skill:") for i in pipe.registered_intents)


def test_ovoscope_e2e():
    """Optional ovoscope round-trip via the OVOS pipeline harness."""
    ovoscope = pytest.importorskip("ovoscope")
    # We only assert the plugin is discoverable via OPM and constructible.
    pipe = DomainLinhaFinaPipeline(bus=FakeBus(), config={})
    assert pipe.containers
    pipe.shutdown()
