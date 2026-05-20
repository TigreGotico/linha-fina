"""Checks that OVOS template syntax is expanded at intent
registration time:

* ``(a|b)``    — alternatives
* ``[opt]``    — optional segments
* ``{slot}``   — placeholders preserved as templates
"""

from linha_fina.engine import IntentEngine


def _samples_for(engine: IntentEngine, label: str):
    """Return positive training samples held by the dynamic classifier."""
    return list(engine.clf.clfs[label].positives)


def test_alternatives_expanded_at_registration():
    engine = IntentEngine()
    engine.register_intent("greet", ["(hello|hi|hey) there"])

    samples = _samples_for(engine, "greet")
    for variant in ("hello there", "hi there", "hey there"):
        assert variant in samples, (variant, samples)


def test_optionals_expanded_at_registration():
    engine = IntentEngine()
    engine.register_intent("lights_on", ["turn [the] lights on"])

    samples = _samples_for(engine, "lights_on")
    assert "turn the lights on" in samples
    assert "turn lights on" in samples


def test_slot_placeholders_preserved_as_templates():
    engine = IntentEngine()
    engine.register_intent(
        "introduce",
        ["(my name is|call me) {name}"],
        entity_samples={"name": ["Miro", "Casimiro"]},
    )

    samples = _samples_for(engine, "introduce")
    # Both alternative variants survive with the slot placeholder intact.
    assert "my name is {name}" in samples
    assert "call me {name}" in samples

    # The template matcher receives both variants for slot extraction.
    templates = engine.t_matchers["introduce"].templates["name"]
    assert "my name is {name}" in templates
    assert "call me {name}" in templates


def test_combined_alternatives_optionals_and_slots():
    engine = IntentEngine()
    engine.register_intent(
        "play",
        ["(play|put on) [the song] {title}"],
        entity_samples={"title": ["yesterday"]},
    )

    samples = _samples_for(engine, "play")
    for variant in (
        "play the song {title}",
        "play {title}",
        "put on the song {title}",
        "put on {title}",
    ):
        assert variant in samples, (variant, samples)
