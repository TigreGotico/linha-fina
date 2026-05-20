"""Domain-aware OPM pipeline wrapping :class:`DomainIntentEngine`.

This pipeline routes adds/removes by the ``skill_id`` extracted from the
intent label (``"skill_id:intent_name"``). Each ``skill_id`` becomes a
*domain* with its own per-intent SVM engine; at inference time every
domain engine produces a candidate and the global argmax wins.

linha-fina's per-intent SVM still needs at least 3 intents *globally*
before training succeeds (its own constraint, unchanged from the flat
pipeline). There is no per-domain minimum any more — without a
top-level router, a domain with a single intent is still perfectly
matchable as long as the corpus as a whole has enough intents to train.
"""

from typing import Dict, List, Optional, Union

from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG

from linha_fina.domain_engine import DomainIntentEngine
from linha_fina.opm import LinhaFinaPipeline, LinhaFinaIntent


def _split_intent_label(label: str):
    """Split ``skill_id:intent_name`` into ``(skill_id, intent_name)``."""
    if ":" in label:
        skill_id, intent_name = label.split(":", 1)
        return skill_id, intent_name
    return label, label


class DomainLinhaFinaPipeline(LinhaFinaPipeline):
    """Domain-partitioned LinhaFina pipeline.

    Intents are partitioned by ``skill_id`` (the prefix before the ``:``
    in the intent label) into per-skill :class:`IntentEngine` instances.
    Inference runs every per-skill engine in parallel and returns the
    global ``argmax`` by confidence.
    """

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        super().__init__(bus=bus, config=config or {})

    # ── hook overrides ────────────────────────────────────────────────────
    def _make_engine(self) -> DomainIntentEngine:
        return DomainIntentEngine()

    def _add_intent(self, lang: str, name: str, samples: List[str]) -> None:
        skill_id, _ = _split_intent_label(name)
        engine: DomainIntentEngine = self.containers[lang]
        engine.register_domain_intent(skill_id, name, samples)

    def _add_entity(self, lang: str, name: str, samples: List[str]) -> None:
        skill_id, _ = _split_intent_label(name)
        engine: DomainIntentEngine = self.containers[lang]
        engine.register_domain_entity(skill_id, name, samples)

    def _remove_intent(self, intent_name: str) -> None:
        skill_id, _ = _split_intent_label(intent_name)
        for lang, engine in self.containers.items():
            try:
                engine.remove_domain_intent(skill_id, intent_name)
            except Exception as e:
                LOG.debug(f"remove_domain_intent({skill_id},{intent_name}): {e}")

    def _remove_entity(self, name: str, lang: str) -> None:
        # entities are scoped per-domain inside DomainIntentEngine; the
        # underlying engines drop them with the domain on remove_domain.
        # individual removal is a no-op here.
        return

    def _remove_skill(self, skill_id: str) -> None:
        for lang, engine in self.containers.items():
            try:
                engine.remove_domain(skill_id)
            except Exception as e:
                LOG.debug(f"remove_domain({skill_id}): {e}")
        # update registered_intents bookkeeping
        keep = [i for i in self.registered_intents
                if _split_intent_label(i)[0] != skill_id]
        self.registered_intents[:] = keep

    # ── inference ─────────────────────────────────────────────────────────
    def calc_intent(self, utterances, lang: str = None,
                    message: Optional[Message] = None) -> Optional[LinhaFinaIntent]:
        if isinstance(utterances, str):
            utterances = [utterances]
        utterances = [u for u in utterances if len(u.split()) < self.max_words]
        if not utterances:
            return None
        lang = self._get_closest_lang(lang or self.lang)
        if lang is None:
            return None
        engine: DomainIntentEngine = self.containers[lang]
        if not engine.domains:
            return None
        best: Optional[LinhaFinaIntent] = None
        for utt in utterances:
            try:
                m = engine.calc_intent(utt)
            except Exception as e:
                LOG.error(f"DomainLinhaFina calc_intent error: {e}")
                continue
            if m is None or m.name is None:
                continue
            if best is None or m.conf > best.conf:
                best = LinhaFinaIntent(sent=utt, name=m.name,
                                       conf=m.conf, matches=m.slots)
        return best
