"""OVOS-style template expansion helpers.

Supports ``(a|b)`` alternatives, ``[opt]`` optionals, and ``{slot}``
placeholders. Inlined locally to avoid a runtime dependency on
``ovos-utils``.
"""
import itertools
import re
from typing import Dict, List


def expand_template(template: str) -> List[str]:
    def expand_optional(text):
        return re.sub(r"\[([^\[\]]+)\]", lambda m: f"({m.group(1)}|)", text)

    def expand_alternatives(text):
        parts = []
        for segment in re.split(r"(\([^\(\)]+\))", text):
            if segment.startswith("(") and segment.endswith(")"):
                parts.append(segment[1:-1].split("|"))
            else:
                parts.append([segment])
        return itertools.product(*parts)

    def fully_expand(texts):
        result = set(texts)
        while True:
            expanded = set()
            for text in result:
                for option in expand_alternatives(text):
                    expanded.add("".join(option).strip())
            if expanded == result:
                break
            result = expanded
        return sorted(result)

    return fully_expand([expand_optional(template)])


def expand_slots(template: str, slots: Dict[str, List[str]]) -> List[str]:
    base = expand_template(template)
    out = []
    for sentence in base:
        matches = re.findall(r"\{([^\{\}]+)\}", sentence)
        if not matches:
            out.append(sentence); continue
        slot_options = [slots.get(m, [f"{{{m}}}"]) for m in matches]
        for combo in itertools.product(*slot_options):
            filled = sentence
            for s, v in zip(matches, combo):
                filled = filled.replace(f"{{{s}}}", v)
            out.append(filled)
    return out
