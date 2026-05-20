# linha-fina

OVOS intent pipeline plugin that wraps the **LinhaFina** intent matcher — a
hybrid classifier combining keyword scoring, template matching, dynamic intent
expansion, and fuzzy slot extraction.

## Install

```bash
pip install linha-fina
```

## OPM entry point

```
opm.pipeline:
  ovos-linha-fina-pipeline-plugin = linha_fina.opm:LinhaFinaPipeline
```

The plugin implements `ConfidenceMatcherPipeline` and exposes `match_high`,
`match_medium`, `match_low` matchers used by the OVOS intent service.

## Configuration

In `mycroft.conf` under `intents` or your pipeline section:

```json
{
  "linha_fina": {
    "conf_high": 0.8,
    "conf_med": 0.6,
    "conf_low": 0.4
  }
}
```

| key         | default | meaning                                            |
| ----------- | ------- | -------------------------------------------------- |
| `conf_high` | 0.8     | threshold for `match_high`                         |
| `conf_med`  | 0.6     | threshold for `match_medium`                       |
| `conf_low`  | 0.4     | threshold for `match_low`                          |

Secondary languages from `core_config.secondary_langs` are auto-loaded; each
gets its own `IntentEngine` instance.

## Programmatic use

```python
from linha_fina.engine import IntentEngine

eng = IntentEngine()
eng.register_intent("greet", ["hello", "hi", "hey there"])
eng.register_intent("bye", ["goodbye", "see you"])
eng.train()

print(eng.calc_intent("hello there"))
# IntentMatch(name='greet', conf=..., slots={...})
```

## Pipeline architecture

The pipeline reacts to OVOS messagebus events:

- `padatious:register_intent` / `padatious:register_entity` — register utterance
  samples into the per-language container.
- `detach_intent` / `detach_skill` — clean up on skill unload.
- `mycroft.ready` — trigger initial training across all languages.

Training is lazy: if `mycroft.ready` doesn't fire, the first inference will
trigger a `train()` automatically inside the engine.

## See also

- [Domain (hierarchical) pipeline](domain_pipeline.md)
- [LinhaFina engine source](../linha_fina/engine.py)
- [Benchmark results](../benchs.md)
