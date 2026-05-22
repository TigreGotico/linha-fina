# Benchmark

Linha Fina ships a comparative accuracy and speed benchmark in
`benchmark/compare.py`. It runs on two OpenVoiceOS evaluation datasets and
reports this repo's three engines side by side with three fixed external
baselines, so results are comparable across the OVOS intent-engine family.

---

## Datasets

Both datasets are loaded from the Hugging Face Hub by `benchmark/dataset.py`.
Each has a `<lang>-templates` config (training templates) and a `<lang>-test`
config (labelled evaluation utterances). Every engine in this benchmark is a
template / sample matcher, so it trains on `-templates` and is evaluated on
`-test`.

| Name | Repo | Intents | Test cases | Notes |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 1750 | Six test splits, including a `far_ood` no-match set |
| `massive` | [`OpenVoiceOS/massive-templates`](https://huggingface.co/datasets/OpenVoiceOS/massive-templates) | 60 | 2974 | OVOS-templated rebuild of the MASSIVE corpus; one labelled split, no no-match cases |

`intents-for-eval` test splits:

| Split | Cases | What it tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

`massive` has a single labelled `test` split and **no no-match cases** — so on
`massive` every engine has zero false positives by construction, and accuracy
equals recall.

### Entities

Each `{slot}` placeholder ships with example values. `benchmark/dataset.py`
collects them into a `Bundle.entities` map. The padaos / padatious / nebulento
baselines register them as named entities (the equivalent of a padatious
`.entity` file) before matching. Linha Fina has an entity-registration API:
`IntentEngine.register_intent` and the domain / hierarchical engines'
`register_domain_intent` accept an `entity_samples` mapping, so the slot
examples are passed in directly per intent.

---

## Engines Compared

The three `linha-fina` rows are the subject of this benchmark. The
padaos / padatious / nebulento rows are fixed external baselines, shared
verbatim across the OVOS intent-engine benchmark family.

| Engine | Role | Description |
|---|---|---|
| `padaos` | baseline | Regex-based exact matcher (no fuzzy) |
| `padatious` | baseline | Neural network matcher (requires a training pass) |
| `nebulento damerau-levenshtein` | baseline | Flat `IntentContainer` at the default `DAMERAU_LEVENSHTEIN_SIMILARITY` strategy |
| `linha-fina flat` | subject | This repo's `IntentEngine` — one SVM per intent over the whole corpus |
| `linha-fina domain` | subject | `DomainIntentEngine` — intents grouped by domain, parallel-argmax routing |
| `linha-fina hierarchical` | subject | `HierarchicalIntentEngine` — two-stage domain classification then intent matching |

---

## How to Run

Install benchmark dependencies:

```bash
pip install linha-fina[benchmark]
# installs: padaos, padatious, fann2==1.0.7, nebulento, datasets
```

Run both datasets:

```bash
python benchmark/compare.py
```

Or one at a time:

```bash
python benchmark/compare.py intents-for-eval
python benchmark/compare.py massive
```

The first run downloads each dataset from the Hugging Face Hub (cached
afterwards). Padatious requires a training pass; the other engines start
immediately.

---

## How Metrics Are Calculated

Source: `compute_metrics` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2 × precision × recall / (precision + recall)
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the
expected intent and `conf >= threshold` (0.5). A no-match case is correct only
when the engine returns no intent or a confidence below threshold.

---

## Interpreting the Results

- **`linha-fina flat`** trains one SVM per intent over the whole corpus — the
  rest of the corpus is each SVM's negative set.
- **`linha-fina domain`** partitions intents into per-domain `IntentEngine`s
  and resolves a query by parallel-argmax across them. Each SVM then sees only
  its domain siblings as negatives, lightening negative-set imbalance.
- **`linha-fina hierarchical`** classifies the domain first with a dedicated
  top-level classifier, then scores only that domain's engine. The
  `domain_threshold` gate (set to `0.0` in this benchmark, so every query is
  routed) can reject off-topic utterances before any intent is scored.

Two-stage routing trades recall for off-topic rejection: a query routed to the
wrong domain cannot be recovered, so the hierarchical engine helps most when
domains are lexically distinct and false-positive rejection matters more than
catching every command.
