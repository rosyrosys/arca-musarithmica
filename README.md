# Arca Musarithmica — an interpretable reconstruction

[![tests](https://github.com/rosyrosys/arca-musarithmica/actions/workflows/tests.yml/badge.svg)](https://github.com/rosyrosys/arca-musarithmica/actions/workflows/tests.yml)

A reference implementation of Athanasius Kircher's 1650 *Arca Musarithmica*
(from *Musurgia Universalis*, Book VIII), operationalized as a five-step,
rule-logged pipeline. This repository accompanies the paper

> **Analysis of harmonic structure of Athanasius Kircher's Arca Musarithmica
> for transparent AI music generation**

and is the prototype referenced in its "Future Work" section: an engine that
is (a) faithful to Kircher's published method, (b) auditable step-by-step via
a structured rule log (the XAI layer), and (c) ready to serve as a symbolic
pre/post-processor for modern neural music models.

---

## What this is

Given a text of any length (Latin, English, or Korean), the engine:

1. **Selection** — counts syllables and picks a matching tablet
   (*musarithmus*) from the Pinax.
2. **Conversion** — looks each tablet number up in the *Mensa Tonographica*
   to produce four-voice pitches in a chosen mode.
3. **Constraint check** — diagnoses parallel 5ths/8ves, voice crossings,
   and out-of-range voices.
4. **Cadential template** — classifies the final V→I motion as PAC, IAC,
   HC, PC, or DC.
5. **Output** — emits MIDI, MusicXML, and a JSON rule log.

Every decision — from which tablet is chosen to why a particular cadence is
labeled — is written to a structured `RuleLog` that is both
human-readable (for reviewers and students) and machine-readable (for
downstream evaluation). This directly operationalizes the paper's XAI
argument.

---

## Quick start

```bash
# 1. Run the CLI on a single phrase
python3 -m arca "Cantate Domino canticum novum"

# 2. Run the three-example script (Latin / English / Korean)
python3 examples/example.py

# 3. Run the test suite (no pytest required)
python3 tests/run_tests.py

# 4. Open the interactive demo in any browser (no server needed)
open demo/index.html        # macOS
xdg-open demo/index.html    # Linux
start demo/index.html       # Windows
```

No external Python packages are required. Tested with Python 3.10+.

---

## Why zero dependencies

This is a research-reproducibility choice. Music21 and mido are the obvious
libraries for MIDI / MusicXML work, but pinning one of them would make the
artifact harder for reviewers to reproduce over time. Instead, the MIDI
writer, MusicXML writer, rule-log serializer, syllable counter, and
interactive web demo are all implemented against standard-library primitives
only. The same JavaScript port runs from `demo/index.html` without a build
step or package manager.

---

## Mapping to the paper

| Paper step                | Module / function                            |
|---------------------------|----------------------------------------------|
| 1. Selection              | `arca.engine.Arca._step_selection`           |
| 2. Conversion             | `arca.engine.Arca._step_conversion`          |
| 3. Constraint check       | `arca.constraints.VoiceLeadingChecker.check` |
| 4. Cadential template     | `arca.constraints.CadentialTemplate.apply`   |
| 5. Output                 | `arca.output.MidiWriter`, `MusicXmlWriter`   |
| Rule log (XAI layer)      | `arca.rulelog.RuleLog`                       |
| Mensa Tonographica        | `arca.data.MENSA_F_MAJOR` (Kircher B-natural),`MENSA_F_MAJOR_MODERN` (B-flat) |
| Pinax I, Syntagma I       | `arca.data.PINAX_I_SYNTAGMA_I`               |

### The worked example from the paper

The paper illustrates the method with a 4-syllable Cantus line
`[5,5] → [7,8] → [2,3] → [5,1]` and notes that this resolves to a
V → I imperfect authentic cadence. Tablet `S1.P1.T4.paper` in
`arca/data.py` is a four-voice realization of this example, and running the
engine on any 4-syllable phrase prefers it by default (`tablet_index=0`).

---

## Repository layout

```
arca/                        Package — the engine
  __init__.py                Public API
  __main__.py                CLI entry point (`python3 -m arca "..."`)
  data.py                    Mensa Tonographica + Pinax + voice placement
  rulelog.py                 Structured rule log (the XAI core)
  constraints.py             Voice-leading + cadence detection
  output.py                  Zero-dep MIDI and MusicXML writers
  engine.py                  The five-step orchestrator
  evaluate.py                v0.2 corpus-level metrics (Evaluator, EvaluableComposition)
  hybrid.py                  v0.2 DiscreteChordDiffusion + RuleProjector + HybridSampler
  integrations/              v0.3 external-model adapters
    __init__.py              Public integration API
    base.py                  MidiSequenceAdapter, midi_to_degree_grid
    reference_ngram.py       Trigram trained on Kircher's Pinax (in-sandbox)
    music_transformer.py     HF transformer + miditok adapter
    musicgen.py              audiocraft MusicGen + basic-pitch adapter
demo/
  index.html                 Single-file interactive web demo (JS port)
examples/
  example.py                 Latin / English / Korean end-to-end examples
  hybrid_demo.py             v0.2 Kircher vs diffusion vs hybrid comparison
  integrate_reference.py     v0.3 n-gram integration (sandbox-reproducible)
  integrate_transformer.py   v0.3 HF symbolic transformer (GPU machine)
  integrate_musicgen.py      v0.3 MusicGen audio (GPU machine)
tests/
  test_engine.py             End-to-end invariants
  test_hybrid.py             v0.2 hybrid + evaluator invariants
  test_integrations.py       v0.3 adapter + n-gram invariants
  run_tests.py               Zero-dep test runner (no pytest needed)
output/                      Generated by example.py and the CLI
README.md                    This file
```

---

## CLI reference

```text
usage: arca [-h] [-m MODE] [-t TABLET_INDEX] [-c CADENCE]
            [-o OUTPUT_DIR] [-n NAME] [--no-files] [--json] [--quiet]
            [text]

positional arguments:
  text                  Input text. If omitted, read from stdin.

options:
  -m, --mode            Mensa line / mode (default: F_major).
  -t, --tablet-index    Which tablet to pick when several match (default 0).
  -c, --cadence         Target cadence (PAC|IAC|HC|PC|DC). Mismatches are
                        logged as warnings.
  -o, --output-dir      Where to write outputs (default: ./output).
  -n, --name            Output filename stem (default: 'arca').
  --no-files            Skip file writing; print log only.
  --json                Print the rule log as JSON.
  --quiet               Suppress the human-readable log.
```

The CLI exits with `0` if no warnings were emitted and `1` otherwise —
useful if you want to bolt the engine into a test pipeline.

---

## The rule log as an XAI metric

`RuleLog.coverage()` returns the number of entries per stage. A fully
populated log across all five stages is a direct operational measure of the
interpretability claim in the paper: *each output is accompanied by a
complete, ordered trace from input text to final audio*.

```python
from arca import Arca
result = Arca().compose("Cantate Domino")
print(result.rule_log.coverage())
# {'selection': 3, 'conversion': 4, 'constraint_check': 3,
#  'cadential_template': 1, 'output': 1}
```

Entries with `status="warning"` mark real voice-leading violations that the
engine surfaces rather than silently repairs. These are intentional: they
demonstrate that the rule log catches violations a pure generative model
would likely hide.

---

## The interactive demo

`demo/index.html` is a standalone page that:

- runs a JavaScript port of the Python engine (byte-for-byte parity on
  representative inputs — see `tests/` and the parity checker),
- draws a four-voice score with Kircher's numerals underneath,
- plays the composition through the Web Audio API,
- shows the full rule log with per-stage coloring,
- lets the user download the MIDI file and the JSON rule log.

Open it directly from your filesystem — no server required.

---

## Reproducing the paper's results

```bash
# Reference runs — outputs land in ./output/
python3 examples/example.py

# Self-check: all five stages are covered for every example
python3 tests/run_tests.py

# Round-trip a rule log as JSON for programmatic evaluation
python3 -m arca "Ave Maria" --json --no-files --quiet > out.json
```

Any reviewer with a Python 3.10+ interpreter can run the above in under a
second and obtain identical byte-level outputs (modulo file system
ordering).

---

## v0.2 — hybrid neural-symbolic generation

The v0.2 modules operationalize the follow-up paper's central claim:
Kircher's rule system can wrap a modern generative model as a
*post-sampling projector*, yielding outputs that are fluent (from the
neural side) and auditable (from the symbolic side).

Three new modules:

- `arca.evaluate` — corpus-level metrics: voice-leading violations,
  cadence distribution, rule-log coverage, pitch-class entropy, mean
  voice motion. Works on any four-voice MIDI grid via the
  `EvaluableComposition` adapter, so the same yardstick measures the
  Kircher engine, a pure diffusion baseline, and the hybrid.
- `arca.hybrid.DiscreteChordDiffusion` — a categorical denoising
  sampler over four-voice scale-degree grids. Zero training required;
  the point is to produce a stand-in for a real neural checkpoint so
  the projector has something to correct. Swap with MusicGen / symbolic
  diffusion / MusicVAE via the `DegreeGridSampler` protocol.
- `arca.hybrid.RuleProjector` — snaps each chord to the nearest
  rule-compliant neighbor, emitting a `correction` entry in the rule
  log for every change. This is the XAI mechanism the paper needs:
  every neural output carries a structured trace of what the rules
  changed and why.
- `arca.hybrid.HybridSampler` — combines the two; toggle
  `use_projector` to switch between neural-only and hybrid.

### Reproducing the comparison experiment

```bash
python3 examples/hybrid_demo.py --n 50 --chord-len 6 --seed 42
```

Representative output (N = 50 per system):

```
system          N   mean_viols  %clean  H(pc)  motion  cov%  cadences
kircher_only    50  1.02         4%     2.01   4.11    100%  PAC:37, IAC:10, OTHER:3
diffusion_only  50  1.92        14%     2.14   2.28    100%  OTHER:25, HC:12, DC:10, IAC:3
hybrid          50  1.20        32%     2.24   2.47    100%  IAC:49, PAC:1
```

Key findings to cite in the follow-up paper:

- **Violations drop 38%** (pure diffusion 1.92 → hybrid 1.20 per
  composition) without any retraining of the sampler.
- **Clean-composition rate more than doubles** (14% → 32%).
- **Cadence closure improves dramatically**: pure diffusion produces
  a recognizable cadence in 3/50 cases (6%); hybrid produces one in
  50/50 (100%). This is the direct effect of the bass cadence clamp.
- **Pitch-class entropy is preserved** (2.14 → 2.24, slightly up).
  The projector does not collapse variety — an important control.

All per-composition metrics are written to `output/hybrid_eval.json`
for downstream statistical analysis.

---

## v0.3 — external-model integrations

The v0.3 `arca.integrations` subpackage is the hinge between the
Kircher rule system and off-the-shelf generative checkpoints. It
defines a single protocol — `DegreeGridSampler` — and ships three
adapters against it:

- `arca.integrations.base.MidiSequenceAdapter` — wraps any
  `(n_chords, rng) -> 4xN MIDI grid` callable. The MIDI is inverted to
  Kircher numbers via the Mensa, so the rest of the pipeline (the
  projector, the evaluator) runs unchanged.
- `arca.integrations.reference_ngram` — a trigram chord-level language
  model trained on Kircher's own Pinax. Laplace-smoothed, interpolation
  backoff, zero external dependencies. *Runs in the repo sandbox*, so
  the full hybrid-comparison experiment is reproducible without a GPU.
- `arca.integrations.music_transformer.MusicTransformerAdapter` — for
  symbolic Music Transformers (HF transformers + REMI tokenization via
  miditok). `load_hf_music_transformer("checkpoint-id")` is the
  one-liner.
- `arca.integrations.musicgen.MusicGenAdapter` — for Meta's MusicGen
  audio generator. The waveform is transcribed to MIDI via basic-pitch,
  reduced to SATB by `polyphony_to_satb`, then fed to the same
  projector.

`transformers`, `miditok`, `audiocraft`, and `basic-pitch` are
*lazy-imported* inside their respective loader functions. Importing
`arca.integrations` on a machine without those libraries installed is
always safe.

### Reproducing the integration experiments

```bash
# In-sandbox: the Kircher-trained n-gram + projector comparison
python3 examples/integrate_reference.py --n 20 --chord-len 6 --seed 42

# On a GPU machine with transformers installed
python3 examples/integrate_transformer.py \
    --checkpoint amaai-lab/muse-tiny-remi --n 10

# On a GPU machine with audiocraft installed
python3 examples/integrate_musicgen.py \
    --checkpoint facebook/musicgen-small \
    --prompt "renaissance polyphony, SATB choir" --n 5
```

Representative output from the reference n-gram (N = 20, 6 chords):

```
system      N   mean_viols  %clean  H(pc)  motion  cov%  cadences
ngram_only  20  1.90        20%     2.05   3.29    100%  OTHER:10, HC:8, IAC:2
hybrid      20  0.75        50%     2.16   3.43    100%  IAC:18, PAC:2
```

Key findings for the follow-up paper:

- **Violations drop 60%** (n-gram 1.90 → hybrid 0.75) on a *sampler
  trained on Kircher's own corpus*. The improvement is not because the
  baseline is weak; it is because Kircher's rules are stricter than
  any corpus statistics can enforce.
- **Clean-rate 2.5×** (20% → 50%).
- **Cadence closure 100%** (OTHER dominant → every composition ends
  on an authentic cadence).
- **Entropy preserved** (2.05 → 2.16). The projector does not homogenize.

All three adapters produce a `rule_log` entry labelled
`checkpoint_metadata` when used through the example scripts, so the
external provenance (checkpoint id, text prompt, tokenizer version)
lives inside the XAI log — a requirement for the paper's auditability
claim.

---

## Known limitations / next steps (v0.4+)

- **Pinax coverage.** Only Pinax I, Syntagma I is reconstructed. Full
  digitization of Kircher's Book VIII tables is a transcription project;
  the engine accepts any `Pinax` object and is ready for the extension.
- **Modes.** Only the F-finalis line of the Mensa Tonographica is
  registered. Adding the remaining seven lines is a data-only change.
- **MusicXML spelling.** Enharmonic spelling is currently heuristic
  (sharp-biased, with B-flat for pitch class 10). A proper key-signature-
  aware speller is straightforward to add.
- **Real neural checkpoint.** The diffusion baseline in v0.2 is a
  heuristic categorical denoiser, not a trained model. v0.3 ships
  adapter classes for MusicGen and symbolic Music Transformers via
  `arca.integrations`; results on an actual GPU-loaded checkpoint
  are the obvious next experiment.
- **Constraint-guided sampling.** v0.2 applies the projector as a
  *post*-processor. A stronger variant would use the projector's
  constraint score to reweight the diffusion sampler's intermediate
  distributions — i.e. classifier guidance with a rule-based
  classifier.
- **Human listening evaluation.** The objective metrics in `arca.evaluate`
  should be complemented by blind human preference tests for the
  follow-up paper's main result.
- **Broader Pinax coverage.** Full digitization of Musurgia
  Universalis Book VIII would remove the scale-degree bottleneck.

---

## Citing this prototype

If you use this code in subsequent work, please cite the accompanying
paper and link back to this repository. A `CITATION.cff` will be added
alongside the v0.1 tag on GitHub.

---

## License

Research prototype released under the MIT license (see `LICENSE` once
added for submission). Kircher's 1650 source text is in the public
domain; the Pinax reconstructions in `arca/data.py` are offered as
scholarly annotations and may be freely reused with attribution.
