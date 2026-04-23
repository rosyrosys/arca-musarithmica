"""
arca.integrations
-----------------

Adapters that connect external generative checkpoints (MusicGen, symbolic
Music Transformers, n-gram language models, ...) to the `HybridSampler`
interface. Every adapter exposes the `DegreeGridSampler` protocol defined
in `arca.hybrid`, so the rest of the pipeline — the RuleProjector, the
evaluator, the comparison experiment — works unchanged.

Design principle: *the integration is a one-liner*. The point of the
follow-up paper is the Kircher-rules layer, not the specific model; the
adapters exist so any off-the-shelf checkpoint can be the neural half
of the hybrid without engine changes.

Modules
-------

    base                — `MidiSequenceAdapter`: wrap any MIDI-producing
                          callable as a `DegreeGridSampler`. The core hinge.
    music_transformer   — symbolic transformer (HF transformers / miditok)
                          + polyphony_to_satb reduction
    musicgen            — audio transformer (audiocraft + pitch transcription)
    reference_ngram     — concrete n-gram trained on Kircher corpus,
                          runs in-sandbox with zero external deps. Proves
                          the adapter architecture works end-to-end.

Only the `reference_ngram` and `base` modules are self-contained. The
`music_transformer` and `musicgen` modules lazy-import their heavy
dependencies inside their loader functions, so importing this package
never fails on an environment without GPUs or transformers installed.
"""

from .base import MidiGenerator, MidiSequenceAdapter, midi_to_degree_grid
from .music_transformer import (
    MusicTransformerAdapter,
    load_hf_music_transformer,
    polyphony_to_satb,
)
from .musicgen import MusicGenAdapter, load_musicgen
from .reference_ngram import (
    NGramChordModel,
    NGramSampler,
    train_ngram_on_kircher,
)

__all__ = [
    # base
    "MidiGenerator",
    "MidiSequenceAdapter",
    "midi_to_degree_grid",
    # symbolic transformer
    "MusicTransformerAdapter",
    "load_hf_music_transformer",
    "polyphony_to_satb",
    # audio transformer
    "MusicGenAdapter",
    "load_musicgen",
    # reference n-gram
    "NGramChordModel",
    "NGramSampler",
    "train_ngram_on_kircher",
]
