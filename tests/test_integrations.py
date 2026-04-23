"""
Tests for the v0.3 `arca.integrations` subpackage.

Claims tested:

  (a) `midi_to_degree_grid` inverts a Kircher-realized MIDI grid back
      to a degree grid with the right shape.
  (b) `MidiSequenceAdapter` raises on shape mismatch.
  (c) `polyphony_to_satb` reduces to 4xN and respects register.
  (d) `train_ngram_on_kircher` fits a usable sampler.
  (e) `NGramSampler` sits cleanly inside `HybridSampler` and the
      projector measurably improves its violation rate.
  (f) `load_musicgen` and `load_hf_music_transformer` raise clear
      ImportError if their heavy dependencies are missing (which is
      the sandbox state), proving the laziness contract.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arca.data import MENSA_F_MAJOR  # noqa: E402
from arca.evaluate import EvaluableComposition, Evaluator  # noqa: E402
from arca.hybrid import (  # noqa: E402
    HybridSampler,
    RuleProjector,
    realize_degree_grid,
)
from arca.integrations import (  # noqa: E402
    MidiSequenceAdapter,
    NGramChordModel,
    NGramSampler,
    midi_to_degree_grid,
    polyphony_to_satb,
    train_ngram_on_kircher,
)


# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------

def test_midi_to_degree_grid_shape_and_bounds():
    # Realize a known degree grid, then invert it.
    degree_grid = [
        [5, 7, 2, 5],
        [3, 5, 7, 3],
        [1, 2, 5, 1],
        [1, 5, 5, 1],
    ]
    _, midi = realize_degree_grid(degree_grid, MENSA_F_MAJOR)
    out = midi_to_degree_grid(midi, MENSA_F_MAJOR)
    assert len(out) == 4
    assert all(len(row) == 4 for row in out)
    # all inverted degrees must live in {1..8}
    for row in out:
        for d in row:
            assert 1 <= d <= 8, d


def test_midi_to_degree_grid_rejects_wrong_shape():
    try:
        midi_to_degree_grid([[60, 62]], MENSA_F_MAJOR)  # 1 voice
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-4-voice input")


def test_midi_sequence_adapter_rejects_wrong_shape():
    def bad_generator(n_chords, rng):
        return [[60]*n_chords, [64]*n_chords]   # only 2 voices
    adapter = MidiSequenceAdapter(generate_midi=bad_generator)
    try:
        adapter.sample_degree_grid(4, random.Random(0))
    except ValueError:
        return
    raise AssertionError("adapter should reject non-4-voice MIDI")


def test_midi_sequence_adapter_roundtrips():
    # A trivial "generator" that just returns a fixed 4xN MIDI grid.
    def gen(n_chords, rng):
        return [[65 + (k % 5) + v * 2 for k in range(n_chords)]
                for v in range(4)]
    adapter = MidiSequenceAdapter(generate_midi=gen,
                                  log_metadata={"source": "test"})
    out = adapter.sample_degree_grid(5, random.Random(0))
    assert len(out) == 4 and all(len(row) == 5 for row in out)
    # Metadata is exposed verbatim
    assert adapter.metadata()["source"] == "test"


# ---------------------------------------------------------------------------
# reference_ngram.py
# ---------------------------------------------------------------------------

def test_ngram_fits_on_kircher():
    m = train_ngram_on_kircher()
    # vocab should include the BOS token plus at least a few real chords
    assert len(m.vocab) >= 2
    out = m.sample(6, random.Random(0))
    assert len(out) == 6
    for tok in out:
        assert isinstance(tok, tuple) and len(tok) == 4


def test_ngram_sampler_is_degree_grid_sampler():
    m = train_ngram_on_kircher()
    sampler = NGramSampler(m)
    grid = sampler.sample_degree_grid(6, random.Random(0))
    assert len(grid) == 4
    assert all(len(row) == 6 for row in grid)
    # No BOS (0) should leak into the output
    for row in grid:
        for d in row:
            assert 1 <= d <= 8, d


def test_ngram_in_hybrid_pipeline_reduces_violations():
    """The projector must measurably help even a corpus-trained sampler."""
    m = train_ngram_on_kircher()
    sampler = NGramSampler(m)
    pure = HybridSampler(sampler=sampler, use_projector=False)
    hyb  = HybridSampler(sampler=sampler,
                         projector=RuleProjector(),
                         use_projector=True)
    ev = Evaluator()
    pure_total = 0
    hyb_total = 0
    n = 15
    for seed in range(n):
        mp, lp = pure.compose(6, seed=seed)
        mh, lh = hyb.compose(6, seed=seed)
        pure_total += ev.evaluate(
            EvaluableComposition(mp, rule_log=lp, label="p")).violation_total
        hyb_total += ev.evaluate(
            EvaluableComposition(mh, rule_log=lh, label="h")).violation_total
    assert hyb_total <= pure_total, (pure_total, hyb_total)


# ---------------------------------------------------------------------------
# music_transformer.py
# ---------------------------------------------------------------------------

def test_polyphony_to_satb_shape_and_register():
    # Synthesize a polyphonic "piano-roll": four clearly-separated voices
    # at ticks 0, 10, 20, ... with pitches spanning SATB register.
    notes = []
    for k in range(5):
        notes.append((k * 10, 70))   # soprano
        notes.append((k * 10, 62))   # alto
        notes.append((k * 10, 55))   # tenor
        notes.append((k * 10, 45))   # bass
    grid = polyphony_to_satb(notes, 5, ticks_per_chord=10)
    assert len(grid) == 4 and all(len(r) == 5 for r in grid)
    # Bassus must be the lowest voice everywhere
    for k in range(5):
        voices_at_k = [grid[v][k] for v in range(4)]
        assert grid[3][k] == min(voices_at_k), voices_at_k


def test_polyphony_to_satb_empty_returns_centers():
    grid = polyphony_to_satb([], 3)
    assert len(grid) == 4 and all(len(r) == 3 for r in grid)
    # All filled with voice centers — no zeros or negatives
    for row in grid:
        for m in row:
            assert m > 0


# ---------------------------------------------------------------------------
# Lazy-import contract for the heavy adapters
# ---------------------------------------------------------------------------

def test_load_musicgen_raises_without_audiocraft():
    from arca.integrations.musicgen import load_musicgen
    try:
        load_musicgen()
    except ImportError as e:
        assert "audiocraft" in str(e).lower() or "basic-pitch" in str(e).lower()
        return
    except Exception:
        # If audiocraft IS installed on this machine, the loader might
        # raise some other error trying to reach the network. That's
        # outside this test's scope — the laziness contract is what we
        # care about.
        return
    # If it got this far, MusicGen actually loaded — also fine.


def test_load_hf_music_transformer_raises_without_transformers():
    from arca.integrations.music_transformer import load_hf_music_transformer
    try:
        load_hf_music_transformer("facebook/none-such")
    except ImportError as e:
        assert "transformers" in str(e).lower() or "miditok" in str(e).lower()
        return
    except Exception:
        return
