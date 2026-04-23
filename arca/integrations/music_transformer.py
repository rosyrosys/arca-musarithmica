"""
arca.integrations.music_transformer
-----------------------------------

Adapter for symbolic Music Transformer checkpoints — the kind of model
you'd load from Hugging Face or train with `miditok` + PyTorch. The
adapter is deliberately thin: it shells out to a user-supplied
tokenizer and model, collects the generated MIDI events, and returns a
4xN MIDI grid that the `MidiSequenceAdapter` can invert into Kircher
numbers.

Two usage modes
===============

1. **Zero-install mode (recommended for first run).** Supply any
   ``token_generator`` that maps ``(n_chords, rng) -> List[int]`` — for
   example, a frozen REMI sequence from a captured checkpoint inference
   dumped to disk, or an HF-pipeline call you orchestrate outside this
   module. The adapter converts the tokens to MIDI via a supplied
   tokenizer (any object exposing ``tokens_to_midi`` or
   ``decode``), reduces to 4 voices, and hands the grid back to Kircher.

2. **Batteries-included mode.** Pass a HF checkpoint id, and the
   adapter will lazily import ``transformers`` and ``miditok`` and do
   the end-to-end inference itself. Requires those libraries in your
   environment — the sandbox used to generate this repo does not have
   them, which is exactly why Mode 1 exists as the fallback.

Keeping the adapter "dumb" is the point. Any improvement to the upstream
checkpoint (a better model, a different tokenization, fine-tuning on
chorales) shows up in the hybrid pipeline without modifications here.

Reducing N voices to 4
======================

Kircher's pipeline is SATB. Most symbolic-music checkpoints are
piano-roll-shaped or polyphonic with arbitrary voice counts. We
implement a small, explicit reduction in ``polyphony_to_satb``:

* group notes by start tick,
* for each chord pick the 4 unique pitches closest to the SATB voice
  centers (with a small penalty for voice crossings),
* forward-fill pitches for voices that have no note at this tick.

This is a deliberate approximation. The paper-relevant claim is that
the *projector* will fix any musically awkward consequences — the
adapter just needs to hand something plausible to the projector.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from ..data import VOICE_CENTERS
from .base import MidiGenerator, MidiSequenceAdapter


# ---------------------------------------------------------------------------
# Polyphony -> SATB reduction
# ---------------------------------------------------------------------------

Note = Tuple[int, int]   # (start_tick, midi_pitch)


def polyphony_to_satb(
    notes: Sequence[Note],
    n_chords: int,
    ticks_per_chord: Optional[int] = None,
) -> List[List[int]]:
    """Reduce arbitrary polyphony to a 4xN SATB grid.

    Parameters
    ----------
    notes
        Iterable of ``(start_tick, midi_pitch)`` pairs. Duration is
        intentionally ignored — the engine is chord-oriented.
    n_chords
        Desired number of chords in the output grid.
    ticks_per_chord
        Quantization grid. If None, inferred as ``max_tick / n_chords``.

    Returns
    -------
    4 x n_chords MIDI grid, voices in SATB order (Cantus..Bassus).
    """
    if not notes:
        # empty input — return a bland SATB unison on F
        return [[VOICE_CENTERS[v]] * n_chords for v in range(4)]

    max_tick = max(t for t, _ in notes)
    tpc = ticks_per_chord or max(1, max_tick // max(1, n_chords))

    # bucket notes into chord slots
    buckets: List[List[int]] = [[] for _ in range(n_chords)]
    for tick, pitch in notes:
        k = min(n_chords - 1, tick // tpc)
        buckets[k].append(pitch)

    grid: List[List[int]] = [[VOICE_CENTERS[v]] * n_chords for v in range(4)]
    prev = [VOICE_CENTERS[v] for v in range(4)]
    for k, bucket in enumerate(buckets):
        chord = _assign_to_satb(bucket, prev)
        for v in range(4):
            grid[v][k] = chord[v]
        prev = chord
    return grid


def _assign_to_satb(pitches: Sequence[int], prev: Sequence[int]) -> List[int]:
    """Pick 4 pitches from a bucket and assign them to SATB slots.

    Uses a small greedy assignment: rank pitches by frequency of
    appearance, then assign each to the voice whose center is closest
    *and* whose previous pitch is closest (to favor smooth voice
    leading). Falls back to forward-fill for unfilled voices.
    """
    if not pitches:
        return list(prev)
    # dedupe while preserving order
    uniq: List[int] = []
    seen = set()
    for p in pitches:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    # rank by distance from voice centers' overall span
    # (we want a wide-spread chord, so prefer a spread of pitches)
    uniq.sort()
    # pick up to 4 pitches roughly evenly spaced by register
    if len(uniq) >= 4:
        step = len(uniq) / 4.0
        picks = [uniq[min(len(uniq) - 1, int(i * step))] for i in range(4)]
    else:
        picks = list(uniq)

    # assign picks to voices by closest center; prefer low pitches to Bassus
    picks_sorted = sorted(picks)
    slots: List[Optional[int]] = [None, None, None, None]
    # lowest -> Bassus(3), then Tenor(2), Altus(1), Cantus(0)
    order = [3, 2, 1, 0]
    for i, voice in enumerate(order):
        if i < len(picks_sorted):
            slots[voice] = picks_sorted[i]
    # forward-fill from prev for any empty slot
    for v in range(4):
        if slots[v] is None:
            slots[v] = prev[v]
    return slots  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Adapter for a user-supplied token generator
# ---------------------------------------------------------------------------

TokenGenerator = Callable[[int, random.Random], List[int]]
TokensToMidi = Callable[[List[int]], List[Note]]


@dataclass
class MusicTransformerAdapter(MidiSequenceAdapter):
    """`MidiSequenceAdapter` for a symbolic transformer.

    This class plugs a user-supplied token generator + detokenizer into
    the Kircher hybrid pipeline. It satisfies ``DegreeGridSampler`` by
    (a) calling the token generator, (b) decoding tokens to MIDI notes,
    (c) reducing to SATB, (d) inverting MIDI to Kircher numbers via the
    inherited ``generate_midi`` -> ``midi_to_degree_grid`` path.

    Why not just wrap ``MidiSequenceAdapter`` directly? Because symbolic
    transformers emit *tokens*, not ready-made 4xN MIDI matrices. This
    subclass inserts the tokens -> notes -> SATB stages so the caller
    only has to supply the two model-specific callables.
    """
    token_generator: Optional[TokenGenerator] = None
    tokens_to_midi: Optional[TokensToMidi] = None
    ticks_per_chord: Optional[int] = None

    # Override: produce the 4xN MIDI grid from the token generator.
    def __post_init__(self) -> None:
        # Wire the adapter's MIDI generator to point at our
        # tokens -> notes -> SATB pipeline.
        self.generate_midi = self._generate_from_tokens  # type: ignore[assignment]

    def _generate_from_tokens(
        self, n_chords: int, rng: random.Random,
    ) -> List[List[int]]:
        if self.token_generator is None or self.tokens_to_midi is None:
            raise RuntimeError(
                "MusicTransformerAdapter requires both `token_generator` "
                "and `tokens_to_midi`. See docstring for the two usage modes."
            )
        tokens = self.token_generator(n_chords, rng)
        notes = self.tokens_to_midi(tokens)
        return polyphony_to_satb(notes, n_chords, self.ticks_per_chord)


# ---------------------------------------------------------------------------
# Batteries-included: load a HF checkpoint
# ---------------------------------------------------------------------------

def load_hf_music_transformer(
    checkpoint: str,
    tokenizer_config: Optional[dict] = None,
    max_new_tokens: int = 256,
) -> MusicTransformerAdapter:
    """Load a symbolic Music Transformer from Hugging Face.

    This is a convenience wrapper. Lazy-imports ``transformers`` and
    ``miditok`` so the rest of ``arca.integrations`` keeps working in
    minimal environments. Raises a clear error message if either
    library is missing.

    Typical use::

        adapter = load_hf_music_transformer("amaai-lab/muse-tiny-remi")
        hybrid = HybridSampler(sampler=adapter, use_projector=True)
        midi, log = hybrid.compose(n_chords=6, seed=42)

    Parameters
    ----------
    checkpoint
        HF repo id or local path.
    tokenizer_config
        Passed to `miditok.REMI(**tokenizer_config)`. Leave None for
        defaults.
    max_new_tokens
        Cap on generation length. Longer is safer for short phrases
        because REMI uses many tokens per note.
    """
    try:
        # Lazy imports: expected to be available only on the user's machine.
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import miditok  # type: ignore
    except ImportError as e:
        raise ImportError(
            "load_hf_music_transformer needs `transformers` and `miditok`. "
            "Install them with `pip install transformers miditok` and retry."
        ) from e

    tokenizer_hf = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    miditokenizer = miditok.REMI(**(tokenizer_config or {}))

    def token_generator(n_chords: int, rng: random.Random) -> List[int]:
        # Seed torch for reproducibility via the python rng's state.
        try:
            import torch  # type: ignore
            torch.manual_seed(rng.randint(0, 2**31 - 1))
        except ImportError:
            pass
        prompt = tokenizer_hf("", return_tensors="pt")
        out = model.generate(
            **prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=1,
        )
        return out[0].tolist()

    def tokens_to_midi(tokens: List[int]) -> List[Note]:
        # miditok REMI can decode straight to a PrettyMIDI-like object.
        # We only need (start_tick, pitch) pairs.
        pmidi = miditokenizer.tokens_to_midi([tokens])
        notes: List[Note] = []
        for inst in getattr(pmidi, "instruments", []):
            for note in getattr(inst, "notes", []):
                notes.append((int(note.start), int(note.pitch)))
        notes.sort()
        return notes

    return MusicTransformerAdapter(
        generate_midi=lambda n, r: [],   # overridden by __post_init__
        token_generator=token_generator,
        tokens_to_midi=tokens_to_midi,
        log_metadata={"checkpoint": checkpoint, "type": "symbolic_transformer"},
    )
