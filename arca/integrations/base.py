"""
arca.integrations.base
----------------------

The hinge between external generative models and the Kircher pipeline.

Any external model can emit a four-voice MIDI grid — either directly
(a symbolic transformer) or after a transcription step (an audio model
like MusicGen followed by pitch detection). Once we have MIDI, inverting
to Kircher's scale-degree grid is a small lookup:

    Mensa maps   number -> pitch letter
    Mensa_inv    pitch letter -> number

...so `midi_to_degree_grid` just reduces a MIDI matrix modulo 12, resolves
each pitch class to its nearest mensa-consistent scale degree, and hands
the resulting 4xN grid to the rest of the pipeline.

The inversion is deliberately tolerant: out-of-mode pitches snap to the
nearest mode degree. This is the right behavior for a hybrid pipeline,
because the *projector* is where strict compliance is enforced — not
the adapter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..data import MENSA_F_MAJOR, MensaLine, pitch_to_midi


# ---------------------------------------------------------------------------
# Mensa inversion
# ---------------------------------------------------------------------------

def _build_pc_to_degrees(mensa: MensaLine) -> dict:
    """Map each of the 12 pitch classes to the list of mensa degrees that
    share that pitch class. Needed to invert MIDI -> degrees."""
    out: dict = {pc: [] for pc in range(12)}
    for degree, pitch_name in mensa.degree_to_pitch.items():
        pc = pitch_to_midi(pitch_name, 4) % 12
        out[pc].append(degree)
    return out


def midi_to_degree_grid(
    midi_grid: List[List[int]],
    mensa: MensaLine = MENSA_F_MAJOR,
) -> List[List[int]]:
    """Invert a 4xN MIDI grid to a 4xN grid of Kircher numbers (1..8).

    For each MIDI pitch, pick the nearest in-mode degree:
      * exact match if the pc is in the mode
      * otherwise, the degree whose pc is closest modulo 12

    Numbers 1 and 8 both represent the finalis. If the voice is in the
    upper half of the voice's historical range, 8 is preferred over 1;
    otherwise 1. This keeps the `prefer_upper` signal intact when the
    grid is re-realized downstream.
    """
    if len(midi_grid) != 4:
        raise ValueError("midi_to_degree_grid requires 4 voices")
    n = len(midi_grid[0])
    pc_map = _build_pc_to_degrees(mensa)
    finalis_degrees = pc_map[pitch_to_midi(mensa.finalis, 4) % 12]

    # All mode pitch classes, for nearest-pc fallback
    mode_pcs = sorted({pc for pc, ds in pc_map.items() if ds})

    def nearest_pc(target_pc: int) -> int:
        return min(mode_pcs, key=lambda pc:
                   min((pc - target_pc) % 12, (target_pc - pc) % 12))

    # Voice-specific threshold for preferring 8 over 1.
    # Above the voice's center -> 8; below -> 1.
    from ..data import VOICE_CENTERS
    out: List[List[int]] = [[0]*n for _ in range(4)]
    for v in range(4):
        center = VOICE_CENTERS[v]
        for k in range(n):
            midi = midi_grid[v][k]
            pc = midi % 12
            candidates = pc_map.get(pc) or pc_map[nearest_pc(pc)]
            # Pick the finalis variant (1 vs 8) closest to the current octave.
            if set(candidates) & {1, 8}:
                chosen = 8 if midi > center else 1
            else:
                # single-degree pc — just take it
                chosen = candidates[0]
            out[v][k] = chosen
    return out


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# A user-provided MIDI generator has the shape:
#     generate(n_chords: int, rng: random.Random) -> 4xN MIDI grid
# The rng is passed so the adapter can be determined from a seed; the
# external model is free to ignore it.
MidiGenerator = Callable[[int, random.Random], List[List[int]]]


@dataclass
class MidiSequenceAdapter:
    """Wrap any MIDI-producing callable as a `DegreeGridSampler`.

    Usage::

        def my_transformer_midi(n_chords, rng):
            # call HF transformers / music21 / midi file read, etc.
            return four_voice_midi_grid   # shape 4 x n_chords

        adapter = MidiSequenceAdapter(my_transformer_midi)
        hybrid = HybridSampler(sampler=adapter, use_projector=True)

    The adapter is deliberately thin: it inverts MIDI to Kircher numbers
    using the Mensa, and lets the rest of the pipeline (projector,
    evaluator) do its job. The `log_metadata` field is appended to the
    `raw_degree_grid` entry so external provenance (checkpoint name,
    model version) lands in the rule log for auditability.
    """
    generate_midi: MidiGenerator
    mensa: MensaLine = field(default_factory=lambda: MENSA_F_MAJOR)
    log_metadata: dict = field(default_factory=dict)

    # Protocol method
    def sample_degree_grid(
        self, n_chords: int, rng: Optional[random.Random] = None,
    ) -> List[List[int]]:
        r = rng or random.Random()
        midi_grid = self.generate_midi(n_chords, r)
        if len(midi_grid) != 4 or any(len(v) != n_chords for v in midi_grid):
            raise ValueError(
                f"external MIDI generator returned shape "
                f"({len(midi_grid)} x {[len(v) for v in midi_grid]}); "
                f"expected (4 x {n_chords})."
            )
        return midi_to_degree_grid(midi_grid, self.mensa)

    # Introspection: lets examples/integrate_*.py stamp checkpoint info
    # into the rule log for reproducibility.
    def metadata(self) -> dict:
        return dict(self.log_metadata)
