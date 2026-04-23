"""
arca.constraints
----------------

Voice-leading constraint checks and cadential template enforcement.

These are the paper's Step 3 (constraint check) and Step 4 (cadential
template). Each check emits structured RuleLog entries that are the
audit trail of the XAI layer.

Historical scope: the rules implemented here are the minimal set
described as relevant to the source paper:

  * No parallel perfect fifths between adjacent chords.
  * No parallel perfect octaves between adjacent chords.
  * No voice crossing (each voice stays in its SATB range).
  * Cadential closure: final progression is V -> I (or a named variant).

These are diagnostic, not generative: a failed check does NOT rewrite
the tablet; it emits a warning (paper's terminology: "transient
non-chord tones ... resolved according to voice-leading conventions").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .data import MensaLine, Tablet, pitch_to_midi, DEFAULT_VOICE_OCTAVES
from .rulelog import RuleLog, STAGE_CONSTRAINT, STAGE_CADENCE


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _interval_semitones(note_low: int, note_high: int) -> int:
    """Return |note_high - note_low| in semitones."""
    return abs(note_high - note_low)


def _interval_class(semitones: int) -> int:
    """Reduce to an octave-equivalent interval class (0-11)."""
    return semitones % 12


PERFECT_FIFTH = 7
PERFECT_OCTAVE = 0       # same pitch class, can be unison or 8ve


# ---------------------------------------------------------------------------
# Voice-leading checker
# ---------------------------------------------------------------------------

@dataclass
class VoiceLeadingChecker:
    """Diagnostic check for forbidden parallel motion and voice crossing.

    Works on a 4xN matrix of MIDI pitches (voice x chord-position).
    """

    def check(self, midi_grid: List[List[int]], rule_log: RuleLog) -> None:
        self._check_parallels(midi_grid, rule_log)
        self._check_voice_crossing(midi_grid, rule_log)
        self._check_range(midi_grid, rule_log)

    # -- individual checks ---------------------------------------------------

    def _check_parallels(
        self, midi_grid: List[List[int]], rule_log: RuleLog
    ) -> None:
        voice_names = ["Cantus", "Altus", "Tenor", "Bassus"]
        n_chords = len(midi_grid[0])
        found_any = False
        for i in range(4):
            for j in range(i + 1, 4):
                for k in range(n_chords - 1):
                    a_now, a_next = midi_grid[i][k], midi_grid[i][k + 1]
                    b_now, b_next = midi_grid[j][k], midi_grid[j][k + 1]
                    if a_now == a_next and b_now == b_next:
                        continue  # both static — not a parallel motion
                    iv_now = _interval_class(_interval_semitones(a_now, b_now))
                    iv_next = _interval_class(_interval_semitones(a_next, b_next))
                    moved = (a_now != a_next) and (b_now != b_next)
                    if moved and iv_now == iv_next:
                        if iv_now == PERFECT_FIFTH:
                            found_any = True
                            rule_log.add(
                                stage=STAGE_CONSTRAINT,
                                rule="no_parallel_fifths",
                                message=(
                                    f"Parallel perfect 5th between "
                                    f"{voice_names[i]} and {voice_names[j]} "
                                    f"at chord {k} -> {k + 1}."
                                ),
                                data={
                                    "voices": [voice_names[i], voice_names[j]],
                                    "chord_from": k,
                                    "chord_to": k + 1,
                                    "interval": "P5",
                                },
                                status="warning",
                            )
                        elif iv_now == PERFECT_OCTAVE and a_now != b_now:
                            # same pitch class across an octave
                            found_any = True
                            rule_log.add(
                                stage=STAGE_CONSTRAINT,
                                rule="no_parallel_octaves",
                                message=(
                                    f"Parallel perfect octave between "
                                    f"{voice_names[i]} and {voice_names[j]} "
                                    f"at chord {k} -> {k + 1}."
                                ),
                                data={
                                    "voices": [voice_names[i], voice_names[j]],
                                    "chord_from": k,
                                    "chord_to": k + 1,
                                    "interval": "P8",
                                },
                                status="warning",
                            )
        if not found_any:
            rule_log.add(
                stage=STAGE_CONSTRAINT,
                rule="no_parallel_fifths_or_octaves",
                message="No parallel perfect 5ths or 8ves detected across any voice pair.",
                data={},
                status="ok",
            )

    def _check_voice_crossing(
        self, midi_grid: List[List[int]], rule_log: RuleLog
    ) -> None:
        # Convention: Cantus > Altus > Tenor > Bassus at every chord.
        voice_names = ["Cantus", "Altus", "Tenor", "Bassus"]
        n_chords = len(midi_grid[0])
        any_crossing = False
        for k in range(n_chords):
            pitches = [midi_grid[i][k] for i in range(4)]
            for i in range(3):
                if pitches[i] < pitches[i + 1]:
                    any_crossing = True
                    rule_log.add(
                        stage=STAGE_CONSTRAINT,
                        rule="no_voice_crossing",
                        message=(
                            f"{voice_names[i]} ({pitches[i]}) is below "
                            f"{voice_names[i + 1]} ({pitches[i + 1]}) at chord {k}."
                        ),
                        data={
                            "chord": k,
                            "upper_voice": voice_names[i],
                            "lower_voice": voice_names[i + 1],
                            "upper_midi": pitches[i],
                            "lower_midi": pitches[i + 1],
                        },
                        status="warning",
                    )
        if not any_crossing:
            rule_log.add(
                stage=STAGE_CONSTRAINT,
                rule="no_voice_crossing",
                message="Voice order preserved: Cantus > Altus > Tenor > Bassus at every chord.",
                data={},
                status="ok",
            )

    def _check_range(
        self, midi_grid: List[List[int]], rule_log: RuleLog
    ) -> None:
        # Minimal SATB-style range checks.
        ranges = {
            0: (60, 81),   # Cantus: C4 .. A5
            1: (53, 74),   # Altus:  F3 .. D5
            2: (48, 69),   # Tenor:  C3 .. A4
            3: (40, 62),   # Bassus: E2 .. D4
        }
        voice_names = ["Cantus", "Altus", "Tenor", "Bassus"]
        n_chords = len(midi_grid[0])
        out_of_range = 0
        for i in range(4):
            lo, hi = ranges[i]
            for k in range(n_chords):
                p = midi_grid[i][k]
                if p < lo or p > hi:
                    out_of_range += 1
                    rule_log.add(
                        stage=STAGE_CONSTRAINT,
                        rule="voice_range",
                        message=(
                            f"{voice_names[i]} pitch MIDI {p} at chord {k} is "
                            f"outside the conventional range [{lo}, {hi}]."
                        ),
                        data={"voice": voice_names[i], "chord": k,
                              "midi": p, "lo": lo, "hi": hi},
                        status="warning",
                    )
        if out_of_range == 0:
            rule_log.add(
                stage=STAGE_CONSTRAINT,
                rule="voice_range",
                message="All four voices stay within their conventional SATB ranges.",
                data={},
                status="ok",
            )


# ---------------------------------------------------------------------------
# Cadential template
# ---------------------------------------------------------------------------

@dataclass
class CadentialTemplate:
    """Verify and label the final cadence type of a tablet.

    The paper identifies PAC, IAC, HC, PC, DC as the relevant categories,
    with IAC dominant in Pinax I. This class reports the detected cadence
    and, if a target was specified, flags mismatch.
    """
    target: Optional[str] = None     # "PAC" | "IAC" | "HC" | "PC" | "DC" | None

    def apply(
        self,
        midi_grid: List[List[int]],
        mensa: MensaLine,
        rule_log: RuleLog,
    ) -> str:
        """Inspect the last two chords; return cadence label (+log it)."""
        if len(midi_grid[0]) < 2:
            rule_log.add(
                stage=STAGE_CADENCE,
                rule="cadence_detection",
                message="Tablet has < 2 chords; cadence detection skipped.",
                data={},
                status="info",
            )
            return "NONE"

        bass_last2 = (midi_grid[3][-2], midi_grid[3][-1])
        cantus_last = midi_grid[0][-1]

        # Derive tonic pitch class from mensa finalis.
        finalis_pc = pitch_to_midi(mensa.finalis, 4) % 12
        dominant_pc = (finalis_pc + 7) % 12

        bass_from_pc = bass_last2[0] % 12
        bass_to_pc = bass_last2[1] % 12
        cantus_pc = cantus_last % 12

        label = "OTHER"
        reason = ""

        if bass_from_pc == dominant_pc and bass_to_pc == finalis_pc:
            # V -> I in the bass.
            if cantus_pc == finalis_pc:
                label = "PAC"
                reason = (
                    "Bass V -> I and Cantus lands on the finalis — "
                    "perfect authentic cadence."
                )
            else:
                label = "IAC"
                reason = (
                    "Bass V -> I but Cantus does not land on the finalis — "
                    "imperfect authentic cadence."
                )
        elif bass_to_pc == dominant_pc:
            label = "HC"
            reason = "Final chord is a dominant — half cadence."
        elif bass_from_pc == (finalis_pc + 5) % 12 and bass_to_pc == finalis_pc:
            # IV -> I
            label = "PC"
            reason = "Bass IV -> I — plagal cadence."
        elif bass_from_pc == dominant_pc and bass_to_pc != finalis_pc:
            label = "DC"
            reason = "Bass V -> (something other than I) — deceptive cadence."
        else:
            reason = "No standard cadence pattern detected in the bass."

        status = "ok"
        if self.target is not None and self.target != label:
            status = "warning"
            reason += f" (Expected {self.target}, detected {label}.)"

        rule_log.add(
            stage=STAGE_CADENCE,
            rule="cadence_detection",
            message=reason,
            data={
                "detected": label,
                "expected": self.target,
                "bass_progression_pc": [bass_from_pc, bass_to_pc],
                "cantus_final_pc": cantus_pc,
                "finalis_pc": finalis_pc,
                "dominant_pc": dominant_pc,
            },
            status=status,
        )
        return label
