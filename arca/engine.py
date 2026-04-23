"""
arca.engine
-----------

The explicit five-step algorithm of Kircher's Arca Musarithmica,
reconstructed as an auditable, rule-logging pipeline.

From the source paper's "Operationalization of Kircher's Method":

    1. Selection      — a numeric array is selected from Kircher's tables
                        according to poetic meter.
    2. Conversion     — the numbers are mapped to pitches within a chosen key.
    3. Constraint     — voice-leading rules (e.g. no parallel fifths/octaves)
                        are applied; cadential closure is ensured.
    4. Cadential tpl. — the final cadence (V -> I) is enforced / labeled.
    5. Output         — the four-voice harmony is presented with a concise
                        explanatory rule log.

This module makes each step a named method whose log entries feed the
unified `RuleLog`, so that the mapping from input text to output music is
fully traceable — the operational answer to the paper's XAI argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .data import (
    MENSA_F_MAJOR,
    MensaLine,
    MensaTonographica,
    Pinax,
    PINAX_I_SYNTAGMA_I,
    Tablet,
    pitch_to_midi,
    place_in_voice,
    place_chord,
    VOICE_RANGES,
)
from .constraints import CadentialTemplate, VoiceLeadingChecker
from .rulelog import (
    RuleLog,
    STAGE_CADENCE,
    STAGE_CONSTRAINT,
    STAGE_CONVERSION,
    STAGE_OUTPUT,
    STAGE_SELECTION,
)
from .output import MidiNote, MidiWriter, MusicXmlWriter


VOICE_NAMES = ("Cantus", "Altus", "Tenor", "Bassus")


# ---------------------------------------------------------------------------
# Syllable counting
# ---------------------------------------------------------------------------

def _count_syllables_english_latin(word: str) -> int:
    """Vowel-group counting, a cheap but serviceable heuristic for Latin and
    English. Adjusted for Latin by NOT treating trailing 'e' as silent."""
    if not word:
        return 0
    vowels = "aeiouyAEIOUY"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    return max(1, count)


def _count_syllables_korean(text: str) -> int:
    """Count Korean Hangul syllable blocks (U+AC00..U+D7A3)."""
    return sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7A3)


def count_syllables(text: str) -> int:
    """Language-heuristic syllable count, supporting Latin / English / Korean
    mixtures. Splits on whitespace and punctuation, sums per-token counts."""
    n_total = 0
    # Split on whitespace; punctuation removed per-token below.
    for token in re.split(r"\s+", text.strip()):
        if not token:
            continue
        token = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]", "", token)
        if not token:
            continue
        korean = _count_syllables_korean(token)
        if korean > 0:
            n_total += korean
        else:
            n_total += _count_syllables_english_latin(token)
    return n_total


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CompositionResult:
    """Full result of a run of the five-step pipeline."""
    input_text: str
    syllable_count: int
    tablet: Tablet
    mensa: MensaLine
    # 4xN grid of pitch names (letters, with accidental)
    pitch_grid: List[List[str]] = field(default_factory=list)
    # 4xN grid of MIDI numbers
    midi_grid: List[List[int]] = field(default_factory=list)
    cadence_detected: str = ""
    rule_log: RuleLog = field(default_factory=RuleLog)

    # -- pretty printing -----------------------------------------------------

    def chord_sequence(self) -> List[List[str]]:
        """Return chords as lists of pitch names, top-down (Cantus..Bassus)."""
        n = len(self.pitch_grid[0])
        return [[self.pitch_grid[v][k] for v in range(4)] for k in range(n)]

    def pretty(self) -> str:
        lines = [
            f"Input text       : {self.input_text!r}",
            f"Syllable count   : {self.syllable_count}",
            f"Tablet selected  : {self.tablet.id}  "
            f"(cadence={self.tablet.cadence}, provenance={self.tablet.provenance})",
            f"Mensa line       : {self.mensa.name}",
            f"Cadence detected : {self.cadence_detected}",
            "",
            "Four-voice grid (top-down Cantus, Altus, Tenor, Bassus):",
        ]
        for i in range(4):
            pitch_row = " ".join(f"{p:>3}" for p in self.pitch_grid[i])
            midi_row = " ".join(f"{m:>3}" for m in self.midi_grid[i])
            lines.append(f"  {VOICE_NAMES[i]:<7} pitches : {pitch_row}")
            lines.append(f"  {VOICE_NAMES[i]:<7} MIDI    : {midi_row}")
        lines.append("")
        lines.append(self.rule_log.human_readable())
        return "\n".join(lines)

    # -- export --------------------------------------------------------------

    def to_midi(self, path: str, tempo_bpm: float = 60.0) -> None:
        """Write a 4-track MIDI file (one track per voice)."""
        mw = MidiWriter(ticks_per_quarter=480, tempo_bpm=tempo_bpm)
        tpq = mw.tpq
        # Each chord position gets a uniform duration per tablet.rhythm.
        rhythm = self.tablet.rhythm
        for v in range(4):
            notes: List[MidiNote] = []
            t = 0
            for k in range(len(self.midi_grid[v])):
                dur = int(round(rhythm[k] * tpq))
                notes.append(MidiNote(
                    pitch=self.midi_grid[v][k],
                    start=t,
                    duration=dur,
                    velocity=80,
                    channel=v,
                ))
                t += dur
            mw.add_track(notes, name=VOICE_NAMES[v])
        mw.save(path)

    def to_musicxml(self, path: str) -> None:
        """Write a 4-part MusicXML score."""
        mx = MusicXmlWriter(
            title=f"Arca Musarithmica — {self.tablet.id}",
            composer="Kircher (1650) / Arca Engine reconstruction",
        )
        # Distribute syllables across voices as lyrics (on Cantus).
        syllables = re.findall(r"[\w\u3131-\u318E\uAC00-\uD7A3]+", self.input_text)
        flat_syllables: List[str] = []
        for token in syllables:
            # naive per-character breakdown for Korean; whole token for Latin/English
            kor_count = sum(1 for ch in token if 0xAC00 <= ord(ch) <= 0xD7A3)
            if kor_count > 0:
                for ch in token:
                    if 0xAC00 <= ord(ch) <= 0xD7A3:
                        flat_syllables.append(ch)
            else:
                # distribute token across its syllable count
                n = _count_syllables_english_latin(token)
                # the whole token on first position, rest as extension
                flat_syllables.append(token)
                flat_syllables.extend([""] * (n - 1))
        # pad or trim to match syllable_count
        if len(flat_syllables) < self.syllable_count:
            flat_syllables += [""] * (self.syllable_count - len(flat_syllables))
        flat_syllables = flat_syllables[: self.syllable_count]

        for v in range(4):
            notes = []
            for k in range(len(self.midi_grid[v])):
                midi = self.midi_grid[v][k]
                dur = self.tablet.rhythm[k]
                lyric = flat_syllables[k] if v == 0 else None
                notes.append((midi, dur, lyric))
            mx.add_part(VOICE_NAMES[v], notes)
        mx.save(path)

    def to_rule_log_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.rule_log.to_json())


# ---------------------------------------------------------------------------
# The Arca engine
# ---------------------------------------------------------------------------

@dataclass
class Arca:
    """The five-step algorithmic engine.

    Parameters
    ----------
    mode : str
        Name of a Mensa line registered in the MensaTonographica (default:
        'F_major', matching the source paper's normalization).
    pinax : Pinax
        The source of tablets. Defaults to the reference reconstruction of
        Pinax I, Syntagma I.
    tablet_index : int
        Which tablet to pick when multiple are available for a given
        syllable count (used for deterministic demos / tests). Default 0
        picks the first.
    """
    mode: str = "F_major"
    pinax: Pinax = field(default_factory=lambda: PINAX_I_SYNTAGMA_I)
    mensa_tono: MensaTonographica = field(default_factory=MensaTonographica.default)
    tablet_index: int = 0
    cadence_target: Optional[str] = None   # if set, a mismatch emits a warning

    # -- top-level ------------------------------------------------------------

    def compose(self, text: str) -> CompositionResult:
        """Run the full pipeline on an input text."""
        rule_log = RuleLog()

        # Step 1: Selection
        syll = count_syllables(text)
        tablet = self._step_selection(text, syll, rule_log)

        # Step 2: Conversion
        mensa = self.mensa_tono.get(self.mode)
        pitch_grid, midi_grid = self._step_conversion(tablet, mensa, rule_log)

        # Step 3: Constraint check
        VoiceLeadingChecker().check(midi_grid, rule_log)

        # Step 4: Cadential template
        cadence = CadentialTemplate(target=self.cadence_target or tablet.cadence).apply(
            midi_grid, mensa, rule_log
        )

        # Step 5: Output (the log entry; file writing is caller's choice).
        rule_log.add(
            stage=STAGE_OUTPUT,
            rule="output_ready",
            message=(
                f"Four-voice composition ready: {len(midi_grid[0])} chords across "
                f"{len(VOICE_NAMES)} voices, mode '{self.mode}', cadence {cadence}."
            ),
            data={
                "n_chords": len(midi_grid[0]),
                "mode": self.mode,
                "cadence": cadence,
            },
            status="ok",
        )

        return CompositionResult(
            input_text=text,
            syllable_count=syll,
            tablet=tablet,
            mensa=mensa,
            pitch_grid=pitch_grid,
            midi_grid=midi_grid,
            cadence_detected=cadence,
            rule_log=rule_log,
        )

    # -- individual steps ----------------------------------------------------

    def _step_selection(
        self, text: str, syll: int, rule_log: RuleLog,
    ) -> Tablet:
        rule_log.add(
            stage=STAGE_SELECTION,
            rule="syllable_count",
            message=f"Counted {syll} syllable(s) in the input text.",
            data={"text": text, "syllables": syll},
        )

        available = sorted(self.pinax.tablets_by_syllables)
        # Pick the best matching tablet family: exact match preferred,
        # otherwise the closest lower count, otherwise the closest higher.
        chosen_n = None
        if syll in self.pinax.tablets_by_syllables:
            chosen_n = syll
            rule_log.add(
                stage=STAGE_SELECTION,
                rule="tablet_family_selection",
                message=f"Exact tablet family found for {syll} syllables.",
                data={"chosen_syllable_count": chosen_n, "available": available},
            )
        else:
            lower = [n for n in available if n < syll]
            upper = [n for n in available if n > syll]
            if lower:
                chosen_n = max(lower)
                rule_log.add(
                    stage=STAGE_SELECTION,
                    rule="tablet_family_selection",
                    message=(
                        f"No exact {syll}-syllable tablet; falling back to "
                        f"{chosen_n}-syllable family (closest lower)."
                    ),
                    data={
                        "requested_syllables": syll,
                        "chosen_syllable_count": chosen_n,
                        "available": available,
                    },
                    status="info",
                )
            elif upper:
                chosen_n = min(upper)
                rule_log.add(
                    stage=STAGE_SELECTION,
                    rule="tablet_family_selection",
                    message=(
                        f"No {syll}-syllable tablet; falling back to "
                        f"{chosen_n}-syllable family (closest higher)."
                    ),
                    data={
                        "requested_syllables": syll,
                        "chosen_syllable_count": chosen_n,
                        "available": available,
                    },
                    status="info",
                )
            else:
                raise ValueError(f"Pinax has no tablets; cannot select for text {text!r}")

        candidates = self.pinax.tablets_for(chosen_n)
        idx = min(self.tablet_index, len(candidates) - 1)
        tablet = candidates[idx]
        rule_log.add(
            stage=STAGE_SELECTION,
            rule="tablet_choice",
            message=(
                f"Selected tablet '{tablet.id}' (index {idx} of "
                f"{len(candidates)}; cadence target = {tablet.cadence})."
            ),
            data={
                "tablet_id": tablet.id,
                "tablet_index": idx,
                "tablet_count_in_family": len(candidates),
                "declared_cadence": tablet.cadence,
                "provenance": tablet.provenance,
            },
        )
        return tablet

    def _step_conversion(
        self,
        tablet: Tablet,
        mensa: MensaLine,
        rule_log: RuleLog,
    ) -> Tuple[List[List[str]], List[List[int]]]:
        rule_log.add(
            stage=STAGE_CONVERSION,
            rule="mensa_line_selected",
            message=(
                f"Using Mensa line '{mensa.name}' (finalis={mensa.finalis}). "
                f"Numbers 1 and 8 both resolve to the finalis, preserving the "
                f"Pythagorean 1:2 octave ratio."
            ),
            data={
                "mensa_name": mensa.name,
                "finalis": mensa.finalis,
                "degree_map": mensa.degree_to_pitch,
            },
        )

        # Build pitch name grid.
        pitch_grid: List[List[str]] = []
        for v, row in enumerate(tablet.voices):
            pitch_grid.append([mensa.pitch_for(n) for n in row])

        # Build MIDI grid with strict top-down voice ordering per chord,
        # and smooth voice-leading from chord to chord (each voice's next
        # pitch is placed to minimize motion from its previous pitch).
        n_chords = len(tablet.voices[0])
        midi_grid = [[0] * n_chords for _ in range(4)]
        prev_chord: Optional[List[int]] = None
        for k in range(n_chords):
            pitches = [pitch_grid[v][k] for v in range(4)]
            prefer_upper = [tablet.voices[v][k] == 8 for v in range(4)]
            chord_midis = place_chord(pitches, prefer_upper, prev_chord_midis=prev_chord)
            for v in range(4):
                midi_grid[v][k] = chord_midis[v]
            prev_chord = chord_midis

        # Emit a per-chord log entry for human-readable traceability.
        n_chords = len(tablet.voices[0])
        for k in range(n_chords):
            nums = [tablet.voices[v][k] for v in range(4)]
            pcs = [pitch_grid[v][k] for v in range(4)]
            midis = [midi_grid[v][k] for v in range(4)]
            rule_log.add(
                stage=STAGE_CONVERSION,
                rule="number_to_pitch",
                message=(
                    f"Chord {k + 1}: Kircher numbers {nums} -> pitches {pcs} "
                    f"(MIDI {midis})."
                ),
                data={
                    "chord_index": k,
                    "numbers": nums,
                    "pitches": pcs,
                    "midi": midis,
                },
            )

        return pitch_grid, midi_grid
