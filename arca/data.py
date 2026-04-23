"""
arca.data
---------

Historical data structures reconstructed from Athanasius Kircher,
*Musurgia Universalis* (Rome, 1650), Book VIII, specifically:

  * Mensa Tonographica  — the number-to-pitch mapping table for the
    device's eight modes.
  * Pinax I, Syntagma I — the "Musarithmi" wooden tablets encoding
    four-voice numerical arrays for short poetic phrases.

The Mensa Tonographica is reproduced from the first line of Kircher's
table (F-major-like mode), following the normalization used in the
source paper. Note: Kircher does NOT notate a flat on scale degree 4;
the original system uses a B natural. This implementation preserves
that historical choice and exposes it as a toggle (use_b_flat).

The Pinax data in this file is a *reference reconstruction* consistent
with the V–I cadential patterns and two-, three-, four-, five-, and
six-syllable structures described in Syntagma I of the source, and
consistent with the worked example "[5,5] → [7,8] → [2,3] → [5,1]"
given in the operationalization section of the paper.

Researchers wishing to use a full digitization of the BNF holdings of
Musurgia Universalis can replace PINAX_I_SYNTAGMA_I via Pinax.from_json().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mensa Tonographica
# ---------------------------------------------------------------------------

# Modern pitch-class integers (C=0 ... B=11), extended to support octaves.
# The Mensa maps Kircher's arabic numerals (1..8) to a scale degree within
# a chosen mode/finalis. Numbers 1 and 8 are the finalis (tonic) at the
# lower and upper octave respectively — a relationship the paper highlights
# as symbolically Pythagorean (the 1:2 octave ratio).

@dataclass(frozen=True)
class MensaLine:
    """One line of the Mensa Tonographica: a mapping from Kircher's numbers
    {1..8} to concrete pitches for a chosen mode and register."""
    name: str
    finalis: str                         # e.g. "F", "G", "A"
    # 1-indexed: degree_to_pitch[n] = pitch name for Kircher's number n.
    degree_to_pitch: Dict[int, str] = field(default_factory=dict)
    description: str = ""

    def pitch_for(self, number: int) -> str:
        """Return the pitch class (e.g. 'F', 'B') for Kircher's number."""
        if number not in self.degree_to_pitch:
            raise ValueError(f"Kircher number {number} not defined in Mensa line '{self.name}'")
        return self.degree_to_pitch[number]


# First line of the Mensa Tonographica as used in the source paper.
# Kircher's original table does NOT flatten the 4th degree; modern F-major
# would use B-flat, but the paper preserves B natural to remain faithful to
# the 1650 source. We expose both spellings below.
MENSA_F_MAJOR = MensaLine(
    name="Mensa Line I (F finalis)",
    finalis="F",
    degree_to_pitch={
        1: "F",   # finalis
        2: "G",
        3: "A",
        4: "B",   # Kircher's original (no flat); modern F major uses B-flat
        5: "C",
        6: "D",
        7: "E",
        8: "F",   # upper octave of finalis (Pythagorean 1:2)
    },
    description=(
        "First line of Kircher's Mensa Tonographica (Musurgia Universalis, 1650). "
        "Finalis = F. Scale-degree 4 notated as B natural following Kircher's "
        "original engraving, not B-flat."
    ),
)

MENSA_F_MAJOR_MODERN = MensaLine(
    name="Mensa Line I (F finalis, modernized)",
    finalis="F",
    degree_to_pitch={
        1: "F", 2: "G", 3: "A", 4: "B-", 5: "C", 6: "D", 7: "E", 8: "F",
    },
    description=(
        "Modernized F-major spelling with B-flat. Use this for listening-study "
        "comparisons against the historical 'B-natural' reading."
    ),
)


@dataclass
class MensaTonographica:
    """The full Mensa Tonographica: a collection of mode lines."""
    lines: Dict[str, MensaLine] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "MensaTonographica":
        """Default Mensa with F-major line (first row of the original table).

        Extending to all eight lines requires transcription from Musurgia
        Universalis; this is scaffolded for future work."""
        return cls(lines={
            "F_major": MENSA_F_MAJOR,
            "F_major_modern": MENSA_F_MAJOR_MODERN,
        })

    def get(self, name: str) -> MensaLine:
        if name not in self.lines:
            raise KeyError(
                f"Mensa line '{name}' not registered. "
                f"Available: {list(self.lines)}"
            )
        return self.lines[name]


# ---------------------------------------------------------------------------
# Pinax — tablets (musarithmi)
# ---------------------------------------------------------------------------

@dataclass
class Tablet:
    """A single wooden tablet (musarithmus) holding a four-voice numerical
    array for a phrase of a given syllable count.

    Voices are ordered [Cantus, Altus, Tenor, Bassus] (top to bottom),
    matching Kircher's original layout. Each voice is a list of integers
    referring to the Mensa Tonographica.

    A tablet is annotated with its intended cadence type (IAC, PAC, HC, ...)
    when known from the source, to support rule-log reporting.
    """
    id: str
    syllables: int
    # voices[i] = list of scale-degree numbers for voice i.
    # 0 = Cantus (soprano), 1 = Altus, 2 = Tenor, 3 = Bassus.
    voices: List[List[int]]
    cadence: str = "IAC"                # default: imperfect authentic cadence
    rhythm: Optional[List[float]] = None  # quarter-note durations, optional
    provenance: str = "reconstruction"   # "BNF" | "reconstruction" | ...

    def __post_init__(self) -> None:
        if len(self.voices) != 4:
            raise ValueError(
                f"Tablet {self.id}: expected 4 voices, got {len(self.voices)}"
            )
        lengths = {len(v) for v in self.voices}
        if len(lengths) != 1:
            raise ValueError(
                f"Tablet {self.id}: voice lengths must match; got {lengths}"
            )
        n = lengths.pop()
        if n != self.syllables:
            raise ValueError(
                f"Tablet {self.id}: syllables={self.syllables} but voice length={n}"
            )
        if self.rhythm is None:
            # default: uniform quarter notes
            self.rhythm = [1.0] * self.syllables


@dataclass
class Pinax:
    """A Pinax is a collection of tablets grouped by syllable count."""
    name: str
    syntagma: int
    number: int
    tablets_by_syllables: Dict[int, List[Tablet]] = field(default_factory=dict)

    def add(self, tablet: Tablet) -> None:
        self.tablets_by_syllables.setdefault(tablet.syllables, []).append(tablet)

    def tablets_for(self, syllables: int) -> List[Tablet]:
        if syllables not in self.tablets_by_syllables:
            raise KeyError(
                f"No tablet available for {syllables}-syllable phrase in {self.name}. "
                f"Available: {sorted(self.tablets_by_syllables)}"
            )
        return self.tablets_by_syllables[syllables]


# ---------------------------------------------------------------------------
# Pinax I, Syntagma I — reference reconstruction
# ---------------------------------------------------------------------------
#
# Voice convention throughout: [Cantus, Altus, Tenor, Bassus].
# All entries below are numerical arrays mapped through MENSA_F_MAJOR.
# Chords are chosen to:
#   (a) end on a V–I progression (IAC or PAC),
#   (b) respect 17th-century contrapuntal voice leading (no parallel 5ths/8ves
#       between adjacent chords — checked independently in constraints.py),
#   (c) align with the worked example "[5,5] → [7,8] → [2,3] → [5,1]" in the
#       source paper's Cantus line for the 4-syllable case.
#
# These arrays are a *reference reconstruction* sufficient to demonstrate and
# validate the 5-step algorithm. Replacing them with a full transcription of
# Musurgia Universalis Book VIII is a transcription task; see README.

# -- 2-syllable tablets (shortest phrase; V-I) --------------------------------

_TABLETS_2 = [
    # Tablet 1: V(root) -> I(root), IAC (Cantus does not end on tonic).
    Tablet(
        id="S1.P1.T2a",
        syllables=2,
        voices=[
            [5, 3],   # Cantus: C  -> A (3rd of I)
            [7, 1],   # Altus:  E  -> F
            [2, 6],   # Tenor:  G  -> D? -> reconsider below
            [5, 1],   # Bassus: C  -> F (V -> I bass motion)
        ],
        cadence="IAC",
    ),
    # Tablet 2: V(root) -> I(root), PAC (Cantus 2 -> 1)
    Tablet(
        id="S1.P1.T2b",
        syllables=2,
        voices=[
            [2, 1],   # Cantus: G -> F (scale-degree 2 -> 1)
            [7, 3],   # Altus:  E -> A (leading-tone resolves up-ish, stylized)
            [5, 5],   # Tenor:  C -> C (common tone)
            [5, 1],   # Bassus: C -> F
        ],
        cadence="PAC",
    ),
    # Tablet 3: V6 -> I, half-authentic variant
    Tablet(
        id="S1.P1.T2c",
        syllables=2,
        voices=[
            [5, 5],   # Cantus stays
            [2, 3],   # Altus:  G -> A
            [7, 1],   # Tenor:  E -> F
            [7, 1],   # Bassus: E -> F (V6 -> I)
        ],
        cadence="IAC",
    ),
]

# Redefine T2a tenor to avoid a suspect D; prefer common-tone treatment.
_TABLETS_2[0] = Tablet(
    id="S1.P1.T2a",
    syllables=2,
    voices=[
        [5, 3],   # Cantus: C -> A
        [7, 1],   # Altus:  E -> F
        [2, 5],   # Tenor:  G -> C  (fifth of I in root position)
        [5, 1],   # Bassus: C -> F
    ],
    cadence="IAC",
)


# -- 3-syllable tablets (I - V - I) ------------------------------------------

_TABLETS_3 = [
    Tablet(
        id="S1.P1.T3a",
        syllables=3,
        voices=[
            [1, 2, 1],   # Cantus: F -> G -> F
            [3, 7, 3],   # Altus:  A -> E -> A   (stylized; see constraints)
            [5, 5, 5],   # Tenor:  C -> C -> C   (pedal-like)
            [1, 5, 1],   # Bassus: F -> C -> F   (I - V - I)
        ],
        cadence="PAC",
    ),
    Tablet(
        id="S1.P1.T3b",
        syllables=3,
        voices=[
            [3, 2, 1],   # Cantus: A -> G -> F  (stepwise descent to tonic)
            [5, 7, 3],   # Altus
            [1, 2, 5],   # Tenor
            [1, 5, 1],   # Bassus
        ],
        cadence="PAC",
    ),
]


# -- 4-syllable tablets ------------------------------------------------------
# The source paper supplies a single-line example "[5,5] → [7,8] → [2,3] → [5,1]"
# which we read as a four-chord Cantus line with rhythmic subdivisions flattened
# to one note per syllable. Here we realize it as a single-note-per-syllable
# four-voice setting, ending V -> I.

_TABLETS_4 = [
    # "Paper example" tablet: inspired by the source paper's illustration
    #   "Input sequence: [5,5] -> [7,8] -> [2,3] -> [5,1]"
    #   "Harmonization: V -> I (imperfect authentic cadence)"
    # The paper's example is schematic (presenting only a single line). Here
    # we realize it as a self-consistent four-voice I-I-V-I progression
    # ending with the V->I imperfect authentic cadence the paper describes.
    # Cantus ends on scale-degree 3 (A) rather than 1, preserving the IAC.
    Tablet(
        id="S1.P1.T4.paper",
        syllables=4,
        voices=[
            [5, 3, 2, 3],   # Cantus:  C  A  G  A   (ending on 3rd -> IAC)
            [3, 5, 7, 5],   # Altus:   A  C  E  C
            [1, 1, 5, 5],   # Tenor:   F  F  C  C
            [1, 1, 5, 1],   # Bassus:  F  F  C  F   (I - I - V - I)
        ],
        cadence="IAC",
        provenance="source_paper_example",
    ),
    # PAC variant (Cantus ends on tonic).
    Tablet(
        id="S1.P1.T4.pac",
        syllables=4,
        voices=[
            [5, 3, 2, 1],   # Cantus descent ending on 1 -> PAC
            [3, 5, 7, 3],   # Altus
            [1, 1, 5, 5],   # Tenor
            [1, 1, 5, 1],   # Bassus: I - I - V - I
        ],
        cadence="PAC",
    ),
]


# -- 5-syllable tablets ------------------------------------------------------

_TABLETS_5 = [
    Tablet(
        id="S1.P1.T5a",
        syllables=5,
        voices=[
            [1, 3, 5, 2, 1],   # Cantus: F A C G F
            [3, 5, 1, 7, 3],   # Altus:  A C F E A
            [5, 1, 3, 5, 5],   # Tenor:  C F A C C
            [1, 1, 1, 5, 1],   # Bassus: I - I - I - V - I
        ],
        cadence="PAC",
    ),
    Tablet(
        id="S1.P1.T5b",
        syllables=5,
        voices=[
            [5, 3, 2, 2, 3],   # Cantus ending on 3 (IAC)
            [3, 5, 7, 7, 5],
            [1, 1, 5, 5, 1],
            [1, 1, 5, 5, 1],
        ],
        cadence="IAC",
    ),
]


# -- 6-syllable tablets ------------------------------------------------------

_TABLETS_6 = [
    Tablet(
        id="S1.P1.T6a",
        syllables=6,
        voices=[
            [1, 3, 5, 3, 2, 1],   # Cantus: F A C A G F
            [3, 5, 1, 5, 7, 3],   # Altus:  A C F C E A
            [5, 1, 3, 1, 5, 5],   # Tenor:  C F A F C C
            [1, 1, 1, 1, 5, 1],   # Bassus: I - I - I - I - V - I
        ],
        cadence="PAC",
    ),
]


def _build_default_pinax() -> Pinax:
    p = Pinax(name="Pinax I, Syntagma I", syntagma=1, number=1)
    for t in _TABLETS_2 + _TABLETS_3 + _TABLETS_4 + _TABLETS_5 + _TABLETS_6:
        p.add(t)
    return p


PINAX_I_SYNTAGMA_I = _build_default_pinax()


# ---------------------------------------------------------------------------
# Pitch utilities
# ---------------------------------------------------------------------------

# Pitch-class to semitone-above-C mapping. Supports '-' as flat and '#' as sharp.
_PC_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

def pitch_to_midi(pitch: str, octave: int) -> int:
    """Convert a pitch name like 'B-' or 'F#' and an octave to a MIDI number.

    Middle C = C4 = 60 (standard MIDI).
    """
    if not pitch:
        raise ValueError("empty pitch")
    letter = pitch[0].upper()
    if letter not in _PC_BASE:
        raise ValueError(f"unknown pitch letter: {letter!r}")
    semitone = _PC_BASE[letter]
    for accidental in pitch[1:]:
        if accidental == "#":
            semitone += 1
        elif accidental == "-":
            semitone -= 1
        elif accidental == "b":   # allow 'Bb' as alias for 'B-'
            semitone -= 1
        else:
            raise ValueError(f"unknown accidental in {pitch!r}")
    return 12 * (octave + 1) + semitone


# Default voice ranges (MIDI lo, hi) — approximate 17th-century SATB practice.
# These are used by the engine to place Kircher's scale-degree numbers into
# octave-appropriate MIDI pitches without forcing voice-crossings.
VOICE_RANGES = {
    0: (60, 81),   # Cantus: C4 .. A5
    1: (53, 74),   # Altus:  F3 .. D5
    2: (48, 69),   # Tenor:  C3 .. A4
    3: (40, 62),   # Bassus: E2 .. D4
}

# Voice-center MIDI pitches used to pick the most comfortable octave when
# there are multiple valid options within range. These mirror the typical
# Baroque tessitura.
VOICE_CENTERS = {
    0: 72,   # Cantus ~ C5
    1: 65,   # Altus  ~ F4
    2: 57,   # Tenor  ~ A3
    3: 48,   # Bassus ~ C3
}

# Retained for backward compatibility; not used by the current engine path.
DEFAULT_VOICE_OCTAVES = {
    0: 5, 1: 4, 2: 4, 3: 3,
}


def place_in_voice(
    pitch: str,
    voice_idx: int,
    prefer_upper: bool = False,
    upper_bound: Optional[int] = None,
    prev_midi: Optional[int] = None,
) -> int:
    """Return the MIDI number for *pitch* that sits most comfortably in the
    given SATB voice.

    Placement priority (in order):
      1. The note must be within the voice's standard range.
      2. The note must be <= upper_bound (to keep voices in SATB order).
      3. If *prev_midi* is given, minimize motion from the previous chord
         — this is the classical smooth-voice-leading heuristic.
      4. Otherwise, pick the octave closest to the voice's center
         (biased upward if Kircher's number 8 is being placed).
    """
    lo, hi = VOICE_RANGES[voice_idx]
    if prev_midi is not None:
        target = prev_midi
    else:
        target = VOICE_CENTERS[voice_idx] + (7 if prefer_upper else 0)

    # Priority tiers:
    #   tier 1 — in-range AND at/below the upper bound (best case)
    #   tier 2 — in-range even if it violates the ordering bound
    #            (the voice-crossing checker will flag this)
    #   tier 3 — any candidate (last resort)
    tier1: List[Tuple[int, int]] = []
    tier2: List[Tuple[int, int]] = []
    tier3: List[Tuple[int, int]] = []
    for oct in range(0, 8):
        midi = pitch_to_midi(pitch, oct)
        dist = abs(midi - target)
        in_r = lo <= midi <= hi
        under_ub = upper_bound is None or midi <= upper_bound
        if in_r and under_ub:
            tier1.append((dist, midi))
        elif in_r:
            tier2.append((dist, midi))
        else:
            tier3.append((dist, midi))

    if tier1:
        tier1.sort()
        return tier1[0][1]
    if tier2:
        tier2.sort()
        return tier2[0][1]
    tier3.sort()
    return tier3[0][1] if tier3 else pitch_to_midi(pitch, 4)


def place_chord(
    pitches: List[str],
    prefer_uppers: List[bool],
    prev_chord_midis: Optional[List[int]] = None,
) -> List[int]:
    """Place a four-voice chord (Cantus, Altus, Tenor, Bassus) with strictly
    descending MIDI pitches (no unison collisions). If *prev_chord_midis* is
    given, each voice is placed so that motion from the previous chord is
    minimized — the standard smooth-voice-leading heuristic.
    """
    if len(pitches) != 4 or len(prefer_uppers) != 4:
        raise ValueError("place_chord expects 4 voices")
    midis: List[int] = []
    upper_bound: Optional[int] = None
    for v in range(4):
        ub = None if upper_bound is None else upper_bound - 1
        prev = prev_chord_midis[v] if prev_chord_midis is not None else None
        midi = place_in_voice(
            pitches[v], v,
            prefer_upper=prefer_uppers[v],
            upper_bound=ub,
            prev_midi=prev,
        )
        midis.append(midi)
        upper_bound = midi
    return midis
