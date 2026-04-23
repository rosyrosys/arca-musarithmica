"""
arca.output
-----------

Zero-dependency writers for two standard music file formats:

  * Standard MIDI File (SMF) format 1, via raw byte serialization.
  * MusicXML 3.1 partwise, via plain XML string generation.

Keeping these dependency-free matters for the research prototype:
reviewers can reproduce outputs without a specific library version.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# MIDI writer
# ---------------------------------------------------------------------------

def _write_vlq(n: int) -> bytes:
    """MIDI variable-length quantity encoding."""
    if n < 0:
        raise ValueError("VLQ must be non-negative")
    out = [n & 0x7F]
    n >>= 7
    while n > 0:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


@dataclass
class MidiNote:
    """A MIDI note event with absolute timing in ticks."""
    pitch: int       # 0..127
    start: int       # ticks
    duration: int    # ticks
    velocity: int = 80
    channel: int = 0


class MidiWriter:
    """Minimal Standard MIDI File format-1 writer.

    Usage:
        mw = MidiWriter(ticks_per_quarter=480)
        mw.add_track([MidiNote(...), ...], name="Cantus")
        mw.save("out.mid")
    """

    def __init__(self, ticks_per_quarter: int = 480, tempo_bpm: float = 90.0):
        self.tpq = ticks_per_quarter
        self.tempo_bpm = tempo_bpm
        self.tracks: List[Tuple[str, List[MidiNote]]] = []

    def add_track(self, notes: Sequence[MidiNote], name: str = "Voice") -> None:
        self.tracks.append((name, list(notes)))

    # -- serialization -------------------------------------------------------

    def _build_track_chunk(
        self, notes: Sequence[MidiNote], name: str, is_first: bool
    ) -> bytes:
        data = bytearray()

        # Track name meta event
        name_bytes = name.encode("utf-8")
        data += _write_vlq(0) + b"\xff\x03" + _write_vlq(len(name_bytes)) + name_bytes

        # Tempo meta event on first track
        if is_first:
            microseconds_per_quarter = int(60_000_000 / self.tempo_bpm)
            data += _write_vlq(0) + b"\xff\x51\x03" + \
                    microseconds_per_quarter.to_bytes(3, "big")

        # Build all note_on / note_off events with absolute ticks
        events: List[Tuple[int, int, bytes]] = []
        # tie-break: note_off before note_on at the same tick to avoid
        # the same-pitch note collapsing.
        for n in notes:
            on = bytes([0x90 | (n.channel & 0x0F), n.pitch & 0x7F, n.velocity & 0x7F])
            off = bytes([0x80 | (n.channel & 0x0F), n.pitch & 0x7F, 0])
            events.append((n.start, 1, on))
            events.append((n.start + n.duration, 0, off))
        events.sort()

        last_tick = 0
        for tick, _, raw in events:
            delta = tick - last_tick
            data += _write_vlq(delta) + raw
            last_tick = tick

        # End of track
        data += _write_vlq(0) + b"\xff\x2f\x00"

        return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            # Header
            n_tracks = len(self.tracks)
            f.write(b"MThd")
            f.write(struct.pack(">IHHH", 6, 1, n_tracks, self.tpq))
            for i, (name, notes) in enumerate(self.tracks):
                f.write(self._build_track_chunk(notes, name, is_first=(i == 0)))


# ---------------------------------------------------------------------------
# MusicXML writer
# ---------------------------------------------------------------------------

# Map pitch letter to MusicXML step / alter.
_LETTER_TO_STEP = {"C": "C", "D": "D", "E": "E", "F": "F",
                   "G": "G", "A": "A", "B": "B"}


def _pitch_to_step_alter_octave(midi: int) -> Tuple[str, int, int]:
    """Convert a MIDI number to (step, alter, octave) for MusicXML."""
    # Prefer sharp spelling for simplicity except for B-flat (common in F major).
    # For a prototype this is acceptable; rigorous spelling is deferred to
    # notation-layer enhancements.
    octave = midi // 12 - 1
    semitone = midi % 12
    # Use flat spelling for pc 10 so that Bb reads as "B flat" not "A#"
    sharp_map = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
                 ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("B", -1), ("B", 0)]
    step, alter = sharp_map[semitone]
    return step, alter, octave


class MusicXmlWriter:
    """Minimal MusicXML 3.1 Partwise score writer for 4-voice compositions."""

    def __init__(self, title: str = "Arca Musarithmica",
                 composer: str = "Kircher / Arca Engine"):
        self.title = title
        self.composer = composer
        # parts[i] = list of (midi, duration_quarter, lyric_or_None)
        self.parts: List[Tuple[str, List[Tuple[int, float, Optional[str]]]]] = []

    def add_part(self, name: str,
                 notes: Sequence[Tuple[int, float, Optional[str]]]) -> None:
        self.parts.append((name, list(notes)))

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _escape(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))

    def _note_xml(self, midi: int, duration_q: float,
                  lyric: Optional[str], divisions: int) -> str:
        step, alter, octave = _pitch_to_step_alter_octave(midi)
        dur = int(round(duration_q * divisions))
        # pick a simple type
        type_map = {1: "whole", 2: "half", 3: "half", 4: "quarter",
                    6: "quarter", 8: "eighth", 12: "eighth", 16: "16th"}
        note_type = "quarter"
        for d, t in sorted(type_map.items()):
            if dur <= d:
                note_type = t
                break
        alter_xml = f"<alter>{alter}</alter>" if alter else ""
        lyric_xml = (
            f"<lyric><text>{self._escape(lyric)}</text></lyric>"
            if lyric else ""
        )
        return (
            f"<note>"
            f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
            f"<duration>{dur}</duration><type>{note_type}</type>{lyric_xml}"
            f"</note>"
        )

    def to_string(self) -> str:
        divisions = 4  # quarter-note = 4 divisions
        parts_xml = []
        part_list_xml = []

        for i, (name, notes) in enumerate(self.parts):
            pid = f"P{i+1}"
            part_list_xml.append(
                f'<score-part id="{pid}"><part-name>{self._escape(name)}</part-name></score-part>'
            )
            measure_notes = []
            # Single measure per composition for simplicity of v0.1
            measure_notes.append(
                "<attributes>"
                f"<divisions>{divisions}</divisions>"
                "<key><fifths>-1</fifths><mode>major</mode></key>"   # F major
                "<time><beats>4</beats><beat-type>4</beat-type></time>"
                "<clef><sign>G</sign><line>2</line></clef>"
                "</attributes>"
            )
            for midi, dur, lyric in notes:
                measure_notes.append(self._note_xml(midi, dur, lyric, divisions))
            parts_xml.append(
                f'<part id="{pid}"><measure number="1">'
                + "".join(measure_notes)
                + "</measure></part>"
            )

        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
            '<!DOCTYPE score-partwise PUBLIC '
            '"-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">'
            '<score-partwise version="3.1">'
            f'<work><work-title>{self._escape(self.title)}</work-title></work>'
            f'<identification><creator type="composer">{self._escape(self.composer)}</creator></identification>'
            f'<part-list>{"".join(part_list_xml)}</part-list>'
            + "".join(parts_xml) +
            '</score-partwise>'
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_string())
