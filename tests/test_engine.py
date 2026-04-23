"""
End-to-end tests for the Arca engine.

These are written against the paper's acceptance conditions:

  (a) the engine produces a full four-voice grid for any non-empty text,
  (b) it runs without raising across Latin / English / Korean,
  (c) voice-range and voice-crossing checks fire as expected on known
      clean tablets,
  (d) cadence detection labels the final chord pair,
  (e) the rule log covers all five paper stages on any successful run.

Run with:
    python3 -m pytest -q
"""

from __future__ import annotations

import os
import sys
import tempfile

# Allow running without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arca import Arca, CompositionResult  # noqa: E402
from arca.rulelog import (  # noqa: E402
    STAGE_SELECTION,
    STAGE_CONVERSION,
    STAGE_CONSTRAINT,
    STAGE_CADENCE,
    STAGE_OUTPUT,
)


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------

def test_latin_input_produces_full_grid():
    result = Arca().compose("Cantate Domino canticum novum")
    assert isinstance(result, CompositionResult)
    assert len(result.midi_grid) == 4
    n = len(result.midi_grid[0])
    for row in result.midi_grid:
        assert len(row) == n
    assert result.syllable_count > 0


def test_english_input():
    result = Arca().compose("Hello world")
    assert len(result.midi_grid[0]) > 0


def test_korean_input():
    result = Arca().compose("안녕하세요 코워크")
    assert len(result.midi_grid[0]) > 0
    # 안녕하세요 (5) + 코워크 (3) = 8 Hangul blocks
    assert result.syllable_count == 8


def test_descending_voice_order_per_chord():
    """Cantus >= Altus >= Tenor >= Bassus at every chord (strict preferred)."""
    result = Arca().compose("Ave Maria gratia plena")
    n = len(result.midi_grid[0])
    violations = 0
    for k in range(n):
        pitches = [result.midi_grid[v][k] for v in range(4)]
        for i in range(3):
            if pitches[i] < pitches[i + 1]:
                violations += 1
    # The range tier of place_in_voice is allowed to produce a crossing
    # rather than a range violation; in the showcase tablets this should be 0.
    assert violations == 0


def test_rule_log_covers_all_stages():
    result = Arca().compose("Cantate Domino")
    cov = result.rule_log.coverage()
    for s in (STAGE_SELECTION, STAGE_CONVERSION, STAGE_CONSTRAINT,
              STAGE_CADENCE, STAGE_OUTPUT):
        assert cov[s] >= 1, f"stage {s} had no entries"


def test_cadence_labeled():
    result = Arca().compose("Cantate Domino")
    assert result.cadence_detected in {"PAC", "IAC", "HC", "PC", "DC", "OTHER"}


# ---------------------------------------------------------------------------
# File writers — zero-dep sanity
# ---------------------------------------------------------------------------

def test_midi_and_musicxml_written():
    result = Arca().compose("Cantate Domino")
    with tempfile.TemporaryDirectory() as td:
        mid = os.path.join(td, "out.mid")
        mxl = os.path.join(td, "out.musicxml")
        rl = os.path.join(td, "out.json")
        result.to_midi(mid)
        result.to_musicxml(mxl)
        result.to_rule_log_json(rl)

        # MIDI header magic
        with open(mid, "rb") as f:
            assert f.read(4) == b"MThd"

        # MusicXML must contain partwise root element
        with open(mxl, "r", encoding="utf-8") as f:
            xml = f.read()
        assert "<score-partwise" in xml
        assert "<part " in xml

        # JSON parseable
        import json
        with open(rl, "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert "entries" in payload
        assert "coverage" in payload


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_runs_without_files():
    from arca.__main__ import main
    rc = main(["Cantate Domino", "--no-files", "--quiet"])
    assert rc in (0, 1)  # warnings-present is allowed


def test_cli_writes_files(tmp_path):
    from arca.__main__ import main
    rc = main([
        "Cantate Domino",
        "-o", str(tmp_path),
        "-n", "cli_test",
        "--quiet",
    ])
    assert rc in (0, 1)
    assert (tmp_path / "cli_test.mid").exists()
    assert (tmp_path / "cli_test.musicxml").exists()
    assert (tmp_path / "cli_test.rulelog.json").exists()
