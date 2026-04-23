"""
arca.__main__
-------------

Command-line entry point: `python3 -m arca "<text>" [options]`.

Rationale: the CLI gives reviewers a zero-install, one-line path from a text
input to (a) a four-voice score, (b) a MIDI and MusicXML file, and (c) a
structured rule log. It is the simplest possible front-end to the paper's
five-step pipeline and makes the reproducibility claim auditable from a
shell prompt.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .engine import Arca, CompositionResult


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arca",
        description=(
            "Arca Musarithmica — an interpretable, rule-logged reconstruction "
            "of Kircher's five-step composition algorithm. Converts input "
            "text to a four-voice composition, writing MIDI, MusicXML and a "
            "JSON rule log."
        ),
    )
    p.add_argument(
        "text",
        nargs="?",
        help="Input text (syllable-counted). If omitted, read from stdin.",
    )
    p.add_argument(
        "-m", "--mode",
        default="F_major",
        help="Mensa line / mode name (default: F_major).",
    )
    p.add_argument(
        "-t", "--tablet-index",
        type=int,
        default=0,
        help="Which tablet to pick when several match the syllable count "
             "(default: 0 — first).",
    )
    p.add_argument(
        "-c", "--cadence",
        default=None,
        help="Optional target cadence (PAC|IAC|HC|PC|DC). If set, the engine "
             "flags mismatches in the rule log.",
    )
    p.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to write output files (default: ./output).",
    )
    p.add_argument(
        "-n", "--name",
        default="arca",
        help="Base filename stem for outputs (default: 'arca').",
    )
    p.add_argument(
        "--no-files",
        action="store_true",
        help="Do not write output files; print log only.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print the full rule log as JSON to stdout.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable rule log.",
    )
    return p


def _read_text(cli_text: Optional[str]) -> str:
    if cli_text and cli_text.strip():
        return cli_text
    if sys.stdin.isatty():
        print(
            "Enter text (end with Ctrl-D / Ctrl-Z):",
            file=sys.stderr,
        )
    data = sys.stdin.read().strip()
    if not data:
        raise SystemExit("arca: no input text provided.")
    return data


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    text = _read_text(args.text)

    arca = Arca(
        mode=args.mode,
        tablet_index=args.tablet_index,
        cadence_target=args.cadence,
    )
    result: CompositionResult = arca.compose(text)

    if not args.quiet:
        print(result.pretty())

    if args.json:
        print(result.rule_log.to_json())

    if not args.no_files:
        os.makedirs(args.output_dir, exist_ok=True)
        base = os.path.join(args.output_dir, args.name)
        result.to_midi(base + ".mid")
        result.to_musicxml(base + ".musicxml")
        result.to_rule_log_json(base + ".rulelog.json")
        print(
            f"\nWrote: {base}.mid, {base}.musicxml, {base}.rulelog.json",
            file=sys.stderr,
        )

    # Exit code: 0 if no warnings, 1 if warnings present (useful for CI).
    return 1 if result.rule_log.warnings() else 0


if __name__ == "__main__":
    raise SystemExit(main())
