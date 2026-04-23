"""
examples/integrate_musicgen.py
------------------------------

Runnable path for plugging Meta's MusicGen (audio transformer) into the
Kircher hybrid pipeline. This is the paper's strongest generalisation
claim: the rule projector is applicable even to *audio* generators,
once a transcription stage converts the waveform to MIDI.

Dependencies (installed on *your* machine, not in the repo sandbox)::

    pip install audiocraft basic-pitch soundfile torch

A GPU is strongly recommended; audiocraft's MusicGen runs poorly on CPU.

Usage::

    python3 examples/integrate_musicgen.py \
        --checkpoint facebook/musicgen-small \
        --prompt "renaissance polyphony, SATB choir" \
        --n 5 --chord-len 6 --seed 42

Pipeline
--------

    text prompt
        -> audiocraft MusicGen    -> waveform
        -> basic-pitch            -> notes
        -> polyphony_to_satb      -> 4xN MIDI grid
        -> midi_to_degree_grid    -> Kircher-number grid
        -> RuleProjector          -> corrected degree grid
        -> realize_degree_grid    -> final 4xN MIDI grid
        -> Evaluator              -> metrics table

Each step contributes a `rule_log` entry, so the complete provenance
chain — including the audio checkpoint id and the text prompt — is
preserved and dumpable to JSON.

If `audiocraft` or `basic-pitch` is missing, `load_musicgen` raises a
clear ImportError. Use `examples/integrate_reference.py` for an
in-sandbox demonstration of the same adapter interface.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arca.evaluate import EvaluableComposition, Evaluator
from arca.hybrid import HybridSampler, RuleProjector


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="facebook/musicgen-small")
    ap.add_argument("--prompt", default="renaissance polyphony, SATB choir")
    ap.add_argument("--duration", type=float, default=8.0,
                    help="MusicGen audio generation length in seconds")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--chord-len", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/musicgen_eval.json")
    args = ap.parse_args()

    # Lazy import — `arca.integrations.musicgen` can be imported on any
    # machine, but `load_musicgen` is the thing that pulls in audiocraft.
    from arca.integrations import load_musicgen

    adapter = load_musicgen(
        checkpoint=args.checkpoint,
        text_prompt=args.prompt,
        duration_seconds=args.duration,
    )

    pure = HybridSampler(sampler=adapter, use_projector=False)
    hyb  = HybridSampler(sampler=adapter,
                         projector=RuleProjector(),
                         use_projector=True)

    ev = Evaluator()
    pure_items: list = []
    hyb_items: list = []
    for i in range(args.n):
        seed = args.seed + i
        midi_p, log_p = pure.compose(args.chord_len, seed=seed)
        midi_h, log_h = hyb.compose(args.chord_len, seed=seed)
        log_p.add(stage="selection", rule="checkpoint_metadata",
                  message="External checkpoint metadata.",
                  data=adapter.metadata())
        log_h.add(stage="selection", rule="checkpoint_metadata",
                  message="External checkpoint metadata.",
                  data=adapter.metadata())
        pure_items.append(EvaluableComposition(
            midi_grid=midi_p, rule_log=log_p, label="musicgen_only"))
        hyb_items.append(EvaluableComposition(
            midi_grid=midi_h, rule_log=log_h, label="hybrid"))

    pure_metrics = ev.evaluate_corpus(pure_items)
    hyb_metrics = ev.evaluate_corpus(hyb_items)
    pure_summary = ev.summarize(pure_metrics)
    hyb_summary = ev.summarize(hyb_metrics)

    print("\n=== MusicGen (audio) integration — N =", args.n, "===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(ev.comparison_table([pure_summary, hyb_summary]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(ev.to_json(
            [pure_summary, hyb_summary],
            per_item=pure_metrics + hyb_metrics,
        ))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
