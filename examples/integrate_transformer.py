"""
examples/integrate_transformer.py
---------------------------------

Runnable path for plugging a symbolic Music Transformer checkpoint
(e.g. a REMI-tokenized HF model) into the Kircher hybrid pipeline.

Dependencies (installed on *your* machine, not in the repo sandbox)::

    pip install transformers miditok torch

Usage::

    python3 examples/integrate_transformer.py \
        --checkpoint amaai-lab/muse-tiny-remi \
        --n 10 --chord-len 6 --seed 42

What this script demonstrates
-----------------------------

1. Load a symbolic transformer from Hugging Face.
2. Wrap it as a `DegreeGridSampler` using `MusicTransformerAdapter`.
3. Run both the pure-transformer and hybrid pipelines.
4. Report the same evaluation table as `hybrid_demo.py`, so the paper's
   main result — *the projector reduces violations on a real
   checkpoint* — becomes directly reproducible.

If `transformers` or `miditok` is missing, `load_hf_music_transformer`
raises a clear ImportError. In that case either install them or use
`examples/integrate_reference.py` for the sandbox-friendly n-gram path.
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
    ap.add_argument("--checkpoint", required=True,
                    help="Hugging Face checkpoint id or local path")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--chord-len", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="output/transformer_eval.json")
    args = ap.parse_args()

    # Lazy import so the script still imports cleanly in the sandbox
    # if the user wants to read the code without running it.
    from arca.integrations import load_hf_music_transformer

    adapter = load_hf_music_transformer(
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
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
        # metadata from the adapter (checkpoint id) lands in the log
        log_p.add(stage="selection", rule="checkpoint_metadata",
                  message="External checkpoint metadata.",
                  data=adapter.metadata())
        log_h.add(stage="selection", rule="checkpoint_metadata",
                  message="External checkpoint metadata.",
                  data=adapter.metadata())
        pure_items.append(EvaluableComposition(
            midi_grid=midi_p, rule_log=log_p, label="transformer_only"))
        hyb_items.append(EvaluableComposition(
            midi_grid=midi_h, rule_log=log_h, label="hybrid"))

    pure_metrics = ev.evaluate_corpus(pure_items)
    hyb_metrics = ev.evaluate_corpus(hyb_items)
    pure_summary = ev.summarize(pure_metrics)
    hyb_summary = ev.summarize(hyb_metrics)

    print("\n=== Symbolic Music Transformer integration — N =", args.n, "===")
    print(f"Checkpoint: {args.checkpoint}")
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
