"""
examples/integrate_reference.py
-------------------------------

End-to-end demonstration that the `arca.integrations` interface is
functional: train a trigram chord model on the Kircher Pinax, wrap it
as a `DegreeGridSampler`, run it through the hybrid projector, and
compare against the pure-sampler baseline using the v0.2 evaluator.

Runs in the vanilla sandbox: no pip, no GPU, no external checkpoints.
This is the reproducibility anchor for Section 4 of the follow-up
paper.

Usage
-----

    python3 examples/integrate_reference.py --n 30 --chord-len 6 --seed 42

Expected behaviour
------------------

* The n-gram-only baseline is locally plausible (it was trained on
  Kircher's own tablets!) but not strictly rule-compliant because of
  Laplace smoothing and the out-of-context bass line.
* The hybrid system improves violation counts and forces cadence
  closure, same as in `hybrid_demo.py`.

Writes `output/ngram_eval.json` with per-composition metrics for
downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arca.evaluate import EvaluableComposition, Evaluator
from arca.hybrid import HybridSampler, RuleProjector
from arca.integrations import NGramSampler, train_ngram_on_kircher


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="compositions per system")
    ap.add_argument("--chord-len", type=int, default=6, help="chords per composition")
    ap.add_argument("--alpha", type=float, default=0.5, help="Laplace smoothing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/ngram_eval.json")
    args = ap.parse_args()

    # 1. Train the reference n-gram (instant — fits ~20 phrases)
    model = train_ngram_on_kircher(alpha=args.alpha)
    ngram_sampler = NGramSampler(model)

    # 2. Build the two systems: pure n-gram and hybrid
    pure = HybridSampler(sampler=ngram_sampler, use_projector=False)
    hyb  = HybridSampler(sampler=ngram_sampler,
                         projector=RuleProjector(),
                         use_projector=True)

    # 3. Generate N compositions per system
    pure_items: list = []
    hyb_items: list = []
    for i in range(args.n):
        seed = args.seed + i
        midi_p, log_p = pure.compose(args.chord_len, seed=seed)
        midi_h, log_h = hyb.compose(args.chord_len, seed=seed)
        pure_items.append(EvaluableComposition(
            midi_grid=midi_p, rule_log=log_p, label="ngram_only"))
        hyb_items.append(EvaluableComposition(
            midi_grid=midi_h, rule_log=log_h, label="hybrid"))

    # 4. Evaluate + summarize
    ev = Evaluator()
    pure_metrics = ev.evaluate_corpus(pure_items)
    hyb_metrics = ev.evaluate_corpus(hyb_items)
    pure_summary = ev.summarize(pure_metrics)
    hyb_summary = ev.summarize(hyb_metrics)

    # 5. Print the comparison table
    print("\n=== Reference n-gram integration — N =", args.n, "===")
    print(ev.comparison_table([pure_summary, hyb_summary]))

    # 6. Persist per-composition metrics for downstream analysis
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
