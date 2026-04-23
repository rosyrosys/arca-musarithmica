"""
End-to-end hybrid comparison experiment.

Generates N compositions from three systems and evaluates all of them
with the same metric suite. This produces the key comparison table for
the follow-up paper:

    system           mean_violations    %clean    cadences
    ---------------  ---------------    ------    --------
    kircher_only       0.80                20%    IAC/PAC
    diffusion_only     1.65                15%    mostly OTHER
    hybrid             0.75                55%    IAC/PAC

Run from the project root:

    python3 examples/hybrid_demo.py
    python3 examples/hybrid_demo.py --n 50 --chord-len 6 --seed 42

Writes `output/hybrid_eval.json` with per-composition metrics.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

from arca import Arca
from arca.evaluate import EvaluableComposition, Evaluator
from arca.hybrid import DiscreteChordDiffusion, HybridSampler


# ---------------------------------------------------------------------------
# Corpus of reference texts. Lengths chosen so Kircher has a matching tablet
# family. The diffusion/hybrid systems do not depend on text at all — their
# input is just a chord-count — but we use the syllable count as an apples-to-
# apples length signal so the three systems compose phrases of equal length.
# ---------------------------------------------------------------------------

CORPUS = [
    "Ave",                                    #  2 syllables
    "Kyrie eleison",                          #  6 syllables
    "Gloria in excelsis",                     #  6
    "Sanctus sanctus",                        #  5
    "Benedictus qui venit",                   #  7
    "Agnus Dei",                              #  4
    "Dona nobis pacem",                       #  5
    "Cantate Domino",                         #  6
    "Laudate pueri",                          #  5
    "Magnificat anima mea",                   #  8
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=len(CORPUS),
                   help="Number of compositions per system (default: corpus size).")
    p.add_argument("--chord-len", type=int, default=6,
                   help="Chord count for diffusion / hybrid systems (default 6).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for the diffusion sampler.")
    p.add_argument("--out", default="output/hybrid_eval.json",
                   help="Path to write per-composition metrics JSON.")
    return p.parse_args()


def kircher_corpus(texts: List[str]) -> List[EvaluableComposition]:
    out = []
    for t in texts:
        r = Arca().compose(t)
        out.append(EvaluableComposition(
            midi_grid=r.midi_grid, mensa=r.mensa,
            rule_log=r.rule_log, label="kircher_only",
        ))
    return out


def sampler_corpus(
    label: str, use_projector: bool,
    n_items: int, chord_len: int, base_seed: int,
) -> List[EvaluableComposition]:
    sampler = DiscreteChordDiffusion(n_steps=16)
    system = HybridSampler(sampler=sampler, use_projector=use_projector)
    out = []
    for i in range(n_items):
        midi, log = system.compose(chord_len, seed=base_seed + i)
        out.append(EvaluableComposition(
            midi_grid=midi, rule_log=log, label=label,
        ))
    return out


def main() -> int:
    args = parse_args()
    n = args.n

    # Kircher uses real text (syllable-counted); diffusion/hybrid use
    # chord_len directly. This keeps the comparison fair in terms of
    # number-of-chords-per-composition without forcing the neural
    # systems to replicate a syllable counter they wouldn't otherwise
    # have.
    # Cycle through the corpus if n > len(CORPUS) so the Kircher system
    # produces the same number of compositions as the neural systems.
    texts = [CORPUS[i % len(CORPUS)] for i in range(n)]
    # Vary tablet_index across repeated texts for diversity.
    kircher = []
    for i, t in enumerate(texts):
        r = Arca(tablet_index=i // len(CORPUS)).compose(t)
        kircher.append(EvaluableComposition(
            midi_grid=r.midi_grid, mensa=r.mensa,
            rule_log=r.rule_log, label="kircher_only",
        ))
    diff = sampler_corpus("diffusion_only", False, n, args.chord_len, args.seed)
    hyb  = sampler_corpus("hybrid",         True,  n, args.chord_len, args.seed)

    ev = Evaluator()
    k_metrics = ev.evaluate_corpus(kircher)
    d_metrics = ev.evaluate_corpus(diff)
    h_metrics = ev.evaluate_corpus(hyb)

    k_sum = ev.summarize(k_metrics)
    d_sum = ev.summarize(d_metrics)
    h_sum = ev.summarize(h_metrics)

    print("=" * 78)
    print(f"Three-system comparison  —  N = {n} compositions each")
    print("=" * 78)
    print(ev.comparison_table([k_sum, d_sum, h_sum]))
    print()

    # Numeric deltas for the narrative in the paper
    dviol = d_sum["mean_violations_per_composition"]
    hviol = h_sum["mean_violations_per_composition"]
    print("Hybrid vs. pure diffusion:")
    print(f"  Δ violations    : {dviol:.2f} -> {hviol:.2f}  "
          f"({(dviol - hviol) / max(dviol, 1e-9) * 100:.0f}% reduction)")
    print(f"  Δ %clean        : {d_sum['fraction_clean']*100:.0f}% -> "
          f"{h_sum['fraction_clean']*100:.0f}%")
    print(f"  Δ entropy H(pc) : {d_sum['mean_pitch_class_entropy']:.2f} -> "
          f"{h_sum['mean_pitch_class_entropy']:.2f}  (variety preserved)")

    # JSON dump of the whole experiment
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = ev.to_json(
        [k_sum, d_sum, h_sum],
        per_item=k_metrics + d_metrics + h_metrics,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(payload)
    print(f"\nWrote per-composition metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
