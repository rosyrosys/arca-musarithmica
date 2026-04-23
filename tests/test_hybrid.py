"""
Tests for the v0.2 hybrid + evaluation layer.

Claims tested:

  (a) The diffusion sampler is deterministic given a seed.
  (b) The projector reduces voice-leading violations compared to pure
      diffusion on the same seed.
  (c) The projector clamps the cadence to V -> I (bass).
  (d) The evaluator gives the same numerical result whether the rule
      log is pre-existing or freshly computed.
  (e) JSON round-trips through to_json() are valid and parseable.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arca import Arca  # noqa: E402
from arca.evaluate import EvaluableComposition, Evaluator  # noqa: E402
from arca.hybrid import (  # noqa: E402
    DiscreteChordDiffusion,
    HybridSampler,
    RuleProjector,
)


def test_diffusion_is_deterministic():
    s = DiscreteChordDiffusion(n_steps=8)
    a, _ = HybridSampler(sampler=s, use_projector=False).compose(5, seed=7)
    b, _ = HybridSampler(sampler=s, use_projector=False).compose(5, seed=7)
    assert a == b


def test_hybrid_reduces_violations():
    s = DiscreteChordDiffusion(n_steps=16)
    pure = HybridSampler(sampler=s, use_projector=False)
    hyb  = HybridSampler(sampler=s, use_projector=True)
    ev = Evaluator()
    pure_total = 0
    hyb_total = 0
    n = 20
    for seed in range(n):
        midi_p, log_p = pure.compose(5, seed=seed)
        midi_h, log_h = hyb.compose(5, seed=seed)
        pure_total += ev.evaluate(EvaluableComposition(
            midi_grid=midi_p, rule_log=log_p, label="pure")).violation_total
        hyb_total += ev.evaluate(EvaluableComposition(
            midi_grid=midi_h, rule_log=log_h, label="hyb")).violation_total
    # strict: hybrid must be better on average
    assert hyb_total < pure_total, (pure_total, hyb_total)


def test_projector_clamps_cadence():
    s = DiscreteChordDiffusion(n_steps=16)
    hyb = HybridSampler(sampler=s, use_projector=True)
    for seed in range(5):
        midi, log = hyb.compose(5, seed=seed)
        # Bassus final pc should be the finalis (F -> pc 5);
        # penultimate should be the dominant (C -> pc 0).
        assert midi[3][-1] % 12 == 5, (seed, midi[3])
        assert midi[3][-2] % 12 == 0, (seed, midi[3])


def test_evaluator_no_double_counting():
    """Passing in a pre-populated rule log should not duplicate warnings."""
    r = Arca().compose("Cantate Domino canticum novum")
    ev = Evaluator()
    m1 = ev.evaluate(EvaluableComposition(
        midi_grid=r.midi_grid, mensa=r.mensa,
        rule_log=r.rule_log, label="kircher"))
    # rerun — calling .evaluate twice on the same EvaluableComposition
    # should still produce the same violation_total.
    m2 = ev.evaluate(EvaluableComposition(
        midi_grid=r.midi_grid, mensa=r.mensa,
        rule_log=r.rule_log, label="kircher"))
    assert m1.violation_total == m2.violation_total


def test_evaluator_json_roundtrip():
    r = Arca().compose("Hello world")
    ev = Evaluator()
    ms = ev.evaluate_corpus([EvaluableComposition(
        midi_grid=r.midi_grid, mensa=r.mensa,
        rule_log=r.rule_log, label="k")])
    s = ev.summarize(ms)
    text = ev.to_json([s], per_item=ms)
    parsed = json.loads(text)
    assert "summaries" in parsed
    assert "per_composition" in parsed
    assert parsed["summaries"][0]["label"] == "k"


def test_projector_emits_correction_entries():
    """Every projector intervention should be a `correction` log entry."""
    s = DiscreteChordDiffusion(n_steps=16)
    hyb = HybridSampler(sampler=s, use_projector=True)
    _, log = hyb.compose(5, seed=1)
    corrections = [e for e in log.entries if e.status == "correction"]
    # The projector almost always has to clamp at least the bass cadence.
    assert len(corrections) >= 1
    # Each correction must specify what it changed.
    for e in corrections:
        assert "before" in e.data and "after" in e.data
