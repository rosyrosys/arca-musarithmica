"""
arca.evaluate
-------------

Corpus-level evaluation pipeline. Operationalizes the paper's future-work
claim that the rule log gives us a yardstick usable on *any* four-voice
composition — whether it came from Kircher's Arca, a pure diffusion model,
or a hybrid of the two.

The module has two layers:

1. `EvaluableComposition` — a lightweight adapter. Any system that can
   produce a 4xN MIDI grid plus an optional rule log can be evaluated
   with the same metrics.

2. `Evaluator` — computes a fixed set of metrics per composition, then
   aggregates them across a corpus:

   - `voice_leading_violations`   parallel 5ths/8ves + crossings + OOR
   - `cadence_label`              PAC/IAC/HC/PC/DC/OTHER
   - `rule_log_coverage`          entries per stage (1..5)
   - `pitch_class_entropy`        Shannon entropy of pitch-class usage
   - `mean_voice_motion_semitones` mean absolute motion per voice-step

The metric names are the ones the follow-up paper will reference; they
are chosen to be invariant to phrase length and directly comparable
between systems.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .constraints import CadentialTemplate, VoiceLeadingChecker
from .data import MENSA_F_MAJOR, MensaLine, VOICE_RANGES
from .rulelog import RuleLog


# ---------------------------------------------------------------------------
# Composition adapter
# ---------------------------------------------------------------------------

@dataclass
class EvaluableComposition:
    """A minimal view of a four-voice composition for evaluation.

    This decouples the evaluator from the Arca engine — any generator
    (pure diffusion, external MIDI file, ...) can be wrapped by building
    an EvaluableComposition directly.
    """
    midi_grid: List[List[int]]        # shape 4 x N
    mensa: MensaLine = MENSA_F_MAJOR
    # Optional pre-computed rule log; if None, evaluation will synthesize
    # one by running the constraint + cadence checks on the grid.
    rule_log: Optional[RuleLog] = None
    label: str = ""                   # e.g. "kircher", "diffusion", "hybrid"

    def __post_init__(self) -> None:
        if len(self.midi_grid) != 4:
            raise ValueError("EvaluableComposition requires 4 voices")
        lens = {len(v) for v in self.midi_grid}
        if len(lens) != 1:
            raise ValueError(f"voice length mismatch: {lens}")

    def n_chords(self) -> int:
        return len(self.midi_grid[0])


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _count_violations(rule_log: RuleLog) -> Dict[str, int]:
    """Bucket warning entries by rule slug for easier comparison."""
    buckets = {
        "parallel_fifths":   0,
        "parallel_octaves":  0,
        "voice_crossing":    0,
        "voice_range":       0,
        "other":             0,
    }
    for e in rule_log.entries:
        if e.status != "warning":
            continue
        if e.rule == "no_parallel_fifths":
            buckets["parallel_fifths"] += 1
        elif e.rule == "no_parallel_octaves":
            buckets["parallel_octaves"] += 1
        elif e.rule == "no_voice_crossing":
            buckets["voice_crossing"] += 1
        elif e.rule == "voice_range":
            buckets["voice_range"] += 1
        else:
            buckets["other"] += 1
    return buckets


def _pitch_class_entropy(midi_grid: List[List[int]]) -> float:
    counts: Dict[int, int] = {}
    total = 0
    for row in midi_grid:
        for m in row:
            pc = m % 12
            counts[pc] = counts.get(pc, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def _mean_voice_motion(midi_grid: List[List[int]]) -> float:
    total = 0
    count = 0
    for row in midi_grid:
        for k in range(len(row) - 1):
            total += abs(row[k + 1] - row[k])
            count += 1
    return total / count if count else 0.0


@dataclass
class CompositionMetrics:
    label: str
    n_chords: int
    violations: Dict[str, int]
    violation_total: int
    cadence_label: str
    rule_log_coverage: Dict[str, int]
    pitch_class_entropy: float
    mean_voice_motion_semitones: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n_chords": self.n_chords,
            "violations": dict(self.violations),
            "violation_total": self.violation_total,
            "cadence_label": self.cadence_label,
            "rule_log_coverage": dict(self.rule_log_coverage),
            "pitch_class_entropy": self.pitch_class_entropy,
            "mean_voice_motion_semitones": self.mean_voice_motion_semitones,
        }


class Evaluator:
    """Compute and aggregate per-composition metrics."""

    def evaluate(self, comp: EvaluableComposition) -> CompositionMetrics:
        rule_log = comp.rule_log
        if rule_log is None:
            rule_log = RuleLog()
        # Only run checks that the caller hasn't already performed —
        # otherwise warnings would be double-counted when evaluating a
        # composition that already went through the Arca engine.
        # Check for the specific rule entries the metrics depend on, not
        # just the stage — a log can contain projector `correction` entries
        # on the cadential_template stage without ever having run the
        # actual cadence detector.
        has_constraint = any(
            e.stage == "constraint_check" and e.rule in {
                "no_parallel_fifths_or_octaves", "no_parallel_fifths",
                "no_parallel_octaves", "no_voice_crossing", "voice_range",
            } for e in rule_log.entries)
        has_cadence = any(
            e.stage == "cadential_template" and e.rule == "cadence_detection"
            for e in rule_log.entries)
        if not has_constraint:
            VoiceLeadingChecker().check(comp.midi_grid, rule_log)
        if not has_cadence:
            CadentialTemplate().apply(comp.midi_grid, comp.mensa, rule_log)
        # Pull the cadence label from the log (whether fresh or pre-existing).
        cadence = "OTHER"
        for e in rule_log.entries:
            if e.stage == "cadential_template" and e.rule == "cadence_detection":
                cadence = e.data.get("detected", "OTHER")
        viols = _count_violations(rule_log)
        return CompositionMetrics(
            label=comp.label,
            n_chords=comp.n_chords(),
            violations=viols,
            violation_total=sum(viols.values()),
            cadence_label=cadence,
            rule_log_coverage=rule_log.coverage(),
            pitch_class_entropy=_pitch_class_entropy(comp.midi_grid),
            mean_voice_motion_semitones=_mean_voice_motion(comp.midi_grid),
        )

    # -- aggregation --------------------------------------------------------

    def evaluate_corpus(
        self, comps: Sequence[EvaluableComposition],
    ) -> List[CompositionMetrics]:
        return [self.evaluate(c) for c in comps]

    def summarize(
        self, metrics: Sequence[CompositionMetrics],
    ) -> Dict[str, Any]:
        """Aggregate per-system summary: means and distributions."""
        if not metrics:
            return {}
        n = len(metrics)
        cadence_hist: Dict[str, int] = {}
        for m in metrics:
            cadence_hist[m.cadence_label] = cadence_hist.get(m.cadence_label, 0) + 1
        return {
            "n_compositions": n,
            "label": metrics[0].label,
            "mean_violations_per_composition": sum(m.violation_total for m in metrics) / n,
            "fraction_clean": sum(1 for m in metrics if m.violation_total == 0) / n,
            "mean_pitch_class_entropy": sum(m.pitch_class_entropy for m in metrics) / n,
            "mean_voice_motion_semitones": sum(m.mean_voice_motion_semitones for m in metrics) / n,
            "cadence_histogram": cadence_hist,
            "full_coverage_fraction": sum(
                1 for m in metrics
                if all(v >= 1 for v in m.rule_log_coverage.values())
            ) / n,
        }

    # -- reporting ----------------------------------------------------------

    @staticmethod
    def comparison_table(summaries: Sequence[Dict[str, Any]]) -> str:
        """Render a fixed-width comparison table for stdout / papers."""
        rows = []
        header = [
            "system",
            "N",
            "mean_viols",
            "%clean",
            "H(pc)",
            "motion",
            "cov%",
            "cadences",
        ]
        rows.append(header)
        for s in summaries:
            rows.append([
                s.get("label", "?"),
                str(s.get("n_compositions", 0)),
                f"{s.get('mean_violations_per_composition', 0):.2f}",
                f"{100 * s.get('fraction_clean', 0):.0f}%",
                f"{s.get('mean_pitch_class_entropy', 0):.2f}",
                f"{s.get('mean_voice_motion_semitones', 0):.2f}",
                f"{100 * s.get('full_coverage_fraction', 0):.0f}%",
                ", ".join(f"{k}:{v}" for k, v in sorted(
                    s.get("cadence_histogram", {}).items(),
                    key=lambda x: -x[1])),
            ])
        widths = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
        out = []
        for j, row in enumerate(rows):
            line = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
            out.append(line)
            if j == 0:
                out.append("  ".join("-" * w for w in widths))
        return "\n".join(out)

    @staticmethod
    def to_json(
        summaries: Sequence[Dict[str, Any]],
        per_item: Optional[Sequence[CompositionMetrics]] = None,
    ) -> str:
        payload: Dict[str, Any] = {"summaries": list(summaries)}
        if per_item is not None:
            payload["per_composition"] = [m.to_dict() for m in per_item]
        return json.dumps(payload, indent=2, ensure_ascii=False)
