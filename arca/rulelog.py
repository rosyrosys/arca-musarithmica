"""
arca.rulelog
------------

Structured rule log — the XAI core of the Arca prototype.

Every decision the engine makes emits a RuleEntry. The log is both
machine-readable (JSON) for downstream evaluation, and human-readable
(formatted text) for reviewer/reader inspection. This is the direct
operationalization of the paper's claim that Kircher's method is
"auditable and teachable" step-by-step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# Stage identifiers follow the paper's five-step pseudocode exactly.
STAGE_SELECTION = "selection"
STAGE_CONVERSION = "conversion"
STAGE_CONSTRAINT = "constraint_check"
STAGE_CADENCE = "cadential_template"
STAGE_OUTPUT = "output"

ALL_STAGES = (
    STAGE_SELECTION,
    STAGE_CONVERSION,
    STAGE_CONSTRAINT,
    STAGE_CADENCE,
    STAGE_OUTPUT,
)


@dataclass
class RuleEntry:
    """A single step in the rule log."""
    stage: str
    rule: str                # short slug, e.g. "no_parallel_fifths"
    message: str             # human-readable sentence
    # Arbitrary structured payload — may hold numerical arrays, pitches,
    # chord symbols, violation indices, etc.
    data: Dict[str, Any] = field(default_factory=dict)
    # Indicates whether this entry represents a passing check, a correction,
    # or an informational note. Kept as a string (not enum) for JSON clarity.
    status: str = "ok"       # "ok" | "warning" | "correction" | "info"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuleLog:
    """Ordered collection of rule entries, emitted by the engine."""
    entries: List[RuleEntry] = field(default_factory=list)

    # ---- writers -----------------------------------------------------------

    def add(
        self,
        stage: str,
        rule: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        status: str = "ok",
    ) -> RuleEntry:
        entry = RuleEntry(
            stage=stage, rule=rule, message=message,
            data=dict(data or {}), status=status,
        )
        self.entries.append(entry)
        return entry

    def extend(self, other: "RuleLog") -> None:
        self.entries.extend(other.entries)

    # ---- readers -----------------------------------------------------------

    def by_stage(self, stage: str) -> List[RuleEntry]:
        return [e for e in self.entries if e.stage == stage]

    def warnings(self) -> List[RuleEntry]:
        return [e for e in self.entries if e.status == "warning"]

    def corrections(self) -> List[RuleEntry]:
        return [e for e in self.entries if e.status == "correction"]

    def coverage(self) -> Dict[str, int]:
        """Entries per stage — useful as a paper-ready interpretability metric.

        The paper defines interpretability as "the ability to trace and explain
        the mapping from inputs to musical outputs step-by-step by a human
        observer". A fully populated rule log across all five stages is a
        direct operational measure of that claim.
        """
        return {stage: len(self.by_stage(stage)) for stage in ALL_STAGES}

    # ---- serialization -----------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "entries": [e.to_dict() for e in self.entries],
                "coverage": self.coverage(),
            },
            indent=indent,
            ensure_ascii=False,
        )

    def human_readable(self) -> str:
        """Return a formatted log for inclusion in reports / paper appendix."""
        lines: List[str] = []
        current: Optional[str] = None
        stage_labels = {
            STAGE_SELECTION:   "STEP 1 — SELECTION",
            STAGE_CONVERSION:  "STEP 2 — CONVERSION",
            STAGE_CONSTRAINT:  "STEP 3 — CONSTRAINT CHECK",
            STAGE_CADENCE:     "STEP 4 — CADENTIAL TEMPLATE",
            STAGE_OUTPUT:      "STEP 5 — OUTPUT",
        }
        status_marker = {
            "ok":         " ok ",
            "warning":    "warn",
            "correction": "corr",
            "info":       "info",
        }
        for e in self.entries:
            if e.stage != current:
                if current is not None:
                    lines.append("")
                lines.append(stage_labels.get(e.stage, e.stage.upper()))
                lines.append("-" * len(stage_labels.get(e.stage, e.stage)))
                current = e.stage
            marker = status_marker.get(e.status, " ?  ")
            lines.append(f"  [{marker}] {e.rule}: {e.message}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)
