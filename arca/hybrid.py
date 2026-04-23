"""
arca.hybrid
-----------

Hybrid neural-symbolic integration. The paper's future-work section
argues that Kircher's rule system can wrap a modern generative model as
a pre/post-processor, yielding an output that is both fluent (from the
neural side) and auditable (from Kircher's side). This module is the
minimal working scaffolding for that claim.

Three components:

1. `DiscreteChordDiffusion`
   A categorical-denoising sampler over four-voice scale-degree grids.
   It starts from uniform random tokens and iteratively denoises toward
   a distribution shaped by (a) a mild tonic/dominant prior at the
   cadence and (b) neighborhood smoothing. The noise schedule is a
   simple cosine ramp. *No training required* — the point is to produce
   a sampler that is clearly *not* rule-compliant so that the projector
   has something real to correct. A real diffusion checkpoint (e.g.
   MusicLDM, stable-audio, a symbolic Music-Transformer) slots in
   behind the same interface by implementing `sample_degree_grid`.

2. `RuleProjector`
   A post-sampling projector. For each chord, it attempts to snap the
   proposal to the nearest rule-compliant chord. Each correction it
   makes is written to the `RuleLog` as a `"correction"` entry — this
   is the XAI mechanism the next paper needs: every neural output
   carries a structured trace of what Kircher's rules changed and
   why.

3. `HybridSampler`
   Wraps any `DegreeGridSampler` and a `RuleProjector`. Running it is
   the hybrid system referenced in the paper; running the sampler
   alone is the pure-neural baseline. This symmetry is what lets
   `arca.evaluate` produce apples-to-apples comparison tables.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

from .data import (
    MENSA_F_MAJOR,
    MensaLine,
    VOICE_CENTERS,
    VOICE_RANGES,
    pitch_to_midi,
    place_chord,
)
from .rulelog import RuleLog


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

# A degree-grid sampler returns a 4xN grid of Kircher numbers (1..8).
# This is the "symbolic latent" the rest of the pipeline can place into
# MIDI via the Mensa + place_chord routines.
class DegreeGridSampler(Protocol):
    def sample_degree_grid(self, n_chords: int, rng: random.Random) -> List[List[int]]:
        ...


# ---------------------------------------------------------------------------
# 1.  Discrete diffusion over scale-degree grids
# ---------------------------------------------------------------------------

# A neural-like baseline. It is purposely "weak on theory but fluent" so
# that the hybrid's contribution is measurable. No gradients; uses only
# random.Random so the output is fully deterministic given a seed.

_DEGREE_TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)


@dataclass
class DiscreteChordDiffusion:
    """Categorical denoising sampler over four-voice Kircher-number grids.

    T denoising steps. At each step, a random subset of cells is resampled
    from a conditional that mildly prefers the local mean and (for bass)
    tonic/dominant at phrase boundaries. Early steps are near-uniform;
    later steps sharpen the distribution (temperature decay).

    The output obeys the scale-degree vocabulary but NOT any voice-leading
    or cadence constraint — so it reliably generates the kinds of errors
    the projector needs to catch. Swap with a real checkpoint later.
    """
    n_steps: int = 16
    temperature_start: float = 1.4
    temperature_end: float = 0.4
    seed: Optional[int] = None

    def sample_degree_grid(
        self, n_chords: int, rng: Optional[random.Random] = None,
    ) -> List[List[int]]:
        r = rng or random.Random(self.seed)
        # initial state: uniform random choice per cell
        grid = [[r.choice(_DEGREE_TOKENS) for _ in range(n_chords)]
                for _ in range(4)]

        for step in range(self.n_steps):
            # cosine temperature schedule
            frac = step / max(1, self.n_steps - 1)
            tau = self.temperature_start + frac * (
                self.temperature_end - self.temperature_start)
            # proportion of cells resampled this step (noise fraction)
            resample_fraction = 0.5 * (1 - frac) + 0.05
            for v in range(4):
                for k in range(n_chords):
                    if r.random() > resample_fraction:
                        continue
                    grid[v][k] = self._resample_cell(grid, v, k, n_chords, tau, r)
        return grid

    # -- conditional resampling --------------------------------------------

    def _resample_cell(
        self, grid: List[List[int]], v: int, k: int, n: int,
        tau: float, r: random.Random,
    ) -> int:
        # Build unnormalized log-probabilities per token.
        logits: List[float] = []
        # local mean (smoothness prior)
        neighbors = []
        if k > 0: neighbors.append(grid[v][k-1])
        if k < n-1: neighbors.append(grid[v][k+1])
        if v > 0: neighbors.append(grid[v-1][k])
        if v < 3: neighbors.append(grid[v+1][k])
        local_mean = sum(neighbors) / len(neighbors) if neighbors else 4.5

        for tok in _DEGREE_TOKENS:
            # proximity prior — prefer tokens near the local mean
            score = -abs(tok - local_mean)
            # bass cadence prior: final chord likes 1 or 8; penultimate likes 5
            if v == 3:
                if k == n-1 and tok in (1, 8):
                    score += 1.0
                elif k == n-2 and tok == 5:
                    score += 0.8
            # cantus cadence prior: final prefers 1, 3
            if v == 0 and k == n-1 and tok in (1, 3):
                score += 0.6
            logits.append(score / max(1e-3, tau))

        # softmax -> multinomial draw
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        u = r.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if u < cum:
                return _DEGREE_TOKENS[i]
        return _DEGREE_TOKENS[-1]


# ---------------------------------------------------------------------------
# 2. Degree-grid -> MIDI grid, with smooth voice leading
# ---------------------------------------------------------------------------

def realize_degree_grid(
    degree_grid: List[List[int]],
    mensa: MensaLine,
) -> Tuple[List[List[str]], List[List[int]]]:
    """Realize a 4xN Kircher-number grid into (pitch names, MIDI) grids."""
    n = len(degree_grid[0])
    pitch_grid: List[List[str]] = [
        [mensa.pitch_for(d) for d in row] for row in degree_grid]
    midi_grid: List[List[int]] = [[0]*n for _ in range(4)]
    prev: Optional[List[int]] = None
    for k in range(n):
        pitches = [pitch_grid[v][k] for v in range(4)]
        prefer = [degree_grid[v][k] == 8 for v in range(4)]
        chord = place_chord(pitches, prefer, prev_chord_midis=prev)
        for v in range(4):
            midi_grid[v][k] = chord[v]
        prev = chord
    return pitch_grid, midi_grid


# ---------------------------------------------------------------------------
# 3. Rule projector
# ---------------------------------------------------------------------------

@dataclass
class RuleProjector:
    """Projects a degree grid to the nearest rule-compliant neighbor.

    Strategy (local, greedy, interpretable):

      * Sweep chords left to right. At each chord k>0, enumerate small
        perturbations of the inner voices (Altus, Tenor) and accept the
        first one that removes all parallel-5ths/8ves with chord k-1.
        Inner voices are the correct repair site historically — Cantus
        carries melody, Bassus carries harmonic function.

      * Clamp Bassus at the end to 1 (or 5 at penultimate, 1 at final)
        so a cadence is reachable. This matches Kircher's insistence on
        cadential closure.

    Every change is logged as a `correction` entry with before/after, so
    the projector's contribution is fully auditable.
    """
    max_inner_perturbations: int = 4
    perturbation_radius: int = 2     # ± degrees tested per inner voice

    def project(
        self, degree_grid: List[List[int]], rule_log: RuleLog,
    ) -> List[List[int]]:
        n = len(degree_grid[0])
        grid = [row[:] for row in degree_grid]

        # 1. Cadence clamp — Bassus 5 -> 1 at the final two chords.
        if n >= 2:
            if grid[3][-2] != 5:
                rule_log.add(
                    stage="cadential_template",
                    rule="bass_cadence_clamp",
                    message=f"Clamped Bassus penultimate from {grid[3][-2]} to 5 for V -> I.",
                    data={"chord": n-2, "before": grid[3][-2], "after": 5},
                    status="correction",
                )
                grid[3][-2] = 5
            if grid[3][-1] != 1:
                rule_log.add(
                    stage="cadential_template",
                    rule="bass_cadence_clamp",
                    message=f"Clamped Bassus final from {grid[3][-1]} to 1 for V -> I.",
                    data={"chord": n-1, "before": grid[3][-1], "after": 1},
                    status="correction",
                )
                grid[3][-1] = 1

        # 2. Parallel fifths/octaves sweep (inner-voice perturbation)
        for k in range(1, n):
            corrections = 0
            while self._has_parallel(grid, k):
                if corrections >= self.max_inner_perturbations:
                    break
                changed = self._perturb_inner(grid, k, rule_log)
                if not changed:
                    break
                corrections += 1

        return grid

    # -- helpers -----------------------------------------------------------

    def _has_parallel(self, grid: List[List[int]], k: int) -> bool:
        """True if chord k has a parallel 5th/8ve with chord k-1, checked
        on the realized MIDI grid (uses MENSA_F_MAJOR for realization)."""
        _, midi = realize_degree_grid(
            [row[k-1:k+1] for row in grid], MENSA_F_MAJOR)
        for i in range(4):
            for j in range(i+1, 4):
                a0, a1 = midi[i][0], midi[i][1]
                b0, b1 = midi[j][0], midi[j][1]
                if a0 == a1 and b0 == b1:
                    continue
                iv0 = abs(a0 - b0) % 12
                iv1 = abs(a1 - b1) % 12
                moved = a0 != a1 and b0 != b1
                if moved and iv0 == iv1 and iv0 in (0, 7):
                    return True
        return False

    def _perturb_inner(
        self, grid: List[List[int]], k: int, rule_log: RuleLog,
    ) -> bool:
        """Try to fix chord k by adjusting Altus or Tenor by ±1..radius.
        Returns True if a fix was applied."""
        for voice in (1, 2):   # Altus, Tenor
            original = grid[voice][k]
            for delta in self._perturbations():
                candidate = original + delta
                if candidate < 1 or candidate > 8:
                    continue
                grid[voice][k] = candidate
                if not self._has_parallel(grid, k):
                    rule_log.add(
                        stage="constraint_check",
                        rule="parallel_projector",
                        message=(
                            f"Projector: adjusted "
                            f"{['Cantus','Altus','Tenor','Bassus'][voice]} "
                            f"at chord {k} from {original} to {candidate} "
                            f"to break parallel motion."
                        ),
                        data={
                            "chord": k, "voice": voice,
                            "before": original, "after": candidate,
                        },
                        status="correction",
                    )
                    return True
            grid[voice][k] = original
        return False

    def _perturbations(self):
        # interleave ±1, ±2, ... so smaller edits are tried first
        for r in range(1, self.perturbation_radius + 1):
            yield -r
            yield +r


# ---------------------------------------------------------------------------
# 4. Hybrid sampler
# ---------------------------------------------------------------------------

@dataclass
class HybridSampler:
    """Combine a neural-like degree-grid sampler with a RuleProjector.

    `use_projector=False` gives the neural-only baseline;
    `use_projector=True` gives the hybrid.
    """
    sampler: DegreeGridSampler
    projector: RuleProjector = field(default_factory=RuleProjector)
    mensa: MensaLine = MENSA_F_MAJOR
    use_projector: bool = True

    def compose(
        self, n_chords: int, seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], RuleLog]:
        """Return (midi_grid, rule_log). Rule log includes `correction`
        entries for every projector intervention."""
        rng = random.Random(seed)
        log = RuleLog()
        log.add(
            stage="selection",
            rule="hybrid_plan",
            message=(
                f"Hybrid composition: sampler={type(self.sampler).__name__}, "
                f"projector={'on' if self.use_projector else 'off'}, "
                f"n_chords={n_chords}."
            ),
            data={
                "sampler": type(self.sampler).__name__,
                "use_projector": self.use_projector,
                "n_chords": n_chords,
            },
        )

        degree_grid = self.sampler.sample_degree_grid(n_chords, rng)
        log.add(
            stage="conversion",
            rule="raw_degree_grid",
            message="Raw degree grid produced by the diffusion sampler.",
            data={"grid": [row[:] for row in degree_grid]},
        )

        if self.use_projector:
            degree_grid = self.projector.project(degree_grid, log)
            log.add(
                stage="conversion",
                rule="projected_degree_grid",
                message="Degree grid after rule-projector post-processing.",
                data={"grid": [row[:] for row in degree_grid]},
            )

        _, midi_grid = realize_degree_grid(degree_grid, self.mensa)
        log.add(
            stage="output",
            rule="output_ready",
            message=f"Four-voice composition ready: {n_chords} chords.",
            data={"n_chords": n_chords},
        )
        return midi_grid, log
