"""
arca.integrations.reference_ngram
---------------------------------

A concrete, trainable 2nd-order Markov (trigram) model over chord tokens.
Not the main event — but it does three important things for the paper:

  1. Demonstrates that the `DegreeGridSampler` / `MidiSequenceAdapter`
     interface is real. The same protocol that a symbolic Music
     Transformer or MusicGen would use is exercised here by a model
     that actually runs in the sandbox.
  2. Provides a *reproducibility-safe* neural-ish baseline. Reviewers
     with no GPU can still run the full hybrid comparison experiment
     end-to-end.
  3. Gives the follow-up paper a lower-bound datum: if the projector
     improves even a heavily-corpus-trained Markov, it will improve
     a much stronger transformer's raw samples too.

Trained on the Kircher Pinax corpus included in `arca.data`. Training
is instantaneous (N of order 20 phrases) and requires zero external
packages.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..data import PINAX_I_SYNTAGMA_I


# A chord token here is a 4-tuple of Kircher numbers (Cantus, Altus, Tenor, Bassus).
ChordToken = Tuple[int, int, int, int]

BOS: ChordToken = (0, 0, 0, 0)   # sentinel for "start of phrase"


@dataclass
class NGramChordModel:
    """Trigram chord-level language model.

    Context = (prev_prev_chord, prev_chord). Prediction = next chord.
    Backoff: if the full bigram context is unseen, fall back to unigram
    over chord tokens; if that's unseen, uniform over the vocabulary.

    Smoothing: additive (Laplace). `alpha` controls how "diffuse" the
    model feels; small alpha -> sharp, deterministic sampler; large
    alpha -> more surprising, more rule-violating output (which is what
    we actually want as the neural-ish baseline).
    """
    alpha: float = 0.5

    # Populated by .fit()
    vocab: List[ChordToken] = field(default_factory=list)
    _trigram: Dict[Tuple[ChordToken, ChordToken], Dict[ChordToken, int]] = \
        field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _bigram: Dict[ChordToken, Dict[ChordToken, int]] = \
        field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _unigram: Dict[ChordToken, int] = \
        field(default_factory=lambda: defaultdict(int))

    # -- training ----------------------------------------------------------

    def fit(self, phrases: List[List[ChordToken]]) -> "NGramChordModel":
        vocab = set()
        for phrase in phrases:
            vocab.update(phrase)
        vocab.add(BOS)
        self.vocab = sorted(vocab)
        for phrase in phrases:
            ctx = (BOS, BOS)
            for tok in phrase:
                self._trigram[ctx][tok] += 1
                self._bigram[ctx[1]][tok] += 1
                self._unigram[tok] += 1
                ctx = (ctx[1], tok)
        return self

    # -- sampling ----------------------------------------------------------

    def sample(
        self, n_chords: int, rng: Optional[random.Random] = None,
    ) -> List[ChordToken]:
        r = rng or random.Random()
        out: List[ChordToken] = []
        ctx = (BOS, BOS)
        for _ in range(n_chords):
            probs = self._next_distribution(ctx)
            tok = self._multinomial(probs, r)
            out.append(tok)
            ctx = (ctx[1], tok)
        return out

    def _next_distribution(
        self, ctx: Tuple[ChordToken, ChordToken],
    ) -> List[Tuple[ChordToken, float]]:
        """Smoothed trigram with bigram backoff."""
        tri = self._trigram.get(ctx, {})
        bi  = self._bigram.get(ctx[1], {})
        V = len(self.vocab)
        out: List[Tuple[ChordToken, float]] = []
        for tok in self.vocab:
            c3 = tri.get(tok, 0)
            c2 = bi.get(tok, 0)
            c1 = self._unigram.get(tok, 0)
            # Interpolated + Laplace-smoothed probability
            n3 = sum(tri.values())
            n2 = sum(bi.values())
            n1 = sum(self._unigram.values())
            p3 = (c3 + self.alpha) / (n3 + self.alpha * V) if n3 > 0 else 0
            p2 = (c2 + self.alpha) / (n2 + self.alpha * V) if n2 > 0 else 0
            p1 = (c1 + self.alpha) / (n1 + self.alpha * V) if n1 > 0 else 1.0 / V
            # weights favor higher-order when data is dense
            w3 = 0.6 if n3 > 0 else 0.0
            w2 = 0.3 if n2 > 0 else 0.0
            w1 = 1.0 - w3 - w2
            p = w3 * p3 + w2 * p2 + w1 * p1
            out.append((tok, p))
        z = sum(p for _, p in out)
        return [(t, p / z) for t, p in out]

    @staticmethod
    def _multinomial(
        probs: List[Tuple[ChordToken, float]],
        rng: random.Random,
    ) -> ChordToken:
        u = rng.random()
        cum = 0.0
        for tok, p in probs:
            cum += p
            if u < cum:
                return tok
        return probs[-1][0]


# ---------------------------------------------------------------------------
# Training on the Kircher Pinax — produces a ready-to-sample model
# ---------------------------------------------------------------------------

def train_ngram_on_kircher(alpha: float = 0.5) -> NGramChordModel:
    """Train a trigram on every tablet in Pinax I, Syntagma I."""
    phrases: List[List[ChordToken]] = []
    for tablets in PINAX_I_SYNTAGMA_I.tablets_by_syllables.values():
        for tablet in tablets:
            n = tablet.syllables
            phrase = [
                (tablet.voices[0][k], tablet.voices[1][k],
                 tablet.voices[2][k], tablet.voices[3][k])
                for k in range(n)
            ]
            phrases.append(phrase)
    return NGramChordModel(alpha=alpha).fit(phrases)


# ---------------------------------------------------------------------------
# DegreeGridSampler adapter
# ---------------------------------------------------------------------------

class NGramSampler:
    """Wrap an NGramChordModel as a `DegreeGridSampler`."""

    def __init__(self, model: NGramChordModel) -> None:
        self.model = model

    def sample_degree_grid(
        self, n_chords: int, rng: Optional[random.Random] = None,
    ) -> List[List[int]]:
        tokens = self.model.sample(n_chords, rng)
        grid = [[0]*n_chords for _ in range(4)]
        for k, tok in enumerate(tokens):
            for v in range(4):
                val = tok[v]
                if val == 0:  # BOS leaked through; resample from unigram
                    rr = rng or random.Random()
                    vocab = [t for t in self.model.vocab if t != BOS]
                    if vocab:
                        val = rr.choice(vocab)[v]
                    else:
                        val = 1
                grid[v][k] = val
        return grid
