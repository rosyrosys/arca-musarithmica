"""
arca — An interpretable, rule-logged reconstruction of Athanasius Kircher's
Arca Musarithmica (Musurgia Universalis, 1650), operationalizing the
five-step algorithm described in:

  "Analysis of harmonic structure of Athanasius Kircher's Arca Musarithmica
   for transparent AI music generation."

The package implements an XAI-style music-generation engine:

    Selection -> Conversion -> Constraint check -> Cadential template -> Output

Each step emits a structured entry in a human-readable rule log so that the
mapping from input text to output four-part harmony is fully auditable.

Public API
----------
    from arca import Arca, MensaTonographica, Pinax

    arca = Arca(mode="F_major")
    result = arca.compose("Cantate Domino canticum novum")
    result.to_midi("out.mid")
    result.to_musicxml("out.xml")
    print(result.rule_log.human_readable())
"""

from .engine import Arca, CompositionResult
from .data import MensaTonographica, Pinax, PINAX_I_SYNTAGMA_I, MENSA_F_MAJOR
from .rulelog import RuleLog, RuleEntry
from .constraints import VoiceLeadingChecker, CadentialTemplate
from .output import MidiWriter, MusicXmlWriter
from .evaluate import EvaluableComposition, Evaluator, CompositionMetrics
from .hybrid import (
    DiscreteChordDiffusion,
    RuleProjector,
    HybridSampler,
    realize_degree_grid,
)
from .integrations import (
    MidiSequenceAdapter,
    midi_to_degree_grid,
    NGramChordModel,
    NGramSampler,
    train_ngram_on_kircher,
    MusicTransformerAdapter,
    load_hf_music_transformer,
    polyphony_to_satb,
    MusicGenAdapter,
    load_musicgen,
)

__version__ = "0.3.0"

__all__ = [
    "Arca",
    "CompositionResult",
    "MensaTonographica",
    "Pinax",
    "PINAX_I_SYNTAGMA_I",
    "MENSA_F_MAJOR",
    "RuleLog",
    "RuleEntry",
    "VoiceLeadingChecker",
    "CadentialTemplate",
    "MidiWriter",
    "MusicXmlWriter",
    # v0.2 — hybrid + evaluation
    "EvaluableComposition",
    "Evaluator",
    "CompositionMetrics",
    "DiscreteChordDiffusion",
    "RuleProjector",
    "HybridSampler",
    "realize_degree_grid",
    # v0.3 — external-model integrations
    "MidiSequenceAdapter",
    "midi_to_degree_grid",
    "NGramChordModel",
    "NGramSampler",
    "train_ngram_on_kircher",
    "MusicTransformerAdapter",
    "load_hf_music_transformer",
    "polyphony_to_satb",
    "MusicGenAdapter",
    "load_musicgen",
]
