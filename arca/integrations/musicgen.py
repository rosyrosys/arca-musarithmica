"""
arca.integrations.musicgen
--------------------------

Adapter for audio-generative checkpoints — specifically Meta's MusicGen
(audiocraft) — that emit waveforms rather than symbolic events. The
pipeline:

    text prompt
        |                (1) audiocraft MusicGen
        v
    16-24 kHz audio
        |                (2) pitch transcription
        v                    (basic-pitch / crepe / Spotify's
    piano-roll                basic-pitch; any `audio -> MIDI` callable)
        |                (3) polyphony_to_satb (from music_transformer)
        v
    4xN MIDI grid
        |                (4) inherited MidiSequenceAdapter +
        v                    midi_to_degree_grid
    Kircher scale-degree grid
        |                (5) RuleProjector, evaluator, output
        v
    Audited hybrid composition.

None of this runs in the sandbox the repo is developed in — pip is
firewalled and the audio-transcription packages are GPU-heavy. The
adapter is structured so a user with a local GPU can:

    from arca.integrations.musicgen import load_musicgen
    adapter = load_musicgen("facebook/musicgen-small",
                            text_prompt="renaissance polyphony, SATB")
    hybrid = HybridSampler(sampler=adapter, use_projector=True)

and have the pipeline "just work". Every model-specific dependency is
lazily imported inside the loader so importing
``arca.integrations.musicgen`` itself never fails.

Why MusicGen, specifically
==========================

MusicGen is the best-known open-weights text-to-music transformer, and
its audio output is a forcing function for the paper's claim: *even a
continuous-audio generator can be made rule-compliant via a symbolic
post-processor*. If the hybrid reduces voice-leading violations on
MusicGen output (measured after transcription), the argument
generalizes from symbolic models to audio models, which is exactly the
"modern generative model" framing the follow-up paper uses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .base import MidiSequenceAdapter
from .music_transformer import Note, polyphony_to_satb


# The transcriber type: (audio waveform, sample_rate) -> list of
# (start_tick, midi_pitch) pairs.
Transcriber = Callable[[object, int], List[Note]]


@dataclass
class MusicGenAdapter(MidiSequenceAdapter):
    """`MidiSequenceAdapter` for an audio-generative model + transcriber.

    Exposes the MusicGen pipeline through the same ``DegreeGridSampler``
    interface as the n-gram and symbolic transformer adapters, so the
    hybrid-comparison harness doesn't need to know what kind of model
    it's wrapping.
    """
    audio_generator: Optional[Callable[[int, random.Random], Tuple[object, int]]] = None
    transcriber: Optional[Transcriber] = None
    ticks_per_chord: Optional[int] = None

    def __post_init__(self) -> None:
        self.generate_midi = self._generate_from_audio  # type: ignore[assignment]

    def _generate_from_audio(
        self, n_chords: int, rng: random.Random,
    ) -> List[List[int]]:
        if self.audio_generator is None or self.transcriber is None:
            raise RuntimeError(
                "MusicGenAdapter requires both `audio_generator` and "
                "`transcriber`. See `load_musicgen` for a canned setup."
            )
        waveform, sr = self.audio_generator(n_chords, rng)
        notes = self.transcriber(waveform, sr)
        return polyphony_to_satb(notes, n_chords, self.ticks_per_chord)


# ---------------------------------------------------------------------------
# Loader: audiocraft MusicGen + basic-pitch transcription
# ---------------------------------------------------------------------------

def load_musicgen(
    checkpoint: str = "facebook/musicgen-small",
    text_prompt: str = "renaissance polyphony, SATB choir, a cappella",
    duration_seconds: float = 8.0,
) -> MusicGenAdapter:
    """Build an adapter around audiocraft + basic-pitch.

    Lazy-imports `audiocraft` and `basic_pitch` so this module stays
    importable even without those packages installed. Raises with a
    helpful message if either is missing.

    Example::

        adapter = load_musicgen(text_prompt="gregorian chant, four voices")
        hybrid = HybridSampler(sampler=adapter, use_projector=True)
        midi, log = hybrid.compose(n_chords=6, seed=42)

    Parameters
    ----------
    checkpoint
        audiocraft MusicGen checkpoint name.
    text_prompt
        Prompt given to MusicGen. Kept deliberately "polyphonic SATB"
        by default — it's the regime where Kircher's rules are
        meaningful.
    duration_seconds
        MusicGen generation length. Longer gives more chords after
        quantization but is slower.
    """
    try:
        from audiocraft.models import MusicGen  # type: ignore
        from basic_pitch.inference import predict  # type: ignore
    except ImportError as e:
        raise ImportError(
            "load_musicgen needs `audiocraft` and `basic-pitch`. "
            "Install with `pip install audiocraft basic-pitch` and retry. "
            "A GPU is strongly recommended."
        ) from e

    model = MusicGen.get_pretrained(checkpoint)
    model.set_generation_params(duration=duration_seconds)

    def audio_generator(n_chords: int, rng: random.Random) -> Tuple[object, int]:
        # audiocraft accepts only a list-of-strings prompt interface
        res = model.generate([text_prompt], progress=False)
        # res: [1, channels, samples] tensor; collapse to mono
        wav = res[0].mean(dim=0).cpu().numpy()
        return wav, int(model.sample_rate)

    def transcriber(waveform: object, sr: int) -> List[Note]:
        # basic-pitch returns (model_output, midi_data, note_events).
        # We only need the note events.
        import tempfile, soundfile as sf  # type: ignore
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sr)
            _, _, note_events = predict(tmp.name)
        # note_events: list of (start_time_s, end_time_s, pitch_midi, ...)
        # Convert seconds -> ticks at 480 ppq / 120 bpm -> 960 tpb, scale
        # by the beat rate. For quantization we just need *relative* order.
        notes: List[Note] = []
        for ev in note_events:
            start_s = ev[0]
            pitch = int(ev[2])
            ticks = int(start_s * 480.0)   # 480 ticks per second (arbitrary)
            notes.append((ticks, pitch))
        notes.sort()
        return notes

    return MusicGenAdapter(
        generate_midi=lambda n, r: [],   # overridden by __post_init__
        audio_generator=audio_generator,
        transcriber=transcriber,
        log_metadata={
            "checkpoint": checkpoint,
            "text_prompt": text_prompt,
            "type": "audio_transformer",
        },
    )
