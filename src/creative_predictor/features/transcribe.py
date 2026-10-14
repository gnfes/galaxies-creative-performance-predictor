"""Transcrição do áudio com faster-whisper, com cache por vídeo.

Além do texto, guarda idioma detectado, duração com fala e densidade de fala
(palavras por segundo de vídeo). O filtro de voz (VAD) evita que trechos só
com música virem texto inventado.
"""

import json
from pathlib import Path

from creative_predictor import config

WHISPER_MODEL = "small"
TRANSCRIPTS_DIR = config.INTERIM_DIR / "transcripts"

_model = None


def _load(model_name: str):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _model


def summarize(segments: list[dict], info: dict, duration: float) -> dict:
    """Texto e métricas de fala a partir dos segmentos do Whisper."""
    text = " ".join(s["text"].strip() for s in segments).strip()
    speech = sum(s["end"] - s["start"] for s in segments)
    words = len(text.split())
    return {
        "text": text,
        "language": info["language"],
        "language_probability": round(info["language_probability"], 3),
        "duration": round(duration, 2),
        "speech_seconds": round(speech, 2),
        "speech_share": round(speech / duration, 3) if duration else 0.0,
        "n_words": words,
        "words_per_second": round(words / duration, 3) if duration else 0.0,
    }


def transcribe(
    video_id: str, audio: Path | None, duration: float, model_name: str = WHISPER_MODEL
) -> dict:
    """Transcrição de um Short; vídeo sem áudio vira transcrição vazia."""
    path = TRANSCRIPTS_DIR / model_name / f"{video_id}.json"
    if path.is_file():
        return json.loads(path.read_text())
    if audio is None:
        result = summarize(
            [], {"language": None, "language_probability": 0.0}, duration
        )
    else:
        segments, info = _load(model_name).transcribe(
            str(audio), vad_filter=True, beam_size=1
        )
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        result = summarize(
            segments,
            {
                "language": info.language,
                "language_probability": info.language_probability,
            },
            duration,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False))
    return result
