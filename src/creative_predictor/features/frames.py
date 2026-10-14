"""Frames e áudio de cada Short, extraídos com ffmpeg.

8 frames por vídeo: 3 nos primeiros 3 segundos (o gancho) e 5 espalhados pelo
resto. O áudio vira WAV mono de 16 kHz, o formato que o Whisper espera. Tudo
fica em `data/interim/frames/<video_id>/` e `data/interim/audio/<video_id>.wav`
e não é extraído de novo.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from creative_predictor import config

FRAMES_DIR = config.INTERIM_DIR / "frames"
AUDIO_DIR = config.INTERIM_DIR / "audio"
HOOK_TIMES = (0.5, 1.5, 2.5)  # segundos, dentro do gancho
REST_FRAMES = 5


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def frame_times(duration: float) -> list[float]:
    """Instantes dos frames: o gancho e o restante do vídeo.

    Vídeos com menos de 4 s recebem 8 instantes espalhados por toda a duração.
    """
    n = len(HOOK_TIMES) + REST_FRAMES
    if duration < 4:
        return [round(t, 2) for t in np.linspace(0, duration, n + 2)[1:-1]]
    rest = np.linspace(3.0, duration - 0.5, REST_FRAMES)
    return [*HOOK_TIMES, *(round(float(t), 2) for t in rest)]


def extract_frames(video: Path, out_dir: Path) -> list[Path]:
    """Extrai os frames (JPEG, lado menor 360 px); reaproveita os existentes."""
    times = frame_times(probe_duration(video))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times):
        path = out_dir / f"frame_{i}.jpg"
        if not path.is_file():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(t),
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(path),
                ],
                check=True,
            )
        paths.append(path)
    return paths


def extract_audio(video: Path, out: Path) -> Path | None:
    """WAV mono 16 kHz; None se o vídeo não tem trilha de áudio."""
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,  # vídeo sem trilha de áudio falha aqui e vira None
    )
    if done.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        return None
    return out


def frames_for(video_id: str, video: Path) -> list[Path]:
    return extract_frames(video, FRAMES_DIR / video_id)


def audio_for(video_id: str, video: Path) -> Path | None:
    return extract_audio(video, AUDIO_DIR / f"{video_id}.wav")
