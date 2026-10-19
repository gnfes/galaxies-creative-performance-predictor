"""Embeddings do vídeo, do áudio e da transcrição com o EmbeddingGemma 2.

O mesmo modelo multimodal gera três vetores por Short, no mesmo espaço do
embedding do título:
- frames: os 8 frames juntos numa entrada (`<|image|>` × 8), um vetor que
  resume o visual do vídeo;
- áudio: a trilha em 16 kHz (`<|audio|>`), lida em memória;
- transcrição: o texto do Whisper, com o mesmo prefixo de classificação do
  título.

Os vetores (256 dimensões, normalizados) ficam em
`data/interim/embeddings/<tipo>/<video_id>.npy`, um arquivo por vídeo, e não
são recalculados.
"""

import wave
from pathlib import Path

import numpy as np

from creative_predictor import config
from creative_predictor.features.text import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROMPT,
)

MEDIA_EMBEDDINGS_DIR = config.INTERIM_DIR / "embeddings"
KINDS = ("frames", "audio", "transcript")
# Áudio além disso é cortado: o contexto do modelo é dividido entre as
# modalidades, e o começo do vídeo concentra o que decide a retenção.
MAX_AUDIO_SECONDS = 180


def embedding_path(kind: str, video_id: str) -> Path:
    return MEDIA_EMBEDDINGS_DIR / kind / f"{video_id}.npy"


def done(video_id: str) -> bool:
    return all(embedding_path(k, video_id).is_file() for k in KINDS)


def read_wave(path: Path, max_seconds: float = MAX_AUDIO_SECONDS) -> np.ndarray:
    """Onda mono em float32 (-1 a 1), como o modelo espera para áudio."""
    with wave.open(str(path)) as wav:
        frames = min(wav.getnframes(), int(wav.getframerate() * max_seconds))
        samples = np.frombuffer(wav.readframes(frames), dtype=np.int16)
    return samples.astype(np.float32) / 32768


def _save(kind: str, video_id: str, vector: np.ndarray) -> None:
    path = embedding_path(kind, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, vector.astype(np.float32))
    tmp.replace(path)


class MediaEmbedder:
    """Carrega o EmbeddingGemma 2 completo (texto, imagem e áudio) uma vez."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name, model_kwargs={"torch_dtype": torch.float32}
        )

    def _encode(self, item, **kwargs) -> np.ndarray:
        return self.model.encode(
            item,
            truncate_dim=EMBEDDING_DIM,
            normalize_embeddings=True,
            show_progress_bar=False,
            **kwargs,
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Textos com o prefixo de classificação, iguais aos de
        `features.text.load_encoder` (mesmos pesos; conferido vetor a vetor)."""
        return self._encode(
            list(texts), prompt_name=EMBEDDING_PROMPT, batch_size=64
        ).astype(np.float32)

    def __call__(
        self, video_id: str, frames: list[Path], audio: Path | None, transcript: str
    ) -> None:
        if not embedding_path("frames", video_id).is_file():
            vector = self._encode(
                {"text": "<|image|>" * len(frames), "image": [str(f) for f in frames]}
            )
            _save("frames", video_id, vector)
        if not embedding_path("audio", video_id).is_file():
            if audio is None:
                vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)  # sem trilha
            else:
                vector = self._encode({"text": "<|audio|>", "audio": read_wave(audio)})
            _save("audio", video_id, vector)
        if not embedding_path("transcript", video_id).is_file():
            vector = self._encode(
                [transcript or "(sem fala)"], prompt_name=EMBEDDING_PROMPT
            )[0]
            _save("transcript", video_id, vector)


def load(kind: str, video_ids) -> dict[str, np.ndarray]:
    """Vetores já calculados de um tipo, para os ids pedidos."""
    out = {}
    for video_id in video_ids:
        path = embedding_path(kind, video_id)
        if path.is_file():
            out[video_id] = np.load(path)
    return out
