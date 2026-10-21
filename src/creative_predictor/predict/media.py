"""Extração dos atributos do vídeo de um criativo novo.

Roda as mesmas etapas da base de treino (`features.extract`): frames, áudio,
transcrição, sinais visuais e de áudio, embeddings do EmbeddingGemma 2 e,
se pedido, a rubrica do extrator (`features.vlm.EXTRACTOR`). Usa PyTorch
(EasyOCR e EmbeddingGemma), então só roda no subprocesso de
`predict.worker`, nunca num processo que importou a LightGBM.

Cada vídeo ganha um id pelo conteúdo do arquivo (`criativo_<hash>`), e tudo
fica no mesmo cache da base: o mesmo arquivo não é processado duas vezes.
Um vídeo que já está na base (`data/raw/media/<id>/video.mp4`) usa o próprio
id e reaproveita o que já foi extraído.

A rubrica envia os frames e a transcrição para a API gratuita do Google, que
pode usar os dados para treino: por isso só é pedida quando `use_vlm` é
verdadeiro (criativos públicos ou já liberados).
"""

import hashlib
import threading
from pathlib import Path

import numpy as np

from creative_predictor.collect.media import MEDIA_DIR
from creative_predictor.features import extract, media_embed
from creative_predictor.features.vlm import CANDIDATES, EXTRACTOR, Creative

# A API gratuita levou de 3 a 11 minutos por pedido nos dias de extração.
VLM_TIMEOUT_S = 900


def creative_id(video: Path) -> str:
    """Id do vídeo: o da base, se ele veio de lá, ou um hash do conteúdo."""
    video = Path(video).resolve()
    if video.parent.parent == MEDIA_DIR.resolve():
        return video.parent.name
    digest = hashlib.sha256()
    with video.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return f"criativo_{digest.hexdigest()[:16]}"


Pending = dict[str, tuple[threading.Thread, dict]]


def _rubric(
    creative: Creative, timeout: float, pending: Pending, extractor
) -> str | None:
    """Pede a rubrica (que vai para o cache) com tempo limite.

    Devolve o motivo da falha, ou None. Um pedido que estoura o tempo segue
    rodando e, se responder, fica no cache; um novo pedido do mesmo vídeo
    espera por ele em vez de abrir outro.
    """
    thread, result = pending.get(creative.video_id, (None, {}))
    if thread is None or not thread.is_alive():
        result = {}

        def run() -> None:
            try:
                record = extractor().extract(creative)
                if record["rubric"] is None:
                    result["error"] = record.get("error", "resposta inválida")
            except Exception as error:  # noqa: BLE001 (sem chave, rede, cota)
                result["error"] = str(error)[:300]

        thread = threading.Thread(target=run, daemon=True)
        pending[creative.video_id] = (thread, result)
        thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return f"o extrator não respondeu em {timeout / 60:.0f} min"
    return result.get("error")


class CreativeMedia:
    """Carrega os modelos uma vez e extrai os atributos de cada vídeo."""

    def __init__(self):
        self._visual = extract.VisualExtractor()
        self._embedder: media_embed.MediaEmbedder | None = None
        self._extractor = None
        self._pending: Pending = {}

    @property
    def embedder(self) -> media_embed.MediaEmbedder:
        if self._embedder is None:
            self._embedder = media_embed.MediaEmbedder()
        return self._embedder

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Embeddings de texto com o mesmo modelo (ver `MediaEmbedder`)."""
        return self.embedder.encode_texts(texts)

    def _rubric_extractor(self):
        if self._extractor is None:
            self._extractor = CANDIDATES[EXTRACTOR]()
        return self._extractor

    def __call__(
        self,
        video: Path,
        sector: str,
        title: str,
        use_vlm: bool = False,
        vlm_timeout: float = VLM_TIMEOUT_S,
    ) -> dict:
        """Atributos do vídeo, vetores dos embeddings e avisos.

        `record` tem as mesmas colunas de `data/processed/multimodal.parquet`;
        `vectors`, os embeddings de frames, áudio e transcrição (256-d).
        """
        video = Path(video)
        video_id = creative_id(video)
        frames, audio, transcript = extract.run_local(video_id, self._visual, video)
        if not media_embed.done(video_id):
            self.embedder(video_id, frames, audio, transcript)
        notes = []
        if use_vlm:
            error = _rubric(
                Creative(
                    video_id=video_id,
                    sector=sector,
                    video=video,
                    frames=frames,
                    transcript=transcript,
                    title=title,
                ),
                vlm_timeout,
                self._pending,
                self._rubric_extractor,
            )
            if error:
                notes.append(f"rubrica do extrator indisponível: {error}")
        else:
            notes.append("rubrica do extrator não pedida (use_vlm=false)")
        vectors = {
            kind: media_embed.load(kind, [video_id])[video_id].astype(float).tolist()
            for kind in media_embed.KINDS
        }
        record = extract.media_record(video_id, sector)
        if not use_vlm:  # uma rubrica antiga em cache não entra sem ser pedida
            record = {k: v for k, v in record.items() if not k.startswith("vlm_")}
        return {
            "video_id": video_id,
            "record": _plain(record),
            "vectors": vectors,
            "notes": notes,
        }


def _plain(record: dict) -> dict:
    """Valores em tipos do JSON (o resultado atravessa o subprocesso)."""
    out = {}
    for key, value in record.items():
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out
