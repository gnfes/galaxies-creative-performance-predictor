"""Onde está o vídeo do criativo: caminho local ou URL.

O contrato aceita um caminho na máquina que roda a previsão ou uma URL
`http(s)`. Links do YouTube (Shorts, `watch`, `youtu.be`) são baixados pelo
mesmo código da coleta (`collect.media`, em 360p e H.264), e por isso um Short
que já está na base reaproveita os atributos extraídos dele. Outros links são
baixados direto, com limite de tamanho, para `data/interim/creative_downloads/`.
Cada URL é baixada uma vez.
"""

import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from creative_predictor import config
from creative_predictor.files import sha256, write_atomic

DOWNLOADS_DIR = config.INTERIM_DIR / "creative_downloads"
MAX_BYTES = 300 * 1024 * 1024  # um Short em 1080p passa longe disso
YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "youtu.be"}
YOUTUBE_ID = re.compile(
    r"(?:youtube\.com/(?:shorts/|watch\?(?:.*&)?v=|embed/)|youtu\.be/)([\w-]{11})"
)


def youtube_id(url: str) -> str | None:
    """Id do vídeo quando a URL é do YouTube (pelo endereço, não pelo texto)."""
    host = (urlparse(url).hostname or "").removeprefix("www.")
    if host not in YOUTUBE_HOSTS:
        return None
    match = YOUTUBE_ID.search(url)
    return match.group(1) if match else None


def is_url(creative: str | Path | None) -> bool:
    return creative is not None and bool(re.match(r"https?://", str(creative)))


def _from_youtube(video_id: str) -> Path:
    from creative_predictor.collect import media

    status = media.read_status(video_id) or media.download(video_id)
    if status.get("status") != "ok":
        detail = str(status.get("erro", status.get("status")))[:200]
        raise RuntimeError(f"vídeo do YouTube indisponível ({detail})")
    return media.media_dir(video_id) / "video.mp4"


def _from_link(url: str, root: Path) -> Path:
    suffix = Path(urlparse(url).path).suffix.lower() or ".mp4"
    path = root / f"{sha256(url.encode())[:16]}{suffix}"
    if path.is_file():
        return path

    def copy(tmp: Path) -> None:
        size = 0
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError(f"vídeo maior que {MAX_BYTES // 2**20} MB")
                out.write(chunk)

    try:
        return write_atomic(path, copy)
    finally:
        path.with_suffix(".tmp").unlink(missing_ok=True)


def local_video(creative: str | Path | None, root: Path = DOWNLOADS_DIR) -> Path | None:
    """Caminho local do vídeo; baixa antes quando `creative` é uma URL."""
    if creative is None:
        return None
    if not is_url(creative):
        return Path(creative)
    url = str(creative)
    video_id = youtube_id(url)
    return _from_youtube(video_id) if video_id else _from_link(url, root)
