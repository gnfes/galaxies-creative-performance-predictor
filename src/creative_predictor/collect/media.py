"""Download dos Shorts (vídeo em 360p com áudio e thumbnail) com yt-dlp.

Cada vídeo fica em `data/raw/media/<video_id>/` (`video.mp4` e
`thumbnail.jpg`), e o resultado de cada tentativa em `status.json`, para
nunca baixar de novo nem repetir um vídeo que saiu do ar. Downloads em
sequência, com pausa aleatória entre eles e uma pausa longa quando o YouTube
começa a bloquear (várias falhas seguidas), sem login. A ordem é embaralhada
de forma fixa: se o download parar no meio, o que já baixou representa todos
os canais.

Antes de cada rodada de download, atualizar o yt-dlp, que muda com
frequência para acompanhar o YouTube:
    uv lock --upgrade-package yt-dlp && uv sync

Uso:
    uv run python -m creative_predictor.collect.media [--limit N] [--ids a,b]
"""

import argparse
import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from creative_predictor import config

MEDIA_DIR = config.RAW_DIR / "media"
SLEEP_RANGE = (8.0, 15.0)  # segundos entre downloads, sorteados
COOLDOWN_AFTER = 10  # falhas temporárias seguidas = bloqueio anti-robô
COOLDOWN_SECONDS = 45 * 60
# Vídeo que saiu do ar ou ficou privado não volta: o erro fica registrado.
# Qualquer outro erro (bloqueio anti-robô, rede, 403/429) é passageiro e o
# vídeo é tentado de novo na próxima execução.
PERMANENT_ERRORS = re.compile(
    r"not available|private video|has been removed|terminated|copyright",
    re.IGNORECASE,
)

YDL_OPTIONS = {
    # Menor lado de até 360 px (Shorts são verticais: 360x640), mp4 com áudio.
    # H.264 em vez de AV1/VP9: decodifica no OpenCV instalado pelo pip, no
    # ffmpeg de qualquer sistema e nas APIs de vídeo.
    "format": "bv*+ba/b",
    "format_sort": ["res:360", "vcodec:h264", "acodec:aac", "ext:mp4:m4a"],
    "merge_output_format": "mp4",
    "writethumbnail": True,
    "postprocessors": [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}],
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
    "retries": 3,
}


def media_dir(video_id: str, root: Path = MEDIA_DIR) -> Path:
    return root / video_id


def read_status(video_id: str, root: Path = MEDIA_DIR) -> dict | None:
    path = media_dir(video_id, root) / "status.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _write_status(folder: Path, status: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "status.tmp"
    tmp.write_text(json.dumps(status, ensure_ascii=False))
    tmp.replace(folder / "status.json")


def download(video_id: str, root: Path = MEDIA_DIR, ydl_class=None) -> dict:
    """Baixa um Short; devolve o status (`ok`, `erro` ou `falha_temporaria`).

    Só `ok` e `erro` (vídeo fora do ar) ficam gravados; falha temporária é
    tentada de novo na próxima execução.
    """
    if ydl_class is None:
        from yt_dlp import YoutubeDL as ydl_class
    from yt_dlp.utils import DownloadError

    folder = media_dir(video_id, root)
    folder.mkdir(parents=True, exist_ok=True)
    options = YDL_OPTIONS | {
        "outtmpl": {
            "default": str(folder / "video.%(ext)s"),
            "thumbnail": str(folder / "thumbnail.%(ext)s"),
        }
    }
    started = time.time()
    try:
        with ydl_class(options) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/shorts/{video_id}", download=True
            )
        status = {
            "status": "ok",
            "width": info.get("width"),
            "height": info.get("height"),
            "duration": info.get("duration"),
            "vcodec": info.get("vcodec"),
        }
    except DownloadError as error:
        message = str(error)[:500]
        permanent = PERMANENT_ERRORS.search(message)
        status = {
            "status": "erro" if permanent else "falha_temporaria",
            "erro": message,
        }
    status["video_id"] = video_id
    status["seconds"] = round(time.time() - started, 1)
    if status["status"] != "falha_temporaria":
        _write_status(folder, status)
    return status


def shuffled_sample(shorts: pd.DataFrame) -> pd.DataFrame:
    """Shorts da amostra multimodal numa ordem embaralhada e fixa (por hash
    do id), a mesma em todas as etapas da extração."""
    sample = shorts[shorts["media_sample"]].copy()
    sample["order"] = sample["video_id"].map(
        lambda v: hashlib.sha256(v.encode()).hexdigest()
    )
    return sample.sort_values("order").drop(columns="order")


def download_many(
    video_ids: Iterable[str],
    root: Path = MEDIA_DIR,
    sleep: tuple[float, float] = SLEEP_RANGE,
    ydl_class=None,
    pause: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Baixa em sequência os que ainda não têm status; devolve todos os status.

    Depois de `COOLDOWN_AFTER` falhas temporárias seguidas, pausa por
    `COOLDOWN_SECONDS` antes de continuar; os vídeos que falharam ficam para
    a próxima execução.
    """
    results, streak = [], 0
    for video_id in video_ids:
        status = read_status(video_id, root)
        if status is None:
            status = download(video_id, root, ydl_class)
            print(f"{video_id}: {status['status']} ({status['seconds']} s)")
            streak = streak + 1 if status["status"] == "falha_temporaria" else 0
            if streak >= COOLDOWN_AFTER:
                print(f"{streak} falhas seguidas: pausa de {COOLDOWN_SECONDS} s")
                pause(COOLDOWN_SECONDS)
                streak = 0
            else:
                pause(random.uniform(*sleep))
        results.append(status)
    return pd.DataFrame(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="baixa só os N primeiros")
    parser.add_argument("--ids", help="lista de video_id separada por vírgula")
    args = parser.parse_args(argv)
    if args.ids:
        ids = args.ids.split(",")
    else:
        shorts = pd.read_parquet(config.PROCESSED_DIR / "shorts.parquet")
        ids = shuffled_sample(shorts)["video_id"].tolist()
    if args.limit:
        ids = ids[: args.limit]
    statuses = download_many(ids)
    print(statuses["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
