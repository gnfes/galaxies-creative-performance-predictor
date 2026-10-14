"""Coleta de metadados e métricas dos Shorts dos canais de `configs/channels.yaml`.

Para cada canal (treino e demonstração):
1. Lista a playlist de Shorts (`UUSH…`, 1 unidade a cada 50 vídeos): todos os
   Shorts da janela e os `HISTORY_BEFORE_WINDOW` anteriores a ela, que servem
   só de histórico do canal (mediana de curtidas dos posts anteriores).
2. Busca os detalhes em `videos.list` (1 unidade a cada 50 vídeos): texto,
   duração, idioma, categoria, publicidade declarada, views, curtidas e
   comentários na data da coleta.
3. Busca as estatísticas do canal na data da coleta (`channels.list`).
4. Marca em `media_sample` até `media_shorts_per_channel` Shorts por canal,
   distribuídos ao longo da janela, que seguem para a extração multimodal.

As respostas brutas ficam em `data/raw/youtube_collection/<data da coleta>/`:
rodar de novo no mesmo dia reaproveita o cache, e uma coleta em outra data
consulta a API de novo (estatísticas novas).

Saída em `data/processed/`: `shorts.parquet` (um Short por linha) e
`channels.parquet` (um canal por linha).

Uso:
    uv run python -m creative_predictor.collect.shorts [--collected-on AAAA-MM-DD]
"""

import argparse
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from googleapiclient.errors import HttpError

from creative_predictor import config
from creative_predictor.collect.channels import CHANNELS_PATH, shorts_playlist_id
from creative_predictor.collect.youtube import YouTubeClient, is_not_found

COLLECTION_CACHE_DIR = config.RAW_DIR / "youtube_collection"
SHORTS_PATH = config.PROCESSED_DIR / "shorts.parquet"
CHANNELS_TABLE_PATH = config.PROCESSED_DIR / "channels.parquet"

# Shorts anteriores à janela guardados como histórico: cobrem a mediana dos
# posts anteriores (até 20 posts) dos primeiros Shorts da janela.
HISTORY_BEFORE_WINDOW = 50
# Shorts aceitam até 3 minutos desde 15/10/2024; acima disso é exceção da
# playlist e o vídeo não entra na amostra.
MAX_SHORT_SECONDS = 180
# Métricas só são comparáveis depois de estabilizadas.
MIN_AGE_DAYS = 30
BATCH = 50  # máximo de ids por chamada de `videos.list` e `channels.list`


def list_shorts(
    client: YouTubeClient,
    channel_id: str,
    window_start: date,
    cutoff: date,
    history: int = HISTORY_BEFORE_WINDOW,
) -> list[str]:
    """Ids dos Shorts da janela e dos `history` anteriores a ela.

    A playlist vem do mais recente para o mais antigo; a varredura para quando
    já juntou `history` Shorts anteriores à janela. Shorts depois do corte
    ficam de fora (não são histórico de nenhum Short da janela).
    """
    playlist_id = shorts_playlist_id(channel_id)
    ids: list[str] = []
    before = 0
    page_token = None
    while True:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if page_token:
            params["pageToken"] = page_token
        try:
            response = client.list("playlistItems", **params)
        except HttpError as error:
            if is_not_found(error) and page_token is None:
                return []
            raise
        for item in response.get("items", []):
            details = item.get("contentDetails", {})
            published = details.get("videoPublishedAt")
            if not published:  # vídeo privado ou removido
                continue
            day = datetime.fromisoformat(published).date()
            if day > cutoff:
                continue
            if day < window_start:
                if before >= history:
                    return ids
                before += 1
            ids.append(details["videoId"])
        page_token = response.get("nextPageToken")
        if not page_token:
            return ids


def fetch_videos(client: YouTubeClient, video_ids: list[str]) -> list[dict]:
    parts = (
        "snippet,contentDetails,statistics,status,topicDetails,"
        "paidProductPlacementDetails"
    )
    items = []
    for i in range(0, len(video_ids), BATCH):
        batch = ",".join(video_ids[i : i + BATCH])
        items += client.list("videos", part=parts, id=batch).get("items", [])
    return items


_DURATION = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def parse_duration(value: str | None) -> int | None:
    """Duração ISO 8601 da API (ex.: `PT1M5S`) em segundos."""
    match = _DURATION.fullmatch(value or "")
    if not match or not any(match.groups()):  # vazio, "P" ou "PT"
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _count(stats: dict, key: str) -> int | None:
    return int(stats[key]) if key in stats else None


def _thumbnail(snippet: dict) -> str | None:
    thumbnails = snippet.get("thumbnails", {})
    for size in ("maxres", "standard", "high", "medium", "default"):
        if size in thumbnails:
            return thumbnails[size]["url"]
    return None


def parse_video(item: dict) -> dict:
    """Uma linha da tabela de Shorts a partir de um item de `videos.list`.

    Curtidas e comentários ocultos ficam nulos (nunca zero).
    """
    snippet = item["snippet"]
    details = item.get("contentDetails", {})
    stats = item.get("statistics", {})
    status = item.get("status", {})
    return {
        "video_id": item["id"],
        "channel_id": snippet["channelId"],
        "published_at": pd.Timestamp(snippet["publishedAt"]),
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "tags": snippet.get("tags", []),
        "category_id": snippet.get("categoryId"),
        "default_language": snippet.get("defaultLanguage"),
        "default_audio_language": snippet.get("defaultAudioLanguage"),
        "duration_s": parse_duration(details.get("duration")),
        "definition": details.get("definition"),
        "has_captions": details.get("caption") == "true",
        "licensed_content": details.get("licensedContent"),
        "made_for_kids": status.get("madeForKids"),
        "topic_categories": item.get("topicDetails", {}).get("topicCategories", []),
        "has_paid_placement": item.get("paidProductPlacementDetails", {}).get(
            "hasPaidProductPlacement"
        ),
        "views": _count(stats, "viewCount"),
        "likes": _count(stats, "likeCount"),
        "comments": _count(stats, "commentCount"),
        "thumbnail_url": _thumbnail(snippet),
        "url": f"https://www.youtube.com/shorts/{item['id']}",
    }


def fetch_channels(client: YouTubeClient, channel_ids: list[str]) -> list[dict]:
    """Estatísticas dos canais na data da coleta."""
    rows = []
    for i in range(0, len(channel_ids), BATCH):
        batch = ",".join(channel_ids[i : i + BATCH])
        response = client.list("channels", part="snippet,statistics", id=batch)
        for item in response.get("items", []):
            snippet, stats = item["snippet"], item.get("statistics", {})
            rows.append(
                {
                    "channel_id": item["id"],
                    "title": snippet.get("title"),
                    "country": snippet.get("country"),
                    "channel_created_at": pd.Timestamp(snippet["publishedAt"]),
                    "subscribers": None
                    if stats.get("hiddenSubscriberCount")
                    else _count(stats, "subscriberCount"),
                    "channel_views": _count(stats, "viewCount"),
                    "channel_videos": _count(stats, "videoCount"),
                }
            )
    return rows


def sample_across_window(
    published: pd.Series, n: int, window_start: date, cutoff: date
) -> pd.Index:
    """Até `n` Shorts espalhados no tempo ao longo da janela.

    Divide a janela em `n` instantes igualmente espaçados e, para cada um,
    escolhe o Short ainda não escolhido mais próximo. Assim um canal que posta
    muito num mês não concentra a amostra nesse mês, e todos os canais cobrem
    o mesmo período (importante para o split temporal). Determinístico.
    """
    if len(published) <= n:
        return published.index
    start = pd.Timestamp(window_start, tz="UTC")
    end = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
    times = published.to_numpy(dtype="datetime64[ns]").astype("int64")
    targets = np.linspace(start.value, end.value, n)
    available = np.ones(len(times), dtype=bool)
    chosen = []
    for target in targets:
        distance = np.where(available, np.abs(times - target), np.iinfo("int64").max)
        best = int(np.argmin(distance))
        available[best] = False
        chosen.append(best)
    return published.index[sorted(chosen)]


def _load_channels_config(path: Path = CHANNELS_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def _channel_rows(cfg: dict) -> pd.DataFrame:
    rows = [ch | {"holdout": False} for ch in cfg["channels"]]
    rows += [ch | {"holdout": True} for ch in cfg.get("holdout_channels", [])]
    return pd.DataFrame(rows)


def collect(
    client: YouTubeClient, cfg: dict, collected_on: date, progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coleta os Shorts e as estatísticas dos canais do `channels.yaml`."""
    window_start = date.fromisoformat(str(cfg["window_start"]))
    cutoff = date.fromisoformat(str(cfg["cutoff"]))
    n_media = int(cfg["media_shorts_per_channel"])
    if collected_on - cutoff < timedelta(days=MIN_AGE_DAYS):
        raise ValueError(
            f"coleta em {collected_on} deixa menos de {MIN_AGE_DAYS} dias desde "
            f"o corte {cutoff}: métricas ainda não estabilizadas"
        )

    channels = _channel_rows(cfg)
    stats = fetch_channels(client, channels["channel_id"].tolist())
    if stats:
        channels = channels.merge(pd.DataFrame(stats), on="channel_id", how="left")
    channels["collected_on"] = collected_on

    frames = []
    for n, ch in enumerate(channels.itertuples(), 1):
        ids = list_shorts(client, ch.channel_id, window_start, cutoff)
        rows = [parse_video(item) for item in fetch_videos(client, ids)]
        frames.append(pd.DataFrame(rows))
        if progress:
            print(f"[{n}/{len(channels)}] {ch.name}: {len(rows)} Shorts")
    shorts = pd.concat(frames, ignore_index=True)
    if shorts.empty:
        raise ValueError("nenhum Short encontrado nos canais do channels.yaml")
    shorts = shorts.drop_duplicates("video_id").sort_values(
        ["channel_id", "published_at"], ignore_index=True
    )

    meta = channels.set_index("channel_id")[["sector", "region", "kind", "holdout"]]
    shorts = shorts.join(meta, on="channel_id")
    day = shorts["published_at"].dt.date
    shorts["in_window"] = (day >= window_start) & (day <= cutoff)
    shorts["collected_on"] = collected_on
    shorts["age_days"] = [(collected_on - d).days for d in day]
    shorts["media_sample"] = False
    eligible = (
        shorts["in_window"]
        & shorts["likes"].notna()
        & (shorts["duration_s"].fillna(0) <= MAX_SHORT_SECONDS)
    )
    for _, group in shorts[eligible].groupby("channel_id"):
        picked = sample_across_window(
            group["published_at"], n_media, window_start, cutoff
        )
        shorts.loc[picked, "media_sample"] = True
    for col in ("views", "likes", "comments", "duration_s"):
        shorts[col] = shorts[col].astype("Int64")
    return shorts, channels


def summarize(shorts: pd.DataFrame) -> pd.DataFrame:
    window = shorts[shorts["in_window"]]
    return window.groupby(["holdout", "sector", "region"]).agg(
        canais=("channel_id", "nunique"),
        shorts=("video_id", "size"),
        amostra=("media_sample", "sum"),
        curtidas_ocultas=("likes", lambda s: s.isna().sum()),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--collected-on",
        type=date.fromisoformat,
        default=datetime.now(UTC).date(),
        help="data da coleta (define a pasta do cache); padrão: hoje",
    )
    args = parser.parse_args(argv)

    # A data nomeia o cache e entra na tabela (idade dos vídeos). Uma data
    # passada só vale para reaproveitar o cache daquele dia: sem ele, a API
    # devolveria estatísticas de hoje com a data errada.
    cache_dir = COLLECTION_CACHE_DIR / args.collected_on.isoformat()
    if args.collected_on != datetime.now(UTC).date() and not cache_dir.is_dir():
        raise SystemExit(
            f"Sem cache de {args.collected_on} em {cache_dir}: uma coleta nova "
            "usa a data de hoje (omita --collected-on)."
        )
    client = YouTubeClient(cache_dir=cache_dir)
    shorts, channels = collect(client, _load_channels_config(), args.collected_on)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shorts.to_parquet(SHORTS_PATH, index=False)
    channels.to_parquet(CHANNELS_TABLE_PATH, index=False)
    with pd.option_context("display.width", 200):
        print(summarize(shorts))
    print(f"\nCota gasta nesta execução: {client.quota_used} unidades.")
    print(f"Salvo em {SHORTS_PATH} e {CHANNELS_TABLE_PATH}")


if __name__ == "__main__":
    main()
