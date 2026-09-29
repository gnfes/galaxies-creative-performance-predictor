"""Curadoria de canais: valida candidatos pela API e gera `configs/channels.yaml`.

Fluxo:
1. `configs/channel_candidates.yaml` lista os candidatos (handle ou channel_id,
   setor, região, tipo). Um candidato cortado na revisão recebe `exclude` com o
   motivo, para a decisão ficar registrada; `holdout: true` reserva o canal
   para a demonstração, fora do treino, mesmo abaixo do mínimo de Shorts.
2. `validate` resolve cada canal (`channels.list`, 1 unidade) e conta os Shorts
   publicados até a data de corte pela playlist de Shorts do canal (`UUSH…`,
   1 unidade a cada 50 vídeos). Resultado em `data/interim/channel_validation.csv`.
3. `export` grava em `configs/channels.yaml` os canais aprovados e os de
   demonstração.

Uso:
    uv run python -m creative_predictor.collect.channels validate
    uv run python -m creative_predictor.collect.channels search "loreal paris brasil"
    uv run python -m creative_predictor.collect.channels export
"""

import argparse
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Self

import pandas as pd
import yaml
from googleapiclient.errors import HttpError
from pydantic import BaseModel, model_validator

from creative_predictor import config
from creative_predictor.collect.youtube import YouTubeClient, is_not_found

CANDIDATES_PATH = config.CONFIGS_DIR / "channel_candidates.yaml"
CHANNELS_PATH = config.CONFIGS_DIR / "channels.yaml"
VALIDATION_PATH = config.INTERIM_DIR / "channel_validation.csv"

# Vídeos precisam de ao menos 30 dias de exposição na data da coleta.
DEFAULT_CUTOFF = date(2026, 8, 28)
MIN_SHORTS = 50
# A coleta traz os metadados dos 300 Shorts mais recentes de cada canal (base
# do histórico do canal), então a contagem para nesse número e economiza cota.
MAX_SHORTS_PER_CHANNEL = 300
# Só os 120 mais recentes seguem para download e extração multimodal.
MEDIA_SHORTS_PER_CHANNEL = 120
# Engajamento típico do canal: medianas sobre os 50 Shorts mais recentes
# antes do corte (1 unidade de cota em `videos.list`).
ENGAGEMENT_SAMPLE = 50
# Abaixo disso o alvo relativo ao canal fica ruidoso (poucas curtidas por
# Short); o canal continua na base, só marcado com `low_engagement`.
LOW_ENGAGEMENT_LIKES = 10
LOW_ENGAGEMENT_VIEWS = 1_000  # usado quando o canal oculta as curtidas
# Canal sem Short novo nos 180 dias antes do corte é marcado como inativo.
ACTIVE_WINDOW_DAYS = 180


class Sector(StrEnum):
    MODA = "moda"
    BELEZA = "beleza"
    AUTOMOTIVO = "automotivo"


class Region(StrEnum):
    BR = "br"
    GLOBAL = "global"


class Kind(StrEnum):
    MARCA = "marca"
    CRIADOR = "criador"


class Status(StrEnum):
    OK = "ok"
    NOT_FOUND = "canal_nao_encontrado"
    NO_SHORTS = "sem_shorts"
    FEW_SHORTS = "poucos_shorts"


class Candidate(BaseModel):
    handle: str | None = None
    channel_id: str | None = None
    name: str
    sector: Sector
    region: Region
    kind: Kind
    exclude: str | None = None
    holdout: bool = False

    @model_validator(mode="after")
    def _needs_identifier(self) -> Self:
        if not (self.handle or self.channel_id):
            raise ValueError(f"{self.name}: informe handle ou channel_id")
        if self.handle and not self.handle.startswith("@"):
            self.handle = f"@{self.handle}"
        return self

    @property
    def key(self) -> str:
        return self.channel_id or self.handle


class Validation(BaseModel):
    key: str
    name: str
    sector: Sector
    region: Region
    kind: Kind
    holdout: bool = False
    cutoff: date
    status: Status
    channel_id: str | None = None
    handle: str | None = None
    title: str | None = None
    country: str | None = None
    subscribers: int | None = None
    videos_total: int | None = None
    shorts_total: int | None = None
    shorts_until_cutoff: int | None = None
    latest_short: date | None = None
    active: bool | None = None
    median_views: float | None = None
    median_likes: float | None = None


def load_candidates(path: Path = CANDIDATES_PATH) -> list[Candidate]:
    data = yaml.safe_load(path.read_text()) or {}
    return [Candidate(**item) for item in data.get("candidates", [])]


def shorts_playlist_id(channel_id: str) -> str:
    """Playlist de Shorts do canal: `UC…` → `UUSH…` (não documentada pela API)."""
    if not channel_id.startswith("UC"):
        raise ValueError(f"channel_id inesperado: {channel_id}")
    return "UUSH" + channel_id[2:]


def _published_date(item: dict) -> date | None:
    published = item.get("contentDetails", {}).get("videoPublishedAt")
    if not published:  # vídeo privado ou removido
        return None
    return datetime.fromisoformat(published).date()


@dataclass
class ShortsScan:
    total: int = 0
    until_cutoff: int = 0
    latest: date | None = None
    # (data, video_id) dos Shorts publicados até o corte
    before_cutoff: list[tuple[date, str]] = field(default_factory=list)

    def recent_ids(self, n: int = ENGAGEMENT_SAMPLE) -> list[str]:
        return [video_id for _, video_id in sorted(self.before_cutoff)[::-1][:n]]


def count_shorts(
    client: YouTubeClient,
    channel_id: str,
    cutoff: date,
    limit: int = MAX_SHORTS_PER_CHANNEL,
) -> ShortsScan | None:
    """Conta Shorts publicados até `cutoff`, parando em `limit`.

    Devolve None se o canal não tem playlist de Shorts.
    """
    playlist_id = shorts_playlist_id(channel_id)
    scan = ShortsScan()
    page_token = None
    while True:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if page_token:
            params["pageToken"] = page_token
        try:
            response = client.list("playlistItems", **params)
        except HttpError as error:
            if is_not_found(error):
                return None
            raise
        scan.total = response.get("pageInfo", {}).get("totalResults", scan.total)
        for item in response.get("items", []):
            published = _published_date(item)
            if published is None:
                continue
            scan.latest = max(scan.latest, published) if scan.latest else published
            if published <= cutoff:
                scan.before_cutoff.append(
                    (published, item["contentDetails"]["videoId"])
                )
        scan.until_cutoff = min(len(scan.before_cutoff), limit)
        page_token = response.get("nextPageToken")
        if scan.until_cutoff >= limit or not page_token:
            return scan


def median_engagement(
    client: YouTubeClient, video_ids: list[str]
) -> tuple[float | None, float | None]:
    """Medianas de views e curtidas (curtidas ocultas ficam de fora)."""
    if not video_ids:
        return None, None
    response = client.list("videos", part="statistics", id=",".join(video_ids[:50]))
    stats = [item.get("statistics", {}) for item in response.get("items", [])]
    views = [int(s["viewCount"]) for s in stats if "viewCount" in s]
    likes = [int(s["likeCount"]) for s in stats if "likeCount" in s]
    return (
        statistics.median(views) if views else None,
        statistics.median(likes) if likes else None,
    )


def fetch_channel(client: YouTubeClient, candidate: Candidate) -> dict | None:
    params = {"part": "snippet,statistics"}
    if candidate.channel_id:
        params["id"] = candidate.channel_id
    else:
        params["forHandle"] = candidate.handle
    items = client.list("channels", **params).get("items", [])
    return items[0] if items else None


def validate_candidate(
    client: YouTubeClient,
    candidate: Candidate,
    cutoff: date = DEFAULT_CUTOFF,
    min_shorts: int = MIN_SHORTS,
) -> Validation:
    base = {
        "key": candidate.key,
        "name": candidate.name,
        "sector": candidate.sector,
        "region": candidate.region,
        "kind": candidate.kind,
        "holdout": candidate.holdout,
        "cutoff": cutoff,
    }
    channel = fetch_channel(client, candidate)
    if channel is None:
        return Validation(**base, status=Status.NOT_FOUND)

    snippet, stats = channel["snippet"], channel.get("statistics", {})
    info = {
        "channel_id": channel["id"],
        "handle": snippet.get("customUrl"),
        "title": snippet.get("title"),
        "country": snippet.get("country"),
        "subscribers": (
            None if stats.get("hiddenSubscriberCount") else stats.get("subscriberCount")
        ),
        "videos_total": stats.get("videoCount"),
    }
    scan = count_shorts(client, channel["id"], cutoff)
    if scan is None:
        return Validation(**base, **info, status=Status.NO_SHORTS)

    status = Status.OK if scan.until_cutoff >= min_shorts else Status.FEW_SHORTS
    latest = scan.latest
    active = latest is not None and latest >= cutoff - timedelta(ACTIVE_WINDOW_DAYS)
    median_views, median_likes = median_engagement(client, scan.recent_ids())
    return Validation(
        **base,
        **info,
        status=status,
        shorts_total=scan.total,
        shorts_until_cutoff=scan.until_cutoff,
        latest_short=latest,
        active=active,
        median_views=median_views,
        median_likes=median_likes,
    )


def validate(
    candidates: list[Candidate],
    client: YouTubeClient,
    cutoff: date = DEFAULT_CUTOFF,
) -> pd.DataFrame:
    rows = [
        validate_candidate(client, c, cutoff).model_dump(mode="json")
        for c in candidates
        if not c.exclude
    ]
    return pd.DataFrame(rows)


def is_low_engagement(median_views: float | None, median_likes: float | None) -> bool:
    if median_likes is not None and not pd.isna(median_likes):
        return median_likes < LOW_ENGAGEMENT_LIKES
    if median_views is not None and not pd.isna(median_views):
        return median_views < LOW_ENGAGEMENT_VIEWS
    return True


def _channel_entries(rows: pd.DataFrame) -> list[dict]:
    rows = rows.sort_values(["sector", "region", "kind", "name"])
    return [
        {
            "channel_id": row.channel_id,
            "handle": row.handle,
            "name": row.name,
            "sector": row.sector,
            "region": row.region,
            "kind": row.kind,
            "low_engagement": is_low_engagement(row.median_views, row.median_likes),
        }
        for row in rows.itertuples()
    ]


def export_channels(validation: pd.DataFrame) -> dict:
    """Conteúdo de `configs/channels.yaml` a partir da validação.

    `channels`: aprovados (status ok) que entram no treino. `holdout_channels`:
    canais reservados para a demonstração, encontrados e com Shorts, mesmo
    abaixo do mínimo.
    """
    cutoffs = validation["cutoff"].unique()
    if len(cutoffs) != 1:
        raise ValueError(f"validação com mais de uma data de corte: {cutoffs}")
    holdout = validation["holdout"].astype(bool)
    has_shorts = validation["status"].isin([Status.OK, Status.FEW_SHORTS])
    return {
        "cutoff": str(cutoffs[0]),
        "min_shorts": MIN_SHORTS,
        "max_shorts_per_channel": MAX_SHORTS_PER_CHANNEL,
        "media_shorts_per_channel": MEDIA_SHORTS_PER_CHANNEL,
        "channels": _channel_entries(
            validation[~holdout & (validation["status"] == Status.OK)]
        ),
        "holdout_channels": _channel_entries(validation[holdout & has_shorts]),
    }


def summarize(validation: pd.DataFrame) -> pd.DataFrame:
    ok = validation[(validation["status"] == Status.OK) & ~validation["holdout"]]
    return ok.pivot_table(
        index="sector",
        columns="region",
        values="key",
        aggfunc="count",
        fill_value=0,
        margins=True,
        margins_name="total",
    )


def _cmd_validate(args: argparse.Namespace) -> None:
    client = YouTubeClient()
    validation = validate(load_candidates(), client, args.cutoff)
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(VALIDATION_PATH, index=False)

    columns = [
        "name",
        "handle",
        "title",
        "sector",
        "region",
        "kind",
        "status",
        "country",
        "subscribers",
        "shorts_until_cutoff",
        "latest_short",
        "active",
        "median_views",
        "median_likes",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(validation[columns].to_string(index=False))
        print("\nCanais aprovados por setor e região:")
        print(summarize(validation))
    print(f"\nCota gasta nesta execução: {client.quota_used} unidades.")
    print(f"Resultado salvo em {VALIDATION_PATH}")


def search_channels(
    client: YouTubeClient, query: str, region: Region | None = None
) -> pd.DataFrame:
    """Busca canais por termo (100 unidades) e traz handle, país e inscritos."""
    params = {"part": "snippet", "q": query, "type": "channel", "maxResults": 25}
    if region == Region.BR:
        params |= {"regionCode": "BR", "relevanceLanguage": "pt"}
    found = client.list("search", **params).get("items", [])
    ids = [item["snippet"]["channelId"] for item in found]
    if not ids:
        return pd.DataFrame()
    channels = client.list("channels", part="snippet,statistics", id=",".join(ids))
    rows = [
        {
            "channel_id": ch["id"],
            "handle": ch["snippet"].get("customUrl"),
            "title": ch["snippet"]["title"],
            "country": ch["snippet"].get("country"),
            "subscribers": int(ch.get("statistics", {}).get("subscriberCount", 0)),
            "videos": int(ch.get("statistics", {}).get("videoCount", 0)),
        }
        for ch in channels.get("items", [])
    ]
    return pd.DataFrame(rows).sort_values("subscribers", ascending=False)


def _cmd_search(args: argparse.Namespace) -> None:
    client = YouTubeClient()
    found = search_channels(client, args.query, args.region)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(found.to_string(index=False))
    print(f"\nCota gasta: {client.quota_used} unidades.")


def _cmd_export(args: argparse.Namespace) -> None:
    validation = pd.read_csv(VALIDATION_PATH)
    excluded = {c.key for c in load_candidates() if c.exclude}
    validation = validation[~validation["key"].isin(excluded)]
    content = export_channels(validation)
    CHANNELS_PATH.write_text(
        _EXPORT_HEADER + yaml.safe_dump(content, allow_unicode=True, sort_keys=False)
    )
    print(
        f"{len(content['channels'])} canais de treino e "
        f"{len(content['holdout_channels'])} de demonstração gravados em "
        f"{CHANNELS_PATH}"
    )


_EXPORT_HEADER = """\
# Gerado por `python -m creative_predictor.collect.channels export` a partir de
# configs/channel_candidates.yaml; não editar à mão.
#
# channels: canais da base (treino e teste). holdout_channels: canais fora do
# treino, usados na demonstração com métrica real conhecida.
# low_engagement: mediana de curtidas < 10 (ou de views < 1.000, se as
# curtidas estão ocultas) nos 50 Shorts mais recentes antes do corte.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument(
        "--cutoff", type=date.fromisoformat, default=DEFAULT_CUTOFF
    )
    validate_cmd.set_defaults(func=_cmd_validate)

    sub.add_parser("export").set_defaults(func=_cmd_export)

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--region", type=Region, default=None)
    search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
