"""Curadoria de canais: valida candidatos pela API e gera `configs/channels.yaml`.

Fluxo:
1. `configs/channel_candidates.yaml` lista os candidatos (handle ou channel_id,
   setor, região, tipo). Um candidato cortado na revisão recebe `exclude` com o
   motivo, para a decisão ficar registrada; `holdout: true` reserva o canal
   para fora do treino (demonstração e avaliação de canais com poucos Shorts),
   mesmo abaixo do mínimo.
2. `validate` resolve cada canal (`channels.list`, 1 unidade) e conta os Shorts
   publicados dentro da janela de coleta pela playlist de Shorts do canal
   (`UUSH…`, 1 unidade a cada 50 vídeos). Resultado em
   `data/interim/channel_validation.csv`.
3. `export` grava em `configs/channels.yaml` os canais de treino e os
   reservados.

Uso:
    uv run python -m creative_predictor.collect.channels validate [--refresh]
    uv run python -m creative_predictor.collect.channels search "maquiagem" --br
    uv run python -m creative_predictor.collect.channels export
"""

import argparse
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, Self

import pandas as pd
import yaml
from googleapiclient.errors import HttpError
from pydantic import BaseModel, model_validator

from creative_predictor import config
from creative_predictor.collect.youtube import YouTubeClient, is_not_found

CANDIDATES_PATH = config.CONFIGS_DIR / "channel_candidates.yaml"
CHANNELS_PATH = config.CONFIGS_DIR / "channels.yaml"
VALIDATION_PATH = config.INTERIM_DIR / "channel_validation.csv"

# Janela de coleta. O fim deixa ao menos 30 dias de exposição até a coleta. O
# início é 15/10/2024, quando os Shorts passaram a aceitar até 3 minutos: assim
# todos os vídeos da base seguem a mesma regra da plataforma, e ficam de fora
# vídeos verticais antigos que o YouTube reclassificou como Shorts.
DEFAULT_WINDOW_START = date(2024, 10, 15)
DEFAULT_CUTOFF = date(2026, 8, 28)
MIN_SHORTS = 50  # Shorts dentro da janela
# Até 120 Shorts por canal seguem para download e extração multimodal,
# distribuídos ao longo da janela (e não os mais recentes), para cada canal
# cobrir o mesmo período no split temporal.
MEDIA_SHORTS_PER_CHANNEL = 120
# Engajamento típico do canal: medianas sobre os 50 Shorts mais recentes da
# janela (1 unidade de cota em `videos.list`).
ENGAGEMENT_SAMPLE = 50
# Abaixo disso o alvo relativo ao canal fica ruidoso (poucas curtidas por
# Short); o canal continua na base, só marcado com `low_engagement`.
LOW_ENGAGEMENT_LIKES = 10
LOW_ENGAGEMENT_VIEWS = 1_000  # usado quando o canal oculta as curtidas
# Curtidas abaixo de 0,3% das views indicam Short impulsionado como anúncio
# (views pagas). `paid_share` é a fração da amostra nessa situação; acima de
# 30%, as views do canal medem mais a verba de mídia do que o conteúdo.
PAID_LIKE_RATE = 0.003
MAX_PAID_SHARE = 0.3
# O alvo do modelo são as curtidas: canal que oculta as curtidas na maior parte
# dos Shorts não tem alvo e fica fora da base.
MAX_HIDDEN_LIKES_SHARE = 0.5


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
    PAID_MEDIA = "midia_paga"
    HIDDEN_LIKES = "curtidas_ocultas"


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
        # Handles do YouTube não diferenciam maiúsculas de minúsculas.
        return self.channel_id or self.handle.lower()


class Validation(BaseModel):
    key: str
    name: str
    sector: Sector
    region: Region
    kind: Kind
    holdout: bool = False
    window_start: date
    cutoff: date
    status: Status
    channel_id: str | None = None
    handle: str | None = None
    title: str | None = None
    country: str | None = None
    subscribers: int | None = None
    videos_total: int | None = None
    shorts_total: int | None = None
    shorts_in_window: int | None = None
    latest_short: date | None = None
    median_views: float | None = None
    median_likes: float | None = None
    paid_share: float | None = None
    hidden_likes_share: float | None = None


def load_candidates(path: Path | None = None) -> list[Candidate]:
    data = yaml.safe_load((path or CANDIDATES_PATH).read_text()) or {}
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
    latest: date | None = None
    # (data, video_id) dos Shorts publicados dentro da janela
    in_window: list[tuple[date, str]] = field(default_factory=list)

    def recent_ids(self, n: int = ENGAGEMENT_SAMPLE) -> list[str]:
        return [video_id for _, video_id in sorted(self.in_window)[::-1][:n]]


def count_shorts(
    client: YouTubeClient,
    channel_id: str,
    window_start: date = DEFAULT_WINDOW_START,
    cutoff: date = DEFAULT_CUTOFF,
) -> ShortsScan | None:
    """Conta os Shorts publicados entre `window_start` e `cutoff`.

    A playlist vem do mais recente para o mais antigo, então a varredura para
    na primeira página que já passou do início da janela. Devolve None se o
    canal não tem playlist de Shorts.
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
            # 404 na primeira página = canal sem playlist de Shorts. Numa página
            # seguinte é falha da paginação, e o erro não pode virar "sem Shorts".
            if is_not_found(error) and page_token is None:
                return None
            raise
        scan.total = response.get("pageInfo", {}).get("totalResults", scan.total)
        reached_start = False
        for item in response.get("items", []):
            published = _published_date(item)
            if published is None:
                continue
            scan.latest = max(scan.latest, published) if scan.latest else published
            if published < window_start:
                reached_start = True
            elif published <= cutoff:
                scan.in_window.append((published, item["contentDetails"]["videoId"]))
        page_token = response.get("nextPageToken")
        if reached_start or not page_token:
            return scan


class Engagement(NamedTuple):
    median_views: float | None = None
    median_likes: float | None = None
    paid_share: float | None = None
    hidden_likes_share: float | None = None


def measure_engagement(client: YouTubeClient, video_ids: list[str]) -> Engagement:
    """Medianas de views e curtidas, fração de Shorts com cara de anúncio e
    fração de Shorts com curtidas ocultas.

    Curtidas ocultas ficam de fora das medianas e da fração de anúncios.
    """
    if not video_ids:
        return Engagement()
    response = client.list("videos", part="statistics", id=",".join(video_ids[:50]))
    stats = [item.get("statistics", {}) for item in response.get("items", [])]
    views = [int(s["viewCount"]) for s in stats if "viewCount" in s]
    likes = [int(s["likeCount"]) for s in stats if "likeCount" in s]
    rates = [
        int(s["likeCount"]) / int(s["viewCount"])
        for s in stats
        if "likeCount" in s and int(s.get("viewCount", 0)) > 0
    ]
    return Engagement(
        statistics.median(views) if views else None,
        statistics.median(likes) if likes else None,
        sum(r < PAID_LIKE_RATE for r in rates) / len(rates) if rates else None,
        1 - len(likes) / len(stats) if stats else None,
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
    window_start: date = DEFAULT_WINDOW_START,
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
        "window_start": window_start,
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
    scan = count_shorts(client, channel["id"], window_start, cutoff)
    if scan is None:
        return Validation(**base, **info, status=Status.NO_SHORTS)

    in_window = len(scan.in_window)
    engagement = measure_engagement(client, scan.recent_ids())
    if in_window < min_shorts:
        status = Status.FEW_SHORTS
    elif (engagement.hidden_likes_share or 0) > MAX_HIDDEN_LIKES_SHARE:
        status = Status.HIDDEN_LIKES
    elif (engagement.paid_share or 0) > MAX_PAID_SHARE:
        status = Status.PAID_MEDIA
    else:
        status = Status.OK
    return Validation(
        **base,
        **info,
        status=status,
        shorts_total=scan.total,
        shorts_in_window=in_window,
        latest_short=scan.latest,
        **engagement._asdict(),
    )


def validate(
    candidates: list[Candidate],
    client: YouTubeClient,
    window_start: date = DEFAULT_WINDOW_START,
    cutoff: date = DEFAULT_CUTOFF,
) -> pd.DataFrame:
    if window_start > cutoff:
        raise ValueError(f"janela invertida: início {window_start} > corte {cutoff}")
    rows = [
        validate_candidate(client, c, window_start, cutoff).model_dump(mode="json")
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
    rows = rows.sort_values(
        ["sector", "region", "kind", "name"], key=lambda col: col.str.lower()
    )
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


def apply_candidates(
    validation: pd.DataFrame, candidates: list[Candidate]
) -> pd.DataFrame:
    """Alinha a validação salva com a lista de candidatos atual.

    Candidatos retirados da lista ou marcados com `exclude` saem, e `holdout`
    vem da lista, para uma revisão feita depois do `validate` valer no export.
    """
    current = {c.key: c for c in candidates if not c.exclude}
    aligned = validation[validation["key"].isin(current)].copy()
    aligned["holdout"] = [current[key].holdout for key in aligned["key"]]
    return aligned


def export_channels(validation: pd.DataFrame) -> dict:
    """Conteúdo de `configs/channels.yaml` a partir da validação.

    `channels`: aprovados (status ok) que entram no treino. `holdout_channels`:
    canais reservados, fora do treino, com ao menos um Short na janela e
    curtidas públicas (sem elas não há métrica real para comparar).
    """
    window = validation[["window_start", "cutoff"]].drop_duplicates()
    if len(window) != 1:
        raise ValueError(f"validação com mais de uma janela: {window.values.tolist()}")
    holdout = validation["holdout"].astype(bool)
    has_shorts = validation["shorts_in_window"].fillna(0) >= 1
    hidden = validation["hidden_likes_share"].fillna(0) > MAX_HIDDEN_LIKES_SHARE
    return {
        "window_start": str(window["window_start"].iloc[0]),
        "cutoff": str(window["cutoff"].iloc[0]),
        "min_shorts": MIN_SHORTS,
        "media_shorts_per_channel": MEDIA_SHORTS_PER_CHANNEL,
        "channels": _channel_entries(
            validation[~holdout & (validation["status"] == Status.OK)]
        ),
        "holdout_channels": _channel_entries(
            validation[holdout & has_shorts & ~hidden]
        ),
    }


def summarize(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    ok = validation[(validation["status"] == Status.OK) & ~validation["holdout"]]
    if ok.empty:
        return pd.DataFrame()
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
    client = YouTubeClient(refresh=args.refresh)
    validation = validate(load_candidates(), client, args.window_start, args.cutoff)
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
        "shorts_in_window",
        "latest_short",
        "median_views",
        "median_likes",
        "paid_share",
        "hidden_likes_share",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(validation[columns].to_string(index=False))
        print("\nCanais aprovados por setor e região:")
        print(summarize(validation))
    print(f"\nCota gasta nesta execução: {client.quota_used} unidades.")
    print(f"Resultado salvo em {VALIDATION_PATH}")


def search_channels(
    client: YouTubeClient, query: str, br: bool = False
) -> pd.DataFrame:
    """Busca canais por termo (100 unidades) e traz handle, país e inscritos.

    Com `br=True`, prioriza canais do Brasil em português.
    """
    params = {"part": "snippet", "q": query, "type": "channel", "maxResults": 25}
    if br:
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
    if not rows:  # canais removidos entre a busca e o channels.list
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("subscribers", ascending=False)


def _cmd_search(args: argparse.Namespace) -> None:
    client = YouTubeClient()
    found = search_channels(client, args.query, args.br)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(found.to_string(index=False))
    print(f"\nCota gasta: {client.quota_used} unidades.")


def _cmd_export(args: argparse.Namespace) -> None:
    validation = pd.read_csv(VALIDATION_PATH)
    missing = {"window_start", "shorts_in_window", "hidden_likes_share"} - set(
        validation.columns
    )
    if missing:
        raise SystemExit(
            f"{VALIDATION_PATH} não tem {sorted(missing)}: é de uma versão anterior "
            "da validação. Rode `validate` de novo antes do `export`."
        )
    validation = apply_candidates(validation, load_candidates())
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
# treino, usados na demonstração (métrica real conhecida) e para avaliar o
# modelo em canais com poucos Shorts.
# window_start/cutoff: janela de publicação dos Shorts coletados.
# media_shorts_per_channel: Shorts por canal que seguem para a extração
# multimodal, distribuídos ao longo da janela.
# low_engagement: mediana de curtidas < 10 nos 50 Shorts mais recentes da
# janela. Canais que ocultam as curtidas na maior parte dos Shorts ficam de fora: as
# curtidas são o alvo do modelo.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument(
        "--window-start", type=date.fromisoformat, default=DEFAULT_WINDOW_START
    )
    validate_cmd.add_argument(
        "--cutoff", type=date.fromisoformat, default=DEFAULT_CUTOFF
    )
    validate_cmd.add_argument(
        "--refresh", action="store_true", help="ignora o cache e consulta a API"
    )
    validate_cmd.set_defaults(func=_cmd_validate)

    sub.add_parser("export").set_defaults(func=_cmd_export)

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument(
        "--br", action="store_true", help="prioriza canais do Brasil em português"
    )
    search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
