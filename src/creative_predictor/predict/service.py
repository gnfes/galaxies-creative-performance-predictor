"""Previsão de criativos novos: a função Python que o agente e o MCP expõem.

O criativo entra como mais uma linha no histórico do canal, com a data
planejada e as curtidas desconhecidas, e passa pelas mesmas funções que
montaram a base de treino (`model.dataset.build_table`). Assim as features da
previsão saem do mesmo código do treino.

O histórico do canal vem, nesta ordem: da base coletada (se o canal está
nela), da lista informada na entrada ou da YouTube Data API (pelo id do
canal, com `YOUTUBE_API_KEY`).

    from creative_predictor.predict.service import predict_creative
    prediction = predict_creative(CreativeInput(...))
"""

import functools
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from creative_predictor import config
from creative_predictor.features import frames as fr
from creative_predictor.features.history import HISTORY_GAP_DAYS
from creative_predictor.features.local_time import channel_timezone
from creative_predictor.features.target import MIN_HISTORY, likes_from_target
from creative_predictor.features.text import cache_path
from creative_predictor.model import dataset as ds
from creative_predictor.model.metrics import to_bands
from creative_predictor.predict import bundle as bundle_mod
from creative_predictor.predict import worker
from creative_predictor.predict.schemas import (
    ChannelInfo,
    Comparable,
    CreativeInput,
    Driver,
    Prediction,
    Ranking,
)

NEW_ID = "__criativo__"
N_DRIVERS = 5
N_COMPARABLES = 5

# Nome legível de cada feature, para os fatores da previsão.
FEATURE_LABELS = {
    "duration_s": "duração do vídeo (s)",
    "duration_vs_usual": "duração em relação ao habitual do canal",
    "log_history_median_likes": "tamanho do canal (curtidas medianas)",
    "recent_target_mean": "desempenho recente do canal",
    "recent_target_std": "variação recente do canal",
    "recent_target_max": "melhor desempenho recente do canal",
    "recent_target_count": "posts recentes estabilizados",
    "days_since_prev": "dias desde o Short anterior",
    "posts_prev_30d": "Shorts nos 30 dias anteriores",
    "channel_age_days": "idade do canal (dias)",
    "local_hour": "hora de publicação",
    "local_weekday": "dia da semana",
    "title_chars": "tamanho do título",
    "title_words": "palavras no título",
    "title_hashtags": "hashtags no título",
    "title_emojis": "emojis no título",
    "title_question": "pergunta no título",
    "title_exclamation": "exclamação no título",
    "title_digits": "números no título",
    "title_upper_ratio": "letras maiúsculas no título",
    "description_chars": "tamanho da descrição",
    "description_links": "links na descrição",
    "description_hashtags": "hashtags na descrição",
    "n_tags": "tags",
    "age_days": "idade de referência da previsão (dias)",
    "sector": "setor",
    "region": "região do canal",
    "kind": "tipo de canal",
    "category_id": "categoria do YouTube",
    "definition": "resolução",
}


@functools.cache
def _bundle() -> bundle_mod.ModelBundle:
    return bundle_mod.load()


@functools.cache
def _base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base coletada, quando existe nesta máquina.

    Sem ela (ex.: a Galaxies só com o modelo salvo), a previsão usa o
    histórico informado ou buscado na API, e não há comparáveis.
    """
    shorts_path = config.PROCESSED_DIR / "shorts.parquet"
    channels_path = config.PROCESSED_DIR / "channels.parquet"
    if not (shorts_path.is_file() and channels_path.is_file()):
        return pd.DataFrame(columns=["channel_id"]), pd.DataFrame(
            columns=["channel_id"]
        )
    return pd.read_parquet(shorts_path), pd.read_parquet(channels_path)


def _history_from_api(channel_id: str) -> pd.DataFrame:
    """Shorts recentes do canal pela YouTube Data API (só os com 30+ dias
    entram no histórico de curtidas; os mais novos contam para o ritmo)."""
    from creative_predictor.collect.shorts import fetch_videos, list_shorts, parse_video
    from creative_predictor.collect.youtube import YouTubeClient

    client = YouTubeClient(refresh=True)
    today = datetime.now(UTC).date()
    ids = list_shorts(client, channel_id, today - timedelta(days=365), today, history=0)
    return pd.DataFrame([parse_video(item) for item in fetch_videos(client, ids[:300])])


def channel_frames(
    channel: ChannelInfo, sector: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Histórico de Shorts do canal e a linha do canal, no formato da base."""
    shorts, channels = _base()
    channel_id = channel.channel_id or "__canal__"
    if channel.channel_id and channel.channel_id in set(shorts["channel_id"]):
        history = shorts[shorts["channel_id"] == channel.channel_id]
        info = channels[channels["channel_id"] == channel.channel_id]
        return history, info
    if channel.history:
        history = pd.DataFrame(
            {
                "video_id": [f"h{i}" for i in range(len(channel.history))],
                "published_at": [p.published_at for p in channel.history],
                "likes": pd.array([p.likes for p in channel.history], dtype="Int64"),
                "duration_s": pd.array(
                    [
                        None if p.duration_s is None else round(p.duration_s)
                        for p in channel.history
                    ],
                    dtype="Int64",
                ),
                "title": [p.title for p in channel.history],
            }
        )
    elif channel.channel_id:
        history = _history_from_api(channel.channel_id)
    else:
        history = pd.DataFrame(columns=["video_id", "published_at", "likes"])
    history = history.assign(channel_id=channel_id)
    history["published_at"] = pd.to_datetime(history["published_at"], utc=True)
    created = channel.created_at or (
        history["published_at"].min() if len(history) else datetime.now(UTC)
    )
    created = pd.Timestamp(created)
    created = created.tz_localize("UTC") if created.tzinfo is None else created
    info = pd.DataFrame(
        [
            {
                "channel_id": channel_id,
                "name": channel_id,
                "sector": sector,
                "region": channel.region,
                "kind": channel.kind,
                "country": channel.country,
                "channel_created_at": created.tz_convert("UTC"),
            }
        ]
    )
    return history, info


def _complete(frame: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    return frame


SHORT_DEFAULTS = {
    "description": "",
    "tags": None,
    "category_id": "22",
    "definition": "hd",
    "has_captions": False,
    "made_for_kids": False,
    "licensed_content": False,
    "has_paid_placement": False,
    "views": pd.NA,
    "comments": pd.NA,
    "in_window": True,
    "holdout": False,
    "media_sample": False,
}


# Campos que o YouTube registra na publicação e que o criativo novo não traz:
# vale o hábito do canal (o valor mais comum no histórico dele).
HABIT_FIELDS = (
    "category_id",
    "definition",
    "has_captions",
    "made_for_kids",
    "licensed_content",
    "has_paid_placement",
)


def channel_habits(history: pd.DataFrame) -> dict:
    """Valor mais comum de cada campo de publicação no histórico do canal."""
    habits = {}
    for field in HABIT_FIELDS:
        if field in history and history[field].notna().any():
            habits[field] = history[field].dropna().mode().iloc[0]
    return habits


def planned_time(inp: CreativeInput) -> pd.Timestamp:
    """Data planejada em UTC; sem fuso, vale o horário local do canal."""
    planned = pd.Timestamp(inp.planned_datetime)
    if planned.tzinfo is None:
        tz = channel_timezone(inp.channel.country, inp.channel.region)
        planned = planned.tz_localize(tz)
    return planned.tz_convert("UTC")


def feature_row(inp: CreativeInput) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features do criativo (montadas pelo mesmo código do treino) e o
    histórico do canal usado para elas."""
    history, info = channel_frames(inp.channel, inp.sector)
    duration = inp.duration_s
    if duration is None and inp.creative is not None:
        duration = fr.probe_duration(inp.creative, stream="v:0")
    planned = planned_time(inp)
    new = pd.DataFrame(
        [
            {
                "video_id": NEW_ID,
                "channel_id": info["channel_id"].iloc[0],
                "published_at": planned,
                "title": inp.caption,
                "description": inp.description,
                "duration_s": duration,
                "likes": pd.NA,
            }
            | {k: v for k, v in SHORT_DEFAULTS.items() if k not in ("description",)}
            | channel_habits(history)
            | {"tags": list(inp.tags)}
            | ({"category_id": inp.category_id} if inp.category_id else {})
        ]
    )
    frame = pd.concat([history, new], ignore_index=True)
    frame = _complete(frame, SHORT_DEFAULTS)
    frame["sector"] = inp.sector  # o setor do criativo, informado por quem pede
    for column in ("region", "kind"):
        frame[column] = info[column].iloc[0]
    frame["likes"] = pd.array(frame["likes"], dtype="Int64")
    frame["duration_s"] = pd.array(
        pd.to_numeric(frame["duration_s"]).round(), dtype="Int64"
    )
    table = ds.build_table(frame, info)
    row = table[table["video_id"] == NEW_ID].copy()
    row["age_days"] = ds.PREDICTION_AGE_DAYS
    model = _bundle()
    vector = worker.embed([inp.caption])
    pcs = model.title_pca.transform(vector)
    for i in range(pcs.shape[1]):
        row[f"title_pc{i}"] = pcs[0, i]
    return row, history


def _drivers(model, row: pd.DataFrame) -> list[Driver]:
    contrib = model.contributions(row).iloc[0]
    top = contrib.drop(labels=["age_days"], errors="ignore")
    top = top.reindex(top.abs().sort_values(ascending=False).index)[:N_DRIVERS]
    drivers = []
    for feature, effect in top.items():
        value = row[feature].iloc[0]
        label = FEATURE_LABELS.get(
            feature, "tema do título" if feature.startswith("title_pc") else feature
        )
        shown = "" if feature.startswith("title_pc") else _fmt(value)
        drivers.append(
            Driver(
                feature=feature,
                label=label,
                value=shown,
                effect=round(float(effect), 3),
                direction="aumenta" if effect > 0 else "reduz",
            )
        )
    return drivers


def _fmt(value) -> str:
    if isinstance(value, float | np.floating):
        return "" if np.isnan(value) else f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


@functools.cache
def _comparable_pool() -> pd.DataFrame:
    """Shorts da base com alvo e embedding do título, para comparáveis."""
    shorts, channels = _base()
    if shorts.empty or not cache_path("title").is_file():
        return pd.DataFrame(columns=["channel_id", "sector"])
    vectors = pd.read_parquet(cache_path("title"))
    table = ds.modeling_rows(ds.build_table(shorts, channels))
    table = table[table["video_id"].isin(vectors.index)]
    names = channels.set_index("channel_id")["name"]
    pool = table[
        ["video_id", "channel_id", "sector", "title", "likes", "target"]
    ].copy()
    pool["channel"] = pool["channel_id"].map(names)
    pool["vector"] = list(vectors.loc[pool["video_id"]].to_numpy())
    return pool.reset_index(drop=True)


def comparables(inp: CreativeInput, n: int = N_COMPARABLES) -> list[Comparable]:
    """Shorts reais com título parecido: do próprio canal, se houver, e do setor."""
    pool = _comparable_pool()
    pool = pool[pool["sector"] == inp.sector]
    if pool.empty:
        return []
    query = worker.embed([inp.caption])[0]
    sims = np.vstack(pool["vector"].to_numpy()) @ query
    pool = pool.assign(similarity=sims)
    same = pool[pool["channel_id"] == inp.channel.channel_id].nlargest(
        n // 2 + 1, "similarity"
    )
    other = pool[pool["channel_id"] != inp.channel.channel_id].nlargest(n, "similarity")
    picked = pd.concat([same, other]).head(n)
    return [
        Comparable(
            video_id=r.video_id,
            url=f"https://www.youtube.com/shorts/{r.video_id}",
            title=r.title,
            channel=str(r.channel),
            likes=None if pd.isna(r.likes) else int(r.likes),
            relative_to_channel=round(float(np.exp(r.target)), 2),
            similarity=round(float(r.similarity), 3),
        )
        for r in picked.itertuples()
    ]


@functools.cache
def _sector_medians() -> dict[str, float]:
    shorts, _ = _base()
    if shorts.empty:
        return {}
    window = shorts[shorts["in_window"] & ~shorts["holdout"]]
    return window.groupby("sector")["likes"].median().astype(float).to_dict()


def reference_median(inp: CreativeInput, history: pd.DataFrame) -> tuple[float, str]:
    """Mediana de curtidas quando o canal não tem histórico estabilizado.

    Usa todos os Shorts anteriores do canal (mesmo os com menos de 30 dias,
    cujas curtidas ainda crescem) e, sem nenhum, a mediana do setor na base.
    """
    planned = planned_time(inp)
    if len(history) and "likes" in history:
        before = history[history["published_at"] < planned]["likes"].dropna()
        if len(before):
            return float(
                before.astype(float).median()
            ), "mediana de todos os Shorts do canal"
    sectors = _sector_medians()
    if inp.sector in sectors:
        return float(sectors[inp.sector]), f"mediana do setor de {inp.sector}"
    return 0.0, "falta de referência (canal sem histórico e sem base local)"


def predict_creative(inp: CreativeInput, with_comparables: bool = True) -> Prediction:
    """Prevê as curtidas de um criativo (com ~90 dias) antes da publicação."""
    model = _bundle()
    row, history = feature_row(inp)
    y, low, high = (float(v[0]) for v in model.predict(row))
    median = row["history_median_likes"].iloc[0]
    reasons = []
    if pd.isna(median) or row["history_count"].iloc[0] < MIN_HISTORY:
        reasons.append(
            f"canal com menos de {MIN_HISTORY} Shorts estabilizados "
            f"(publicados há {HISTORY_GAP_DAYS}+ dias)"
        )
    if high - low > model.wide_interval:
        reasons.append("intervalo de previsão mais largo que o habitual")
    if inp.channel.channel_id not in model.channels_in_training:
        reasons.append("canal fora da base de treino")
    if pd.isna(median):
        median, source = reference_median(inp, history)
        reasons.append(f"curtidas absolutas estimadas pela {source}")
    return Prediction(
        creative_id=inp.creative_id,
        predicted_metric=round(float(likes_from_target(y, median)), 1),
        interval_low=round(max(0.0, float(likes_from_target(low, median))), 1),
        interval_high=round(float(likes_from_target(high, median)), 1),
        relative_to_channel=round(float(np.exp(y)), 2),
        band=str(to_bands(np.array([y]), model.band_edges)[0]),
        channel_median_likes=float(median),
        confidence_flag=bool(reasons),
        confidence_reasons=reasons,
        top_drivers=_drivers(model, row),
        explanation=_template_explanation(y, low, high, reasons),
        comparables=comparables(inp) if with_comparables else [],
    )


def _template_explanation(y: float, low: float, high: float, reasons: list[str]) -> str:
    """Explicação curta sem LLM (o agente escreve uma versão mais rica)."""
    text = (
        f"Previsão de {np.exp(y):.2f}× a mediana de curtidas do canal "
        f"(intervalo de 80%: {np.exp(low):.2f}× a {np.exp(high):.2f}×)."
    )
    if reasons:
        text += " Baixa confiança: " + "; ".join(reasons) + "."
    return text


def rank_creatives(inputs: list[CreativeInput]) -> Ranking:
    """Ordena criativos pelo desempenho previsto em relação ao canal."""
    predictions = [predict_creative(i) for i in inputs]
    predictions.sort(key=lambda p: p.relative_to_channel, reverse=True)
    best = predictions[0]
    return Ranking(
        predictions=predictions,
        explanation=f"Melhor previsão: {best.creative_id} "
        f"({best.relative_to_channel:.2f}× a mediana do canal).",
    )
