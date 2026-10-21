"""Previsão de criativos novos: a função Python que o agente e o MCP expõem.

O criativo entra como mais uma linha no histórico do canal, com a data
planejada e as curtidas desconhecidas, e passa pelas mesmas funções que
montaram a base de treino (`model.dataset.build_table`). Assim as features da
previsão saem do mesmo código do treino.

O histórico do canal vem, nesta ordem: da base coletada (se o canal está
nela), da lista informada na entrada ou da YouTube Data API (pelo id do
canal, com `YOUTUBE_API_KEY`).

Com o arquivo do vídeo, o subprocesso de `predict.worker` extrai os mesmos
atributos da base (sinais do vídeo e do áudio, embeddings e, se pedida, a
rubrica do extrator), e as colunas são montadas pelas mesmas funções do
treino (`model.multimodal`). Sem vídeo, esses campos ficam vazios, como na
maior parte da base de treino.

    from creative_predictor.predict.service import predict_creative
    prediction = predict_creative(CreativeInput(...))
"""

import functools
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

from creative_predictor import config
from creative_predictor.features import frames as fr
from creative_predictor.features import history as hist
from creative_predictor.features.history import HISTORY_GAP_DAYS
from creative_predictor.features.local_time import channel_timezone
from creative_predictor.features.rubric import RUBRICS
from creative_predictor.features.target import MIN_HISTORY, likes_from_target
from creative_predictor.features.text import cache_path
from creative_predictor.model import dataset as ds
from creative_predictor.model import multimodal as mm
from creative_predictor.model.metrics import to_bands
from creative_predictor.predict import bundle as bundle_mod
from creative_predictor.predict import source, worker
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
    "brightness_mean": "brilho",
    "contrast_mean": "contraste",
    "saturation_mean": "saturação",
    "colorfulness_mean": "colorido",
    "face_frames_share": "frames com rosto",
    "face_max_count": "rostos no mesmo frame",
    "face_area_max": "tamanho do maior rosto",
    "face_in_hook": "rosto nos 3 primeiros segundos",
    "text_frames_share": "frames com texto na tela",
    "text_area_mean": "área de texto na tela",
    "text_words_mean": "palavras na tela",
    "text_in_hook": "texto na tela nos 3 primeiros segundos",
    "motion_mean": "movimento",
    "motion_std": "variação do movimento",
    "motion_hook": "movimento nos 3 primeiros segundos",
    "cuts": "cortes de cena",
    "cuts_per_10s": "cortes a cada 10 s",
    "cuts_in_hook": "cortes nos 3 primeiros segundos",
    "first_cut_s": "segundo do primeiro corte",
    "audio_db_mean": "volume do áudio",
    "audio_db_std": "variação do volume",
    "silence_share": "fração de silêncio",
    "speech_share": "fração com fala",
    "speech_words": "palavras faladas",
    "speech_words_per_second": "palavras faladas por segundo",
    "speech_language": "idioma da fala",
}
# Componentes de PCA de um mesmo embedding viram um fator só (a soma das
# contribuições): sozinhos, não têm leitura.
EMBEDDING_GROUPS = {
    "title_pc": "tema do título",
    "frames_pc": "visual geral dos frames",
    "audio_pc": "trilha de áudio",
    "transcript_pc": "conteúdo da fala",
}
RUBRIC_LABELS = {
    "hook_primeiros_3s": "gancho nos 3 primeiros segundos",
    "forca_hook_0a10": "força do gancho (0–10)",
    "tem_rosto": "tem rosto",
    "close_up_produto": "close no produto",
    "before_after_ou_transformacao": "antes e depois / transformação",
    "try_on_ou_demonstracao": "prova ou demonstração",
    "texto_na_tela": "texto na tela",
    "densidade_texto_0a10": "densidade de texto (0–10)",
    "ritmo_cortes_0a10": "ritmo de cortes (0–10)",
    "energia_musica_0a10": "energia da música (0–10)",
    "tom_narracao": "tom da narração",
    "brilho_saturacao_0a10": "brilho e saturação (0–10)",
    "cor_dominante": "cor dominante",
    "estetica_geral_0a10": "estética geral (0–10)",
    "clareza_beneficio_produto_0a10": "clareza do benefício do produto (0–10)",
    "elementos_video_top5": "elementos do vídeo",
    "elementos_audio_top5": "elementos do áudio",
    "potencial_viral_0a10": "potencial viral segundo o extrator (0–10)",
    "assunto_no_setor": "assunto do setor",
    "swatch": "swatch",
    "aplicacao_na_pele": "aplicação na pele",
    "corpo_inteiro": "look de corpo inteiro",
    "styling": "styling",
    "close_do_carro": "close do carro",
    "som_do_motor": "som do motor",
    "pov_direcao": "ponto de vista de quem dirige",
    "interior": "interior do carro",
}
FEATURE_LABELS |= {f"vlm_{k}": v for k, v in RUBRIC_LABELS.items()}
# Campos 0/1: os booleanos da rubrica (de todos os setores) e os do gancho.
YES_NO = {"face_in_hook", "text_in_hook"} | {
    f"vlm_{name}"
    for rubric in RUBRICS.values()
    for name, info in rubric.model_fields.items()
    if info.annotation is bool
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


class ChannelFrames(NamedTuple):
    """Histórico do canal, a linha do canal e, se o histórico não pôde ser
    buscado, o motivo (vira motivo de baixa confiança na previsão)."""

    history: pd.DataFrame
    info: pd.DataFrame
    warning: str | None = None


def _api_failure(error: Exception) -> str:
    """Motivo curto para a falha da API do YouTube. Nunca usa o texto de um
    `HttpError`, que traz a URL do pedido com a chave da API."""
    status = getattr(getattr(error, "resp", None), "status", None)
    if status:
        detail = f"erro {status} da API"
    elif isinstance(error, RuntimeError):  # chave ausente (texto do `config`)
        detail = str(error).rstrip(".")[:200]
    else:
        detail = f"falha de rede: {type(error).__name__}"
    return (
        f"histórico do canal indisponível na API do YouTube ({detail}); "
        "previsão sem o histórico do canal"
    )


def channel_frames(channel: ChannelInfo, sector: str) -> ChannelFrames:
    """Histórico de Shorts do canal e a linha do canal, no formato da base."""
    shorts, channels = _base()
    channel_id = channel.channel_id or "__canal__"
    if channel.channel_id and channel.channel_id in set(shorts["channel_id"]):
        history = shorts[shorts["channel_id"] == channel.channel_id]
        info = channels[channels["channel_id"] == channel.channel_id]
        return ChannelFrames(history, info)
    warning = None
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
        from googleapiclient.errors import HttpError

        try:
            history = _history_from_api(channel.channel_id)
        except (RuntimeError, HttpError, OSError) as error:
            history = pd.DataFrame(columns=["video_id", "published_at", "likes"])
            warning = _api_failure(error)
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
    return ChannelFrames(history, info, warning)


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


def local_creative(inp: CreativeInput) -> tuple[CreativeInput, list[str]]:
    """O criativo com o vídeo num caminho local (baixado quando é uma URL) e
    os avisos. Um download que falha não derruba a previsão: ela segue sem o
    vídeo."""
    if not source.is_url(inp.creative):
        return inp, []
    try:
        path = source.local_video(inp.creative)
    except Exception as error:  # noqa: BLE001 (segue com metadados e texto)
        note = f"falha ao baixar o vídeo, previsão sem ele: {error}"[:400]
        return inp.model_copy(update={"creative": None}), [note]
    return inp.model_copy(update={"creative": path}), []


def creative_media(inp: CreativeInput) -> dict | None:
    """Atributos do vídeo do criativo, extraídos no subprocesso (ou None).

    Um vídeo que não pode ser lido (arquivo corrompido, sem trilha de vídeo)
    não derruba a previsão: os campos do vídeo ficam vazios e o motivo vai
    para `notes`.
    """
    inp, notes = local_creative(inp)
    if inp.creative is None:
        return {"record": {}, "vectors": {}, "notes": notes} if notes else None
    try:
        return worker.media(str(inp.creative), inp.sector, inp.caption, inp.use_vlm)
    except Exception as error:  # noqa: BLE001 (segue com metadados e texto)
        return {
            "record": {},
            "vectors": {},
            "notes": [f"falha na extração do vídeo, previsão sem ele: {error}"[:400]],
        }


def add_media(row: pd.DataFrame, media: dict | None, model) -> pd.DataFrame:
    """Linha do criativo com as colunas do vídeo, como na base de treino."""
    record = (media or {}).get("record", {})
    values = {k: v for k, v in record.items() if k not in ("video_id", "transcript")}
    values |= {f: np.nan for f in model.features if f not in row and f not in values}
    row = pd.concat(
        [
            row.drop(columns=list(values), errors="ignore"),
            pd.DataFrame([values], index=row.index),
        ],
        axis=1,
    )
    mm.media_columns(row)
    vectors = {kind: {} for kind in model.media_pcas}
    for kind, vector in (media or {}).get("vectors", {}).items():
        if kind in vectors:
            vectors[kind][NEW_ID] = np.asarray(vector, dtype=np.float32)
    mm.add_media_embeddings(row, model.media_pcas, vectors)
    return row


def blocks_used(row: pd.DataFrame, model) -> list[str]:
    """Blocos de features com algum valor na linha do criativo."""
    present = row.iloc[0].notna()
    return [
        name
        for name, features in model.blocks.items()
        if name == "metadados" or present.reindex(features, fill_value=False).any()
    ]


def attributes(media: dict | None) -> dict[str, str | float | None]:
    """Atributos extraídos do vídeo, com nomes legíveis (sim/não nos
    campos booleanos)."""
    record = (media or {}).get("record", {})
    out = {}
    for key, value in record.items():
        label = FEATURE_LABELS.get(key)
        if label is None or value is None:
            continue
        if isinstance(value, float):
            if np.isnan(value):
                continue
            value = ("sim" if value else "não") if key in YES_NO else round(value, 2)
        out[label] = value
    return out


def feature_row(
    inp: CreativeInput, media: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Features do criativo (montadas pelo mesmo código do treino), o
    histórico do canal usado para elas e o aviso, se o histórico não pôde ser
    buscado. `media` é a saída de `creative_media`; sem ela, é calculada
    quando há arquivo de vídeo."""
    history, info, warning = channel_frames(inp.channel, inp.sector)
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
    decline = hist.channel_decline(table)
    row["channel_decline"] = decline.loc[row.index]
    row["history_risk"] = hist.history_risk(table, decline).loc[row.index]
    model = _bundle()
    vector = worker.embed([inp.caption])
    pcs = model.title_pca.transform(vector)
    for i in range(pcs.shape[1]):
        row[f"title_pc{i}"] = pcs[0, i]
    row = add_media(row, media if media is not None else creative_media(inp), model)
    return row, history, warning


# Campos vazios do vídeo também pesam (a LightGBM aprendeu o efeito da
# ausência); somados, viram um fator que diz o que faltou.
MISSING_GROUPS = {
    "__sem_video": "vídeo não informado (sinais e embeddings vazios)",
    "__video_campos_vazios": "medidas do vídeo sem valor (ex.: vídeo sem fala)",
    "__sem_rubrica": "rubrica do extrator não pedida",
    "__rubrica_outros_setores": "campos da rubrica de outros setores (vazios)",
}


def _groups(model, row: pd.DataFrame) -> dict[str, str]:
    """Grupo de cada feature: embedding, campo vazio do vídeo ou ela mesma."""
    values = row.iloc[0]
    video = set(model.blocks.get("sinais", [])) | set(
        model.blocks.get("embeddings", [])
    )
    rubric = model.blocks.get("extrator", [])
    has_video = values.reindex(list(video)).notna().any()
    has_rubric = values.reindex(rubric).notna().any()
    groups = {}
    for feature in model.features:
        missing = pd.isna(values.get(feature))
        if missing and feature in video:
            # Vídeo extraído com um campo vazio (ex.: sem fala) não é "sem vídeo".
            groups[feature] = "__video_campos_vazios" if has_video else "__sem_video"
        elif missing and feature in rubric:
            groups[feature] = (
                "__rubrica_outros_setores" if has_rubric else "__sem_rubrica"
            )
        else:
            prefix = next((p for p in EMBEDDING_GROUPS if feature.startswith(p)), None)
            groups[feature] = prefix or feature
    return groups


def _drivers(model, row: pd.DataFrame) -> list[Driver]:
    contrib = model.contributions(row).iloc[0]
    contrib = contrib.drop(labels=["age_days"], errors="ignore")
    grouped = contrib.groupby(_groups(model, row)).sum()
    top = grouped.reindex(grouped.abs().sort_values(ascending=False).index)
    drivers = []
    for feature, effect in top[:N_DRIVERS].items():
        names = EMBEDDING_GROUPS | MISSING_GROUPS
        if feature in names:
            label, shown = names[feature], ""
        else:
            label = FEATURE_LABELS.get(feature, feature)
            value = row[feature].iloc[0]
            shown = (
                ("sim" if value else "não")
                if feature in YES_NO and pd.notna(value)
                else _fmt(value)
            )
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
    """Mediana de curtidas quando o canal não tem histórico estabilizado, e o
    motivo de baixa confiança que explica de onde ela veio.

    Usa todos os Shorts anteriores do canal (mesmo os com menos de 30 dias,
    cujas curtidas ainda crescem) e, sem nenhum, a mediana do setor na base.
    Sem nada disso, não há referência: o número absoluto sai zerado e só o
    desempenho relativo vale.
    """
    planned = planned_time(inp)
    if len(history) and "likes" in history:
        before = history[history["published_at"] < planned]["likes"].dropna()
        if len(before):
            return float(
                before.astype(float).median()
            ), "curtidas absolutas estimadas pela mediana de todos os Shorts do canal"
    sectors = _sector_medians()
    if inp.sector in sectors:
        return float(
            sectors[inp.sector]
        ), f"curtidas absolutas estimadas pela mediana do setor de {inp.sector}"
    return 0.0, (
        "sem referência de curtidas (canal sem histórico e sem base local): "
        "o número absoluto não vale, só o desempenho relativo ao canal"
    )


def predict_creative(
    inp: CreativeInput, with_comparables: bool = True, media: dict | None = None
) -> Prediction:
    """Prevê as curtidas de um criativo (com ~90 dias) antes da publicação.

    `media` é a saída de `creative_media`, quando já calculada (o agente
    extrai os atributos num nó próprio).
    """
    model = _bundle()
    inp, download_notes = local_creative(inp)
    if media is None:
        media = creative_media(inp)
    row, history, warning = feature_row(inp, media)
    risk = row["history_risk"].to_numpy(bool)
    y, low, high = (float(v[0]) for v in model.predict(row, risk=risk))
    # A largura "habitual" foi medida com a folga geral; a folga maior dos
    # Shorts de risco já tem motivo próprio e não conta aqui.
    _, usual_low, usual_high = (float(v[0]) for v in model.predict(row))
    median = row["history_median_likes"].iloc[0]
    reasons = []
    if pd.isna(median) or row["history_count"].iloc[0] < MIN_HISTORY:
        reasons.append(
            f"canal com menos de {MIN_HISTORY} Shorts estabilizados "
            f"(publicados há {HISTORY_GAP_DAYS}+ dias)"
        )
    if usual_high - usual_low > model.wide_interval:
        reasons.append("intervalo de previsão mais largo que o habitual")
    if inp.channel.channel_id not in model.channels_in_training:
        reasons.append("canal fora da base de treino")
    age = row["history_age_days"].iloc[0]
    if pd.notna(age) and age > hist.STALE_HISTORY_DAYS:
        reasons.append(
            f"histórico desatualizado: os Shorts que servem de referência têm, "
            f"em mediana, {age:.0f} dias (nesses casos o erro é maior)"
        )
    if row["channel_decline"].iloc[0]:
        reasons.append(
            "canal em queda: os Shorts recentes rendem bem abaixo da mediana do "
            "canal e o ritmo de postagem caiu (nesses casos o erro é maior)"
        )
    if warning:
        reasons.append(warning)
    if pd.isna(median):
        median, reason = reference_median(inp, history)
        reasons.append(reason)
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
        blocks_used=blocks_used(row, model),
        attributes=attributes(media),
        notes=_notes(inp, download_notes, media),
    )


def _notes(inp: CreativeInput, download_notes: list[str], media: dict | None) -> list:
    """Avisos sobre o vídeo: download, extração e, sem vídeo nem aviso, que a
    previsão usou só metadados e texto."""
    notes = download_notes + (media or {}).get("notes", [])
    if inp.creative is None and not notes:
        notes.append("sem arquivo de vídeo: previsão com metadados e texto")
    return notes


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
