"""Tabela de modelagem: alvo, features disponíveis antes da publicação e splits.

Split temporal (principal): treino até fev/2026, validação de mar a mai/2026 e
teste de jun a ago/2026, avaliado só nos canais do Brasil (os clientes da
Galaxies são brasileiros; canais globais entram só no treino). A validação
serve para parada antecipada e escolha de configuração; o teste é usado só
na avaliação final.
"""

import numpy as np
import pandas as pd

from creative_predictor import config
from creative_predictor.features.history import history_features
from creative_predictor.features.local_time import channel_timezone, local_time
from creative_predictor.features.target import add_target
from creative_predictor.features.text import text_stats

TRAIN_END = pd.Timestamp("2026-03-01", tz="UTC")
VALID_END = pd.Timestamp("2026-06-01", tz="UTC")

CATEGORICAL = ["sector", "region", "kind", "category_id", "definition"]
FORMAT_FEATURES = [
    "duration_s",
    "has_captions",
    "made_for_kids",
    "licensed_content",
    "has_paid_placement",
]
TIME_FEATURES = ["local_hour", "local_weekday"]
HISTORY_FEATURES = [
    "log_history_median_likes",
    "recent_target_mean",
    "recent_target_std",
    "recent_target_max",
    "recent_target_count",
    "duration_vs_usual",
    "days_since_prev",
    "posts_prev_30d",
    "channel_age_days",
]
TEXT_STAT_FEATURES = [
    "title_chars",
    "title_words",
    "title_hashtags",
    "title_emojis",
    "title_question",
    "title_exclamation",
    "title_digits",
    "title_upper_ratio",
    "description_chars",
    "description_links",
    "description_hashtags",
    "n_tags",
]
# Idade do Short na coleta: não descreve o criativo, mas as curtidas foram
# medidas nessa idade. Entra como controle; na previsão de um criativo novo,
# recebe `PREDICTION_AGE_DAYS`, e a previsão vale para as curtidas com essa
# idade (cerca de 3 meses, perto da idade típica do teste).
CONTROL_FEATURES = ["age_days"]
PREDICTION_AGE_DAYS = 90

METADATA_FEATURES = (
    CATEGORICAL
    + FORMAT_FEATURES
    + TIME_FEATURES
    + HISTORY_FEATURES
    + TEXT_STAT_FEATURES
    + CONTROL_FEATURES
)


def assign_split(published_at: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [published_at < TRAIN_END, published_at < VALID_END],
            ["train", "valid"],
            "test",
        ),
        index=published_at.index,
    )


def build_table(
    shorts: pd.DataFrame | None = None, channels: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Tabela com alvo, features e split para todos os Shorts coletados.

    Sem argumentos, lê as tabelas de `data/processed/`. Linhas de histórico
    (fora da janela) e de canais de demonstração ficam na tabela, marcadas por
    `in_window` e `holdout`; `modeling_rows` filtra as que entram no modelo.
    """
    if shorts is None:
        shorts = pd.read_parquet(config.PROCESSED_DIR / "shorts.parquet")
    if channels is None:
        channels = pd.read_parquet(config.PROCESSED_DIR / "channels.parquet")
    shorts = add_target(shorts.reset_index(drop=True))

    info = channels.set_index("channel_id")
    timezone = shorts["channel_id"].map(
        {cid: channel_timezone(row.country, row.region) for cid, row in info.iterrows()}
    )
    table = shorts.join(local_time(shorts["published_at"], timezone))
    table = table.join(history_features(table, channels))
    table = table.join(text_stats(table))
    table["split"] = assign_split(table["published_at"])
    for col in CATEGORICAL:
        table[col] = table[col].astype("category")
    for col in [
        "has_captions",
        "made_for_kids",
        "licensed_content",
        "has_paid_placement",
    ]:
        table[col] = table[col].astype(float)
    return table


def modeling_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Shorts da janela, de canais de treino, com alvo."""
    keep = table["in_window"] & ~table["holdout"] & table["target"].notna()
    return table[keep]
