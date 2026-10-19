"""Histórico do canal antes de cada Short.

Toda feature daqui usa só Shorts do mesmo canal publicados estritamente antes
do Short (posts no mesmo instante não contam), então existe no momento de
decidir a publicação e não vaza informação do próprio post. Features que
dependem de curtidas usam só Shorts com pelo menos `HISTORY_GAP_DAYS` dias
(ver `creative_predictor.features.target`); horários e durações dos posts
anteriores já são conhecidos na hora e não precisam do intervalo.
"""

import numpy as np
import pandas as pd

# Curtidas só se estabilizam com cerca de 30 dias (critério do brief).
HISTORY_GAP_DAYS = 30
# Shorts que definem a mediana de curtidas do canal (o alvo, em
# `features.target`) e a idade do histórico.
HISTORY_POSTS = 20
RECENT_POSTS = 20  # janela de posts para as estatísticas de desempenho recente
FREQUENCY_DAYS = 30  # janela de tempo para a frequência de postagem

# Sinais de que o histórico engana a previsão (validados em `model.confidence`).
# Histórico desatualizado: os Shorts de referência têm, em mediana, mais que
# isto de idade (limite escolhido na validação).
STALE_HISTORY_DAYS = 180
# Canal em queda: os últimos Shorts estabilizados rendem bem abaixo da mediana
# (em log; -0,5 é ~60% dela) e o ritmo de postagem caiu.
DECLINE_POSTS = 5
DECLINE_TARGET = -0.5
DECLINE_DAYS = 90


def prior_stats(
    shorts: pd.DataFrame,
    column: str,
    stats: tuple[str, ...],
    window: int = RECENT_POSTS,
    gap_days: int = 0,
) -> pd.DataFrame:
    """Estatísticas de `column` nos `window` Shorts anteriores com valor.

    Com `gap_days`, só entram Shorts publicados pelo menos `gap_days` dias
    antes (métricas já estabilizadas no momento da previsão).

    Devolve, com o mesmo índice de `shorts`, uma coluna por estatística
    (`mean`, `median`, `std`...) e `count` (quantos valores entraram, até
    `window`). Linhas sem valor em `column` também recebem o histórico, mas
    não entram no histórico das seguintes. Espera `channel_id` e
    `published_at`, com índice sem repetição.
    """
    if not shorts.index.is_unique:
        # Os resultados voltam para as linhas pelo índice: rótulos repetidos
        # mandariam o histórico de um Short para outro.
        raise ValueError("índice repetido: use reset_index antes do histórico")
    out = pd.DataFrame(np.nan, index=shorts.index, columns=[*stats, "count"])
    for _, group in shorts.groupby("channel_id", sort=False):
        group = group.sort_values("published_at")
        valid = group[group[column].notna()]
        values = valid[column].astype(float)
        # Estatística dos últimos `window` valores incluindo cada linha; para
        # um Short qualquer vale a linha do último valor anterior a ele.
        rolling = values.rolling(window, min_periods=1)
        table = pd.DataFrame(
            {
                # Series (e não array) mantém o tipo de data mesmo sem linhas.
                "published_at": valid["published_at"].reset_index(drop=True),
                "count": np.minimum(np.arange(1, len(valid) + 1), window),
            }
        )
        for stat in stats:
            table[stat] = getattr(rolling, stat)().to_numpy()
        query = group[["published_at"]].reset_index()
        query["published_at"] -= pd.Timedelta(days=gap_days)
        # Sem intervalo, a comparação é estrita (posts no mesmo instante não
        # contam); com intervalo, um post de exatamente `gap_days` dias entra.
        lookup = pd.merge_asof(
            query, table, on="published_at", allow_exact_matches=gap_days > 0
        ).set_index("index")
        out.loc[lookup.index, [*stats, "count"]] = lookup[[*stats, "count"]]
    out["count"] = out["count"].fillna(0).astype(int)
    return out


def posting_rhythm(shorts: pd.DataFrame, days: int = FREQUENCY_DAYS) -> pd.DataFrame:
    """Dias desde o Short anterior do canal e Shorts nos `days` dias antes."""
    out = pd.DataFrame(index=shorts.index, columns=["days_since_prev", "posts_prev"])
    span = np.timedelta64(days, "D")
    for _, group in shorts.groupby("channel_id", sort=False):
        times = group["published_at"].sort_values()
        t = times.dt.tz_convert(None).to_numpy()  # datetime64 em UTC
        before = np.searchsorted(t, t, side="left")  # posts estritamente antes
        since = np.searchsorted(t, t - span, side="left")
        gap = (t - t[np.maximum(before - 1, 0)]) / np.timedelta64(1, "D")
        gap = np.where(before > 0, gap, np.nan)  # primeiro post do canal
        out.loc[times.index, "days_since_prev"] = gap
        out.loc[times.index, "posts_prev"] = before - since
    return out.astype({"days_since_prev": float, "posts_prev": int})


def history_features(shorts: pd.DataFrame, channels: pd.DataFrame) -> pd.DataFrame:
    """Features do histórico do canal para cada Short.

    Espera em `shorts` as colunas de `add_target` (`target`,
    `history_median_likes`), `likes` e `duration_s`; em `channels`,
    `channel_id` e `channel_created_at`.
    """
    recent = prior_stats(
        shorts, "target", ("mean", "std", "max"), gap_days=HISTORY_GAP_DAYS
    )
    duration = prior_stats(shorts, "duration_s", ("median",))
    rhythm = posting_rhythm(shorts)
    created = shorts["channel_id"].map(
        channels.set_index("channel_id")["channel_created_at"]
    )
    out = pd.DataFrame(index=shorts.index)
    out["log_history_median_likes"] = np.log1p(shorts["history_median_likes"])
    out["recent_target_mean"] = recent["mean"]
    out["recent_target_std"] = recent["std"]
    out["recent_target_max"] = recent["max"]
    out["recent_target_count"] = recent["count"]
    out["duration_vs_usual"] = shorts["duration_s"].astype(float) / duration["median"]
    out["days_since_prev"] = rhythm["days_since_prev"]
    out["posts_prev_30d"] = rhythm["posts_prev"]
    out["channel_age_days"] = (shorts["published_at"] - created).dt.days
    out["history_age_days"] = history_age(shorts)
    return out


def history_age(shorts: pd.DataFrame) -> pd.Series:
    """Idade, em dias, dos Shorts que definem a mediana de curtidas do canal.

    É a mediana da idade dos `HISTORY_POSTS` Shorts anteriores com curtidas e
    pelo menos `HISTORY_GAP_DAYS` dias (os mesmos da mediana do alvo). Um
    canal que ficou anos sem postar tem um histórico velho: a régua da
    previsão é de outra época do canal.
    """
    days = (shorts["published_at"] - pd.Timestamp("2000-01-01", tz="UTC")).dt.days
    reference = shorts.assign(reference_day=days.where(shorts["likes"].notna()))
    stats = prior_stats(
        reference,
        "reference_day",
        ("median",),
        window=HISTORY_POSTS,
        gap_days=HISTORY_GAP_DAYS,
    )
    return days - stats["median"]


def channel_decline(shorts: pd.DataFrame) -> pd.Series:
    """Se o canal está em queda antes de cada Short.

    Os `DECLINE_POSTS` Shorts estabilizados mais recentes têm alvo médio abaixo
    de `DECLINE_TARGET` e houve menos Shorts nos últimos `DECLINE_DAYS` dias
    do que nos `DECLINE_DAYS` anteriores. Espera `target`, além do que
    `prior_stats` e `posting_rhythm` pedem.
    """
    recent = prior_stats(
        shorts,
        "target",
        ("mean",),
        window=DECLINE_POSTS,
        gap_days=HISTORY_GAP_DAYS,
    )
    last = posting_rhythm(shorts, days=DECLINE_DAYS)["posts_prev"]
    both = posting_rhythm(shorts, days=2 * DECLINE_DAYS)["posts_prev"]
    return (
        (recent["count"] >= DECLINE_POSTS)
        & (recent["mean"] < DECLINE_TARGET)
        & (last < both - last)
    )


def history_risk(shorts: pd.DataFrame, decline: pd.Series | None = None) -> pd.Series:
    """Histórico desatualizado ou canal em queda: a previsão erra mais.

    `decline` é o resultado de `channel_decline`, quando já calculado.
    """
    stale = shorts["history_age_days"] > STALE_HISTORY_DAYS
    return stale | (channel_decline(shorts) if decline is None else decline)
