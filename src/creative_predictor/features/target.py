"""Alvo do modelo: curtidas do Short em relação ao histórico do próprio canal.

    y = log((curtidas + 1) / (mediana + 1))

`mediana` é a mediana de curtidas dos `HISTORY_POSTS` Shorts do mesmo canal
publicados pelo menos `HISTORY_GAP_DAYS` dias antes do Short. As curtidas da
base são as da data da coleta: sem o intervalo, um post da véspera entraria
no histórico com curtidas que só acumulou depois da publicação do Short. Com
30 dias, o histórico tem só métricas estabilizadas, que são também as únicas
disponíveis ao prever um criativo novo. Shorts com curtidas ocultas não
entram no histórico. O alvo relativo isola o efeito do conteúdo do tamanho do canal;
o +1 evita divisão por zero e suaviza canais com poucas curtidas.

A volta para curtidas absolutas é `(mediana + 1) * exp(y) - 1`.
"""

import numpy as np
import pandas as pd

from creative_predictor.features.history import HISTORY_GAP_DAYS, prior_stats

HISTORY_POSTS = 20
MIN_HISTORY = 10  # abaixo disso a mediana do canal é instável e não há alvo


def channel_history(
    shorts: pd.DataFrame,
    history_posts: int = HISTORY_POSTS,
    gap_days: int = HISTORY_GAP_DAYS,
) -> pd.DataFrame:
    """Mediana e número de Shorts anteriores com curtidas, por Short.

    Espera `channel_id`, `published_at` e `likes` (nulo quando oculto), com
    índice sem repetição. Devolve um DataFrame com o mesmo índice e as colunas
    `history_median_likes` e `history_count`.
    """
    stats = prior_stats(
        shorts, "likes", ("median",), window=history_posts, gap_days=gap_days
    )
    return stats.rename(
        columns={"median": "history_median_likes", "count": "history_count"}
    )


def relative_target(likes: pd.Series, median: pd.Series) -> pd.Series:
    return np.log((likes.astype(float) + 1) / (median + 1))


def likes_from_target(target, median):
    """Inverso de `relative_target`: curtidas absolutas previstas."""
    return (np.asarray(median) + 1) * np.exp(np.asarray(target)) - 1


def add_target(
    shorts: pd.DataFrame,
    history_posts: int = HISTORY_POSTS,
    min_history: int = MIN_HISTORY,
    gap_days: int = HISTORY_GAP_DAYS,
) -> pd.DataFrame:
    """Acrescenta o histórico do canal e o alvo `target` (nulo sem alvo).

    Fica sem alvo o Short com curtidas ocultas ou com menos de `min_history`
    Shorts anteriores com curtidas.
    """
    out = shorts.join(channel_history(shorts, history_posts, gap_days))
    has_target = out["likes"].notna() & (out["history_count"] >= min_history)
    target = relative_target(out["likes"], out["history_median_likes"])
    out["target"] = target.where(has_target)
    return out
