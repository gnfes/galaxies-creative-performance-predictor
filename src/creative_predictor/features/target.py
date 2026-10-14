"""Alvo do modelo: curtidas do Short em relação ao histórico do próprio canal.

    y = log((curtidas + 1) / (mediana + 1))

`mediana` é a mediana de curtidas dos `HISTORY_POSTS` Shorts do mesmo canal
publicados antes do Short (estritamente antes, para não vazar informação do
próprio post ou de posts simultâneos). Shorts com curtidas ocultas não entram
no histórico. O alvo relativo isola o efeito do conteúdo do tamanho do canal;
o +1 evita divisão por zero e suaviza canais com poucas curtidas.

A volta para curtidas absolutas é `(mediana + 1) * exp(y) - 1`.
"""

import numpy as np
import pandas as pd

HISTORY_POSTS = 20
MIN_HISTORY = 10  # abaixo disso a mediana do canal é instável e não há alvo


def channel_history(
    shorts: pd.DataFrame,
    history_posts: int = HISTORY_POSTS,
) -> pd.DataFrame:
    """Mediana e número de Shorts anteriores com curtidas, por Short.

    Espera `video_id`, `channel_id`, `published_at` e `likes` (nulo quando
    oculto). Devolve um DataFrame com o mesmo índice e as colunas
    `history_median_likes` e `history_count`.
    """
    out = pd.DataFrame(
        {"history_median_likes": np.nan, "history_count": 0}, index=shorts.index
    )
    for _, group in shorts.groupby("channel_id", sort=False):
        group = group.sort_values("published_at")
        valid = group[group["likes"].notna()]
        likes = valid["likes"].astype(float)
        # Estatísticas dos últimos `history_posts` válidos incluindo cada um;
        # para um Short qualquer, vale a linha do último válido anterior a ele.
        rolled = pd.DataFrame(
            {
                "published_at": valid["published_at"].to_numpy(),
                "history_median_likes": likes.rolling(history_posts, min_periods=1)
                .median()
                .to_numpy(),
                "history_count": np.minimum(
                    np.arange(1, len(valid) + 1), history_posts
                ),
            }
        )
        lookup = pd.merge_asof(
            group[["published_at"]].reset_index(),
            rolled,
            on="published_at",
            allow_exact_matches=False,
        ).set_index("index")
        out.loc[lookup.index, "history_median_likes"] = lookup["history_median_likes"]
        out.loc[lookup.index, "history_count"] = lookup["history_count"].fillna(0)
    out["history_count"] = out["history_count"].astype(int)
    return out


def relative_target(likes: pd.Series, median: pd.Series) -> pd.Series:
    return np.log((likes.astype(float) + 1) / (median + 1))


def likes_from_target(target, median):
    """Inverso de `relative_target`: curtidas absolutas previstas."""
    return (np.asarray(median) + 1) * np.exp(np.asarray(target)) - 1


def add_target(
    shorts: pd.DataFrame,
    history_posts: int = HISTORY_POSTS,
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    """Acrescenta o histórico do canal e o alvo `target` (nulo sem alvo).

    Fica sem alvo o Short com curtidas ocultas ou com menos de `min_history`
    Shorts anteriores com curtidas.
    """
    out = shorts.join(channel_history(shorts, history_posts))
    has_target = out["likes"].notna() & (out["history_count"] >= min_history)
    target = relative_target(out["likes"], out["history_median_likes"])
    out["target"] = target.where(has_target)
    return out
