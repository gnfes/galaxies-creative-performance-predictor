"""Métricas de avaliação pedidas pelo brief.

O alvo é `y = log((curtidas + 1) / (mediana + 1))`, e a mediana do canal é
conhecida antes da publicação. Por isso o erro em `y` é o mesmo erro em
`log(curtidas + 1)`: o MAE em log vale para as duas escalas.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

BANDS = ("abaixo", "na_media", "acima", "viral")
MIN_CHANNEL_POSTS = (
    5  # canais com menos Shorts no teste ficam fora das métricas por canal
)


def mae_log(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def mape(likes: np.ndarray, pred_likes: np.ndarray) -> float:
    """Erro percentual absoluto médio nas curtidas (denominador mínimo 1)."""
    likes = np.asarray(likes, dtype=float)
    return float(np.mean(np.abs(np.asarray(pred_likes) - likes) / np.maximum(likes, 1)))


def median_ape(likes: np.ndarray, pred_likes: np.ndarray) -> float:
    likes = np.asarray(likes, dtype=float)
    return float(
        np.median(np.abs(np.asarray(pred_likes) - likes) / np.maximum(likes, 1))
    )


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def spearman_within(groups: pd.Series, a: np.ndarray, b: np.ndarray) -> float:
    """Spearman médio dentro de cada canal (ordenação entre criativos do canal)."""
    frame = pd.DataFrame({"g": np.asarray(groups), "a": a, "b": b})
    values = [
        spearmanr(part["a"], part["b"]).statistic
        for _, part in frame.groupby("g")
        if len(part) >= MIN_CHANNEL_POSTS
        and part["a"].nunique() > 1
        and part["b"].nunique() > 1
    ]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def pairwise_accuracy(groups: pd.Series, y: np.ndarray, pred: np.ndarray) -> float:
    """Fração de pares do mesmo canal em que o modelo acerta qual rende mais.

    Pares com o mesmo valor real ficam de fora; empate na previsão conta meio
    acerto.
    """
    frame = pd.DataFrame({"g": np.asarray(groups), "y": y, "p": pred})
    hits = total = 0.0
    for _, part in frame.groupby("g"):
        ys, ps = part["y"].to_numpy(), part["p"].to_numpy()
        upper = np.triu_indices(len(part), k=1)  # cada par uma vez
        dy = np.sign(np.subtract.outer(ys, ys)[upper])
        dp = np.sign(np.subtract.outer(ps, ps)[upper])
        valid = dy != 0
        total += valid.sum()
        hits += (dy * dp == 1)[valid].sum() + 0.5 * (dp == 0)[valid].sum()
    return float(hits / total) if total else float("nan")


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(low)) & (y <= np.asarray(high))))


def band_edges(y_train: np.ndarray) -> np.ndarray:
    """Limites das faixas pelos quantis do treino: P25, P75 e P95."""
    return np.quantile(np.asarray(y_train), [0.25, 0.75, 0.95])


def to_bands(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.asarray(BANDS)[np.searchsorted(edges, np.asarray(y), side="right")]


def band_f1(y: np.ndarray, pred: np.ndarray, edges: np.ndarray) -> float:
    """F1 macro nas faixas abaixo / na média / acima / viral."""
    return float(
        f1_score(
            to_bands(y, edges),
            to_bands(pred, edges),
            labels=list(BANDS),
            average="macro",
            zero_division=0,
        )
    )


def evaluate(
    frame: pd.DataFrame,
    pred: np.ndarray,
    edges: np.ndarray,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> dict[str, float]:
    """Todas as métricas para uma previsão do alvo relativo.

    `frame` precisa de `target`, `likes`, `history_median_likes` e
    `channel_id`, nas mesmas linhas de `pred`.
    """
    y = frame["target"].to_numpy()
    likes = frame["likes"].astype(float).to_numpy()
    median = frame["history_median_likes"].to_numpy()
    pred_likes = (median + 1) * np.exp(pred) - 1
    groups = frame["channel_id"]
    result = {
        "n": float(len(frame)),
        "mae_log": mae_log(y, pred),
        "mape": mape(likes, pred_likes),
        "median_ape": median_ape(likes, pred_likes),
        "spearman_likes": spearman(likes, pred_likes),
        "spearman_target": spearman(y, pred) if np.ptp(pred) > 0 else float("nan"),
        "spearman_within_channel": spearman_within(groups, y, pred),
        "pairwise_accuracy": pairwise_accuracy(groups, y, pred),
        "band_f1": band_f1(y, pred, edges),
    }
    if low is not None and high is not None:
        result["coverage"] = coverage(y, low, high)
        result["interval_width"] = float(np.mean(np.asarray(high) - np.asarray(low)))
    return result


def conformal_margin(
    y: np.ndarray, low: np.ndarray, high: np.ndarray, coverage: float = 0.8
) -> float:
    """Folga a somar dos dois lados do intervalo para atingir a cobertura.

    Regressão quantílica conformal (CQR, Romano et al., 2019): o escore de
    cada ponto da calibração é quanto ele fica fora do intervalo (negativo se
    dentro), e a folga é o quantil ajustado desses escores.
    """
    y, low, high = (np.asarray(a, dtype=float) for a in (y, low, high))
    scores = np.maximum(low - y, y - high)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))
