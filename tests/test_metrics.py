import numpy as np
import pandas as pd
import pytest

from creative_predictor.model import metrics


def test_mae_and_mape():
    assert metrics.mae_log([0, 1], [1, 1]) == 0.5
    # Denominador mínimo de 1 evita divisão por zero em Shorts sem curtidas.
    assert metrics.mape([0, 100], [2, 50]) == pytest.approx((2 + 0.5) / 2)


def test_pairwise_accuracy_within_channel_only():
    groups = pd.Series(["a", "a", "a", "b", "b"])
    y = np.array([1, 2, 3, 10, 0])
    pred = np.array([1, 3, 2, 0, 0])
    # Canal a: 2 de 3 pares certos; canal b: empate na previsão = meio acerto.
    assert metrics.pairwise_accuracy(groups, y, pred) == pytest.approx(2.5 / 4)


def test_spearman_within_skips_small_channels():
    groups = pd.Series(["a"] * 5 + ["b"] * 2)
    y = np.array([1, 2, 3, 4, 5, 1, 2])
    pred = np.array([1, 2, 3, 4, 5, 2, 1])
    assert metrics.spearman_within(groups, y, pred) == pytest.approx(1.0)


def test_bands_and_f1():
    edges = metrics.band_edges(np.arange(100))
    bands = metrics.to_bands(np.array([0, 50, 80, 99]), edges)
    assert bands.tolist() == ["abaixo", "na_media", "acima", "viral"]
    assert (
        metrics.band_f1(np.array([0, 50, 80, 99]), np.array([0, 50, 80, 99]), edges)
        == 1
    )


def test_coverage():
    assert metrics.coverage([0, 1, 5], [-1, 0, 0], [1, 2, 4]) == pytest.approx(2 / 3)


def test_evaluate_returns_absolute_and_relative_metrics():
    frame = pd.DataFrame(
        {
            "target": [0.0, 1.0, -1.0, 0.5, 0.2],
            "likes": [10, 30, 3, 20, 12],
            "history_median_likes": [10.0, 10.0, 10.0, 10.0, 10.0],
            "channel_id": ["a"] * 5,
        }
    )
    pred = np.array([0.0, 0.8, -0.5, 0.4, 0.1])  # mesma ordem do alvo
    result = metrics.evaluate(
        frame, pred, np.array([-0.5, 0.5, 1.5]), low=pred - 1, high=pred + 1
    )
    assert result["pairwise_accuracy"] == 1.0
    assert result["coverage"] == 1.0
    assert result["interval_width"] == pytest.approx(2.0)
    assert result["mae_log"] == pytest.approx(np.mean(np.abs(frame["target"] - pred)))


def test_conformal_margin_reaches_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2_000)
    # Intervalo estreito demais: cobre ~38% de uma normal padrão.
    low, high = np.full_like(y, -0.5), np.full_like(y, 0.5)
    margin = metrics.conformal_margin(y, low, high, coverage=0.8)
    assert margin > 0
    assert metrics.coverage(y, low - margin, high + margin) == pytest.approx(
        0.8, abs=0.02
    )
    # Intervalo largo demais: a folga fica negativa e o estreita.
    assert metrics.conformal_margin(y, low - 5, high + 5, coverage=0.8) < 0
