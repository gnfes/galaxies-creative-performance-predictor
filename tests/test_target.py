import numpy as np
import pandas as pd
import pytest

from creative_predictor.features import target


def frame(likes, channel="UCa", start="2026-01-01", freq="1D"):
    published = pd.date_range(start, periods=len(likes), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "video_id": [f"{channel}-{i}" for i in range(len(likes))],
            "channel_id": channel,
            "published_at": published,
            "likes": pd.array(likes, dtype="Int64"),
        }
    )


def test_history_uses_only_previous_posts():
    hist = target.channel_history(frame([10, 20, 30, 40]), history_posts=2, gap_days=0)
    assert hist["history_count"].tolist() == [0, 1, 2, 2]
    assert np.isnan(hist["history_median_likes"].iloc[0])
    assert hist["history_median_likes"].iloc[1:].tolist() == [10, 15, 25]


def test_history_skips_hidden_likes():
    hist = target.channel_history(frame([10, None, 30, 50]), gap_days=0)
    # Short oculto não entra no histórico, mas recebe o histórico anterior.
    assert hist["history_count"].tolist() == [0, 1, 1, 2]
    assert hist["history_median_likes"].iloc[1:].tolist() == [10, 10, 20]


def test_history_ignores_posts_at_the_same_instant():
    shorts = frame([10, 100, 1000])
    shorts.loc[2, "published_at"] = shorts.loc[1, "published_at"]
    hist = target.channel_history(shorts, gap_days=0)
    assert hist.loc[2, "history_median_likes"] == 10
    assert hist.loc[2, "history_count"] == 1


def test_history_is_per_channel_and_keeps_index_order():
    shorts = pd.concat([frame([1, 2, 3], "UCa"), frame([100, 200], "UCb")])
    shorts = shorts.sample(frac=1, random_state=0).reset_index(drop=True)
    hist = target.channel_history(shorts, gap_days=0)
    joined = shorts.join(hist).set_index("video_id")
    assert joined.loc["UCa-2", "history_median_likes"] == 1.5
    assert joined.loc["UCb-1", "history_median_likes"] == 100


def test_add_target_requires_min_history():
    out = target.add_target(frame([10] * 12 + [None, 43]), min_history=10, gap_days=0)
    assert out["target"].iloc[:10].isna().all()
    assert out["target"].iloc[10] == pytest.approx(0.0)
    assert np.isnan(out["target"].iloc[12])  # curtidas ocultas
    assert out["target"].iloc[13] == pytest.approx(np.log(44 / 11))


def test_likes_from_target_inverts_relative_target():
    likes = pd.Series([0.0, 5.0, 900.0])
    median = pd.Series([3.0, 5.0, 120.0])
    y = target.relative_target(likes, median)
    np.testing.assert_allclose(target.likes_from_target(y, median), likes)


def test_history_rejects_duplicate_index():
    shorts = pd.concat([frame([1, 2], "UCa"), frame([3], "UCb")])
    with pytest.raises(ValueError, match="índice repetido"):
        target.channel_history(shorts)


def test_history_keeps_only_posts_older_than_the_gap():
    # Um Short a cada 10 dias: com 30 dias de intervalo, o Short do dia 50 só
    # vê os dos dias 0, 10 e 20.
    shorts = frame([10, 20, 30, 1000, 1000, 5], freq="10D")
    hist = target.channel_history(shorts, gap_days=30)
    assert hist["history_count"].tolist() == [0, 0, 0, 1, 2, 3]
    assert hist.loc[5, "history_median_likes"] == 20
