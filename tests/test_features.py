import numpy as np
import pandas as pd
import pytest

from creative_predictor.features import history, text


def shorts_frame(days, channel="UCa", **cols):
    published = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(days, unit="D")
    frame = pd.DataFrame(
        {"channel_id": channel, "published_at": published, **cols},
    )
    return frame


def test_prior_stats_mean_std_and_count():
    shorts = shorts_frame([0, 1, 2, 3], value=[1.0, 3.0, np.nan, 5.0])
    out = history.prior_stats(shorts, "value", ("mean", "std"), window=2)
    assert out["count"].tolist() == [0, 1, 2, 2]
    assert np.isnan(out.loc[0, "mean"])
    assert out["mean"].iloc[1:].tolist() == [1.0, 2.0, 2.0]
    assert out.loc[3, "std"] == pytest.approx(np.std([1, 3], ddof=1))


def test_prior_stats_with_gap_ignores_recent_posts():
    shorts = shorts_frame([0, 20, 40], value=[1.0, 100.0, 7.0])
    out = history.prior_stats(shorts, "value", ("mean",), gap_days=30)
    assert out["count"].tolist() == [0, 0, 1]
    assert out.loc[2, "mean"] == 1.0  # o post do dia 20 ainda não tem 30 dias


def test_prior_stats_channel_without_values():
    shorts = pd.concat(
        [
            shorts_frame([0, 1], "UCa", value=[np.nan, np.nan]),
            shorts_frame([0, 1], "UCb", value=[1.0, 2.0]),
        ],
        ignore_index=True,
    )
    out = history.prior_stats(shorts, "value", ("mean",))
    assert out["count"].tolist() == [0, 0, 0, 1]


def test_posting_rhythm():
    shorts = shorts_frame([0, 2, 2, 10, 50])
    out = history.posting_rhythm(shorts, days=30)
    assert np.isnan(out.loc[0, "days_since_prev"])
    assert out["days_since_prev"].iloc[1:].tolist() == [2, 2, 8, 40]
    # Posts no mesmo instante não contam como anteriores um do outro.
    assert out["posts_prev"].tolist() == [0, 1, 1, 3, 0]


def test_history_features_columns_and_duration_ratio():
    shorts = shorts_frame(
        [0, 40, 80],
        target=[0.5, -0.5, 1.0],
        history_median_likes=[np.nan, 9.0, 9.0],
        duration_s=[20, 40, 90],
    )
    channels = pd.DataFrame(
        {
            "channel_id": ["UCa"],
            "channel_created_at": [pd.Timestamp("2025-12-01", tz="UTC")],
        }
    )
    out = history.history_features(shorts, channels)
    assert out.loc[2, "recent_target_mean"] == 0.0
    assert out.loc[2, "duration_vs_usual"] == 90 / 30
    assert out.loc[1, "log_history_median_likes"] == pytest.approx(np.log(10))
    assert out["channel_age_days"].tolist() == [31, 71, 111]


def test_text_stats():
    shorts = pd.DataFrame(
        {
            "title": ["Look do DIA 2026? #moda #look 🔥", None],
            "description": ["veja https://a.com e https://b.com #x", ""],
            "tags": [["a", "b"], None],
        }
    )
    out = text.text_stats(shorts)
    row = out.loc[0]
    assert row["title_hashtags"] == 2
    assert row["title_emojis"] == 1
    assert row["title_question"] == 1
    assert row["title_digits"] == 4
    assert row["description_links"] == 2
    assert row["description_hashtags"] == 1
    assert row["n_tags"] == 2
    assert out.loc[1, "title_chars"] == 0
    assert out.loc[1, "n_tags"] == 0


def test_embed_texts_computes_only_missing_ids(tmp_path):
    calls = []

    def encoder(texts):
        calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    path = tmp_path / "title.parquet"
    ids = pd.Series(["a", "b"])
    first = text.embed_texts(ids, pd.Series(["xx", "yyy"]), path, encoder)
    assert first.loc["b", "e0"] == 3
    again = text.embed_texts(
        pd.Series(["b", "c", "a"]), pd.Series(["yyy", "z", "xx"]), path, encoder
    )
    assert calls == [["xx", "yyy"], ["z"]]
    assert again.index.tolist() == ["b", "c", "a"]
    assert again["e0"].tolist() == [3, 1, 2]
