import numpy as np
import pandas as pd

from creative_predictor.model import dataset as ds
from creative_predictor.model import train


def rows_frame():
    """Dois canais do Brasil e um global, com Shorts em treino, validação e teste."""
    dates = pd.to_datetime(["2025-06-01", "2026-04-01", "2026-07-01"], utc=True)
    records = []
    for channel, region in [("UCa", "br"), ("UCb", "br"), ("UCg", "global")]:
        for i, published in enumerate(dates):
            records.append(
                {
                    "channel_id": channel,
                    "region": region,
                    "published_at": published,
                    "target": float(i),
                    "likes": 10 + i,
                    "history_median_likes": 10.0,
                    "x": float(i),
                }
            )
    frame = pd.DataFrame(records)
    frame["split"] = ds.assign_split(frame["published_at"])
    return frame


def test_assign_split_boundaries():
    published = pd.Series(
        pd.to_datetime(
            [
                "2026-02-28T23:59Z",
                "2026-03-01T00:00Z",
                "2026-05-31T23:59Z",
                "2026-06-01T00:00Z",
            ],
            utc=True,
        )
    )
    assert ds.assign_split(published).tolist() == ["train", "valid", "valid", "test"]


def test_make_split_tests_only_brazil():
    split = train.make_split(rows_frame(), ("br", "global"))
    assert set(split.test["region"]) == {"br"}
    assert set(split.train["region"]) == {"br", "global"}
    br_only = train.make_split(rows_frame(), ("br",))
    assert set(br_only.train["region"]) == {"br"}


def test_channel_split_never_trains_on_held_out_channels(monkeypatch):
    seen = []

    class Model:
        def predict(self, frame):
            return np.zeros(len(frame))

    def fake_fit(train_rows, valid, features, objective, num_rounds=None):
        seen.append(set(train_rows["channel_id"]))
        return Model()

    monkeypatch.setattr(train, "fit_lgbm", fake_fit)
    monkeypatch.setattr(train, "GroupKFold", lambda n_splits: _TwoFolds())
    test_rows, preds = train.channel_split_predictions(
        rows_frame(), ["x"], "huber", num_rounds=10
    )
    assert seen == [{"UCb", "UCg"}, {"UCa", "UCg"}]
    assert sorted(test_rows["channel_id"]) == ["UCa", "UCb"]
    assert set(test_rows["split"]) == {"test"}
    assert len(preds) == 2


class _TwoFolds:
    def split(self, items, groups):
        yield np.array([1]), np.array([0])
        yield np.array([0]), np.array([1])


def test_subscriber_baseline_handles_hidden_subscribers():
    split = train.make_split(rows_frame(), ("br", "global"))
    channels = pd.DataFrame(
        {"channel_id": ["UCa", "UCb", "UCg"], "subscribers": [1000, None, 5000]}
    )
    pred = train.subscriber_baseline(split, channels)
    test = split.test.reset_index(drop=True)
    assert len(pred) == len(test)
    # Canal sem inscritos fica com a mediana do próprio canal.
    assert pred[test.index[test["channel_id"] == "UCb"][0]] == 0
    assert np.isfinite(pred).all()
