import numpy as np
import pandas as pd

from creative_predictor.model import llm_baseline as lb


def test_llm_guess_uses_the_same_target_as_the_model():
    target = lb.llm_target(pd.Series([99, 0]), pd.Series([99.0, 9.0]))
    assert target.tolist() == [0.0, np.log(1 / 10)]


def test_sample_keeps_whole_channels_with_enough_shorts(monkeypatch):
    monkeypatch.setattr(lb, "frames_of", lambda video_id: [])
    rows = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(14)],
            "channel_id": ["a"] * 7 + ["b"] * 4 + ["c"] * 2 + ["d"],
            "split": ["test"] * 13 + ["train"],
            "region": ["br"] * 14,
        }
    )
    picked = lb.sample(rows, limit=20)
    counts = picked["channel_id"].value_counts().to_dict()
    # "c" tem menos de 3 Shorts no teste; "d" não está no teste.
    assert counts == {"a": lb.PER_CHANNEL, "b": 4}
    # O limite corta canais inteiros, nunca parte de um canal.
    assert set(lb.sample(rows, limit=6)["channel_id"].value_counts()) <= {4, 5}


def test_history_text_only_uses_stabilized_posts():
    now = pd.Timestamp("2026-07-01", tz="UTC")
    table = pd.DataFrame(
        {
            "channel_id": ["a", "a", "a"],
            "published_at": [now - pd.Timedelta(days=d) for d in (60, 40, 10)],
            "title": ["antigo", "estável", "recente"],
            "likes": [10, 20, 30],
        }
    )
    row = pd.Series({"channel_id": "a", "published_at": now})
    text = lb.history_text(table, row)
    assert "antigo → 10" in text and "estável → 20" in text
    assert "recente" not in text
