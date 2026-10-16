import json

import pandas as pd
import pytest

from creative_predictor.features import extract
from creative_predictor.features.rubric import Color, Narration


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_waits_when_window_is_full():
    clock = FakeClock()
    limiter = extract.TokenRateLimiter(10_000, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire(3_000)  # 9 mil na janela: passa direto
    assert clock.sleeps == []
    limiter.acquire(3_000)  # passaria de 10 mil: espera a janela andar
    assert clock.now == pytest.approx(60.0)


def test_rate_limiter_lets_an_oversized_request_through_alone():
    clock = FakeClock()
    limiter = extract.TokenRateLimiter(1_000, clock=clock, sleep=clock.sleep)
    limiter.acquire(5_000)
    assert clock.sleeps == []


def test_sample_ids_is_a_fixed_shuffle_of_the_sample():
    shorts = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(20)],
            "media_sample": [i % 2 == 0 for i in range(20)],
        }
    )
    first = extract.sample_ids(shorts)["video_id"].tolist()
    again = extract.sample_ids(shorts.iloc[::-1])["video_id"].tolist()
    assert first == again
    assert sorted(first) == sorted(f"v{i}" for i in range(0, 20, 2))
    assert first != sorted(first, key=lambda v: int(v[1:]))


def test_flatten_rubric():
    out = extract._flatten(
        {
            "tem_rosto": True,
            "forca_hook_0a10": 7,
            "cor_dominante": Color.AZUL,
            "tom_narracao": Narration.CALMO,
            "elementos_video_top5": ["carro", "estrada"],
            "hook_primeiros_3s": "close no farol",
        }
    )
    assert out == {
        "vlm_tem_rosto": 1.0,
        "vlm_forca_hook_0a10": 7.0,
        "vlm_cor_dominante": "azul",
        "vlm_tom_narracao": "calmo",
        "vlm_elementos_video_top5": "carro; estrada",
        "vlm_hook_primeiros_3s": "close no farol",
    }


def test_follow_stops_after_idle_time(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(extract.time, "monotonic", clock)
    monkeypatch.setattr(extract.time, "sleep", clock.sleep)
    batches = iter([["a", "b"], [], [], []])
    processed = []

    def pending():
        return next(batches, [])

    def process(items):
        processed.extend(items)
        return len(items)

    extract._follow(pending, process, idle_minutes=2)
    assert processed == ["a", "b"]
    assert clock.now >= 120


def test_creative_requires_all_frames_and_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(extract.fr, "FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setattr(extract, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    row = (
        pd.DataFrame({"video_id": ["v1"], "sector": ["moda"], "title": ["look"]})
        .itertuples()
        .__next__()
    )
    assert extract.creative_for(row) is None
    frames = tmp_path / "frames" / "v1"
    frames.mkdir(parents=True)
    for i in range(extract.N_FRAMES):
        (frames / f"frame_{i}.jpg").write_bytes(b"x")
    transcript = tmp_path / "transcripts" / extract.WHISPER_MODEL / "v1.json"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({"text": "olá"}))
    creative = extract.creative_for(row)
    assert creative.transcript == "olá"
    assert creative.sector == "moda"
    assert len(creative.frames) == extract.N_FRAMES


def test_shard_splits_without_overlap():
    sample = pd.DataFrame({"video_id": [f"v{i}" for i in range(7)]})
    parts = [extract.shard(sample, f"{i}/3")["video_id"].tolist() for i in range(3)]
    assert sorted(v for part in parts for v in part) == sorted(sample["video_id"])
    assert extract.shard(sample, None).equals(sample)
    with pytest.raises(ValueError):
        extract.shard(sample, "3/3")


def test_creative_skips_transcript_being_written(tmp_path, monkeypatch):
    monkeypatch.setattr(extract.fr, "FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setattr(extract, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    frames = tmp_path / "frames" / "v1"
    frames.mkdir(parents=True)
    for i in range(extract.N_FRAMES):
        (frames / f"frame_{i}.jpg").write_bytes(b"x")
    transcript = tmp_path / "transcripts" / extract.WHISPER_MODEL / "v1.json"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"text": "pela met')
    row = next(
        pd.DataFrame(
            {"video_id": ["v1"], "sector": ["moda"], "title": ["x"]}
        ).itertuples()
    )
    assert extract.creative_for(row) is None
