import wave

import numpy as np
import pytest

from creative_predictor.features import visual


def solid(bgr, size=(64, 36)):
    return np.full((size[0], size[1], 3), bgr, dtype=np.uint8)


def test_image_stats_gray_vs_color():
    gray = visual.image_stats(solid((128, 128, 128)))
    red = visual.image_stats(solid((0, 0, 255)))
    assert gray["saturation"] == 0
    assert gray["colorfulness"] == pytest.approx(0)
    assert gray["contrast"] == 0
    assert red["saturation"] == pytest.approx(1)
    assert red["colorfulness"] > 50


def test_frame_features_with_fake_detectors():
    frames = [solid((10, 10, 10))] * 8
    # Rosto só no 5º frame; texto só no 1º (dentro do gancho).
    faces = iter([[]] * 4 + [[(0, 0, 18, 32)]] + [[]] * 3)
    texts = iter([[(36 * 64 / 4, "promo hoje")]] + [[]] * 7)
    out = visual.frame_features(frames, lambda _: next(faces), lambda _: next(texts))
    assert out["face_frames_share"] == 1 / 8
    assert out["face_area_max"] == pytest.approx(0.25)
    assert out["face_in_hook"] == 0
    assert out["text_in_hook"] == 1
    assert out["text_area_mean"] == pytest.approx(0.25 / 8)
    assert out["text_words_mean"] == pytest.approx(2 / 8)


def test_cut_features():
    out = visual.cut_features([1.5, 4.0, 9.0], duration=20.0)
    assert out == {
        "cuts": 3.0,
        "cuts_per_10s": 1.5,
        "cuts_in_hook": 1.0,
        "first_cut_s": 1.5,
    }
    assert visual.cut_features([], duration=12.0)["first_cut_s"] == 12.0


def test_scene_regex_reads_ffmpeg_log():
    log = (
        "[scdet @ 0x1] lavfi.scd.score: 23.1, lavfi.scd.time: 2.4\n"
        "[scdet @ 0x1] lavfi.scd.score: 11.0, lavfi.scd.time: 10.033\n"
    )
    assert [float(t) for t in visual._SCENE.findall(log)] == [2.4, 10.033]


def write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype(np.int16).tobytes())


def test_audio_features_tone_and_silence(tmp_path):
    rate = 16000
    t = np.arange(rate) / rate
    tone = 0.5 * 32767 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "a.wav"
    write_wav(path, np.concatenate([tone, np.zeros(rate)]))
    out = visual.audio_features(path)
    assert out["silence_share"] == pytest.approx(0.5, abs=0.05)
    assert visual.audio_features(None)["silence_share"] == 1.0


def test_motion_features_static_vs_changing(tmp_path):
    import cv2

    def video(path, frames):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (36, 64)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        return path

    static = video(tmp_path / "s.avi", [solid((50, 50, 50))] * 40)
    flicker = video(
        tmp_path / "f.avi",
        [solid((0, 0, 0)) if i % 10 < 5 else solid((255, 255, 255)) for i in range(40)],
    )
    assert visual.motion_features(static)["motion_mean"] == pytest.approx(0, abs=0.01)
    assert visual.motion_features(flicker)["motion_mean"] > 0.5


def test_visual_extractor_uses_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(visual, "frame_features", lambda *a: calls.append(1) or {})
    monkeypatch.setattr(visual, "motion_features", lambda v: {"motion_mean": 0.1})
    monkeypatch.setattr(visual, "scene_cuts", lambda v: [1.0])
    monkeypatch.setattr(visual, "audio_features", lambda a: {"silence_share": 0.2})
    extractor = visual.VisualExtractor(
        detect_faces=lambda i: [], read_text=lambda i: [], root=tmp_path
    )
    first = extractor("v1", tmp_path / "v.mp4", [], None, 10.0)
    second = extractor("v1", tmp_path / "v.mp4", [], None, 10.0)
    assert first == second
    assert first["cuts"] == 1.0
    assert len(calls) == 1


def test_ensure_yunet_rejects_wrong_hash(tmp_path):
    source = tmp_path / "fake.onnx"
    source.write_bytes(b"not a model")
    target = tmp_path / "models" / "yunet.onnx"
    with pytest.raises(ValueError, match="hash"):
        visual.ensure_yunet(target, url=source.as_uri())
    assert not target.exists()
    assert not target.with_suffix(".tmp").exists()


def test_ensure_yunet_keeps_existing_file(tmp_path):
    target = tmp_path / "yunet.onnx"
    target.write_bytes(b"x")
    assert visual.ensure_yunet(target, url="file:///nao/existe") == target
