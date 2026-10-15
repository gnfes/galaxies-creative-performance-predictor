import json
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest
from yt_dlp.utils import DownloadError

from creative_predictor.collect import media
from creative_predictor.features import frames, rubric, transcribe, vlm, vlm_benchmark


def rubric_payload(sector="beleza", **overrides):
    base = {
        "hook_primeiros_3s": "close no batom",
        "forca_hook_0a10": 7,
        "tem_rosto": True,
        "close_up_produto": True,
        "before_after_ou_transformacao": False,
        "try_on_ou_demonstracao": True,
        "texto_na_tela": True,
        "densidade_texto_0a10": 3,
        "ritmo_cortes_0a10": 6,
        "energia_musica_0a10": 5,
        "tom_narracao": "animado",
        "brilho_saturacao_0a10": 8,
        "cor_dominante": "rosa",
        "estetica_geral_0a10": 7,
        "clareza_beneficio_produto_0a10": 6,
        "elementos_video_top5": ["batom", "rosto"],
        "elementos_audio_top5": ["música pop"],
        "potencial_viral_0a10": 6,
        "assunto_no_setor": True,
    }
    extra = {
        "beleza": {"swatch": True, "aplicacao_na_pele": True},
        "moda": {"corpo_inteiro": True, "styling": False},
        "automotivo": {
            "close_do_carro": True,
            "som_do_motor": False,
            "pov_direcao": False,
            "interior": True,
        },
    }[sector]
    return base | extra | overrides


@pytest.mark.parametrize("sector", ["beleza", "moda", "automotivo"])
def test_rubric_validates_each_sector(sector):
    parsed = rubric.rubric_for(sector).model_validate(rubric_payload(sector))
    assert parsed.forca_hook_0a10 == 7


def test_rubric_rejects_out_of_range_and_unknown_fields():
    with pytest.raises(ValueError):
        rubric.BeautyRubric.model_validate(rubric_payload(forca_hook_0a10=11))
    with pytest.raises(ValueError):
        rubric.BeautyRubric.model_validate(rubric_payload(campo_extra=1))


def test_instructions_list_every_field():
    text = rubric.instructions("automotivo")
    for field in rubric.AutoRubric.model_fields:
        assert f"- {field}:" in text


def test_parse_rubric_extracts_json_from_text():
    text = "Claro! ```json\n" + json.dumps(rubric_payload()) + "\n```"
    assert vlm.parse_rubric(text, "beleza").swatch is True
    with pytest.raises(ValueError):
        vlm.parse_rubric("sem json aqui", "beleza")


class FakeExtractor(vlm.Extractor):
    def __init__(self, text=None, error=None):
        self.name = "fake/model"
        self.text, self.error, self.calls = text, error, 0

    def call(self, creative):
        self.calls += 1
        if self.error:
            raise self.error
        return self.text, {"input_tokens": 100, "output_tokens": 20}


def creative(video_id="v1", sector="beleza"):
    return vlm.Creative(video_id, sector, Path("v.mp4"), [], "olá", "título")


def test_extract_caches_answers(tmp_path):
    extractor = FakeExtractor(text=json.dumps(rubric_payload()))
    first = extractor.extract(creative(), root=tmp_path)
    second = extractor.extract(creative(), root=tmp_path)
    assert extractor.calls == 1
    assert first["rubric"] == second["rubric"]
    prompt = vlm.prompt_id("beleza")
    assert (tmp_path / "fake__model" / prompt / "v1.json").is_file()


def test_prompt_id_changes_with_instructions(monkeypatch):
    before = vlm.prompt_id("moda")
    monkeypatch.setattr(vlm, "instructions", lambda sector: "outra instrução")
    assert vlm.prompt_id("moda") != before


def test_extract_does_not_cache_failures(tmp_path):
    extractor = FakeExtractor(error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    record = extractor.extract(creative(), root=tmp_path)
    assert record["rubric"] is None
    assert "429" in record["error"]
    extractor.extract(creative(), root=tmp_path)
    assert extractor.calls == 2


def test_extract_flags_invalid_rubric(tmp_path):
    extractor = FakeExtractor(text=json.dumps(rubric_payload(forca_hook_0a10=99)))
    record = extractor.extract(creative(), root=tmp_path)
    assert record["rubric"] is None
    assert record["error"].startswith("rubrica inválida")


@pytest.mark.parametrize(
    ("duration", "expected_first", "expected_last"),
    [(30.0, 0.5, 29.5), (3.0, 3.0 / 9, 3.0 * 8 / 9)],
)
def test_frame_times(duration, expected_first, expected_last):
    times = frames.frame_times(duration)
    assert len(times) == 8
    assert times[0] == pytest.approx(expected_first, abs=0.01)
    assert times[-1] == pytest.approx(expected_last, abs=0.01)
    assert times == sorted(times)


def test_transcript_summary():
    segments = [
        {"start": 0.0, "end": 2.0, "text": " olá pessoal"},
        {"start": 5.0, "end": 6.0, "text": " tudo bem"},
    ]
    out = transcribe.summarize(
        segments, {"language": "pt", "language_probability": 0.97}, 10.0
    )
    assert out["text"] == "olá pessoal tudo bem"
    assert out["n_words"] == 4
    assert out["speech_seconds"] == 3.0
    assert out["words_per_second"] == 0.4


class FakeYDL:
    fail: ClassVar[set[str]] = set()
    blocked: ClassVar[set[str]] = set()

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        video_id = url.rsplit("/", 1)[-1]
        if video_id in self.fail:
            raise DownloadError("ERROR: [youtube] x: This video is not available")
        if video_id in self.blocked:
            raise DownloadError("Sign in to confirm you're not a bot")
        return {"width": 360, "height": 640, "duration": 30}


def test_download_many_records_status_and_skips_done(tmp_path):
    FakeYDL.fail = {"gone"}
    first = media.download_many(["ok1", "gone"], tmp_path, sleep=0, ydl_class=FakeYDL)
    assert first.set_index("video_id")["status"].to_dict() == {
        "ok1": "ok",
        "gone": "erro",
    }
    FakeYDL.fail = set()
    again = media.download_many(["ok1", "gone"], tmp_path, sleep=0, ydl_class=FakeYDL)
    # Status já gravado não é baixado de novo (nem o vídeo que saiu do ar).
    assert again.set_index("video_id")["status"].to_dict() == {
        "ok1": "ok",
        "gone": "erro",
    }


def _record(sector="beleza", seconds=1.0, **overrides):
    parsed = rubric.rubric_for(sector).model_validate(
        rubric_payload(sector, **overrides)
    )
    return {"rubric": parsed, "seconds": seconds, "input_tokens": 10}


def test_compare_against_reference():
    records = {
        vlm.REFERENCE: {"a": _record(), "b": _record(tem_rosto=False)},
        "candidato": {
            "a": _record(forca_hook_0a10=5),
            "b": {"rubric": None, "seconds": 2.0, "error": "x"},
        },
    }
    summary, per_field = vlm_benchmark.compare(records)
    row = summary.set_index("modelo").loc["candidato"]
    assert row["json_valido"] == 0.5
    assert row["segundos_por_video"] == 1.5
    hook = per_field.set_index("campo").loc["forca_hook_0a10"]
    assert hook["mae"] == 2
    assert per_field.set_index("campo").loc["tem_rosto", "concordancia"] == 1.0


def test_temporary_failures_are_retried(tmp_path):
    FakeYDL.fail, FakeYDL.blocked = set(), {"v"}
    first = media.download_many(["v"], tmp_path, sleep=0, ydl_class=FakeYDL)
    assert first["status"].tolist() == ["falha_temporaria"]
    assert media.read_status("v", tmp_path) is None
    FakeYDL.blocked = set()
    again = media.download_many(["v"], tmp_path, sleep=0, ydl_class=FakeYDL)
    assert again["status"].tolist() == ["ok"]


def test_parse_rubric_ignores_text_after_the_json():
    text = json.dumps(rubric_payload()) + "\nObs.: {sem música}"
    assert vlm.parse_rubric(text, "beleza").tem_rosto is True


def test_invalid_rubric_is_asked_again(tmp_path):
    extractor = FakeExtractor(text=json.dumps(rubric_payload(forca_hook_0a10=99)))
    extractor.extract(creative(), root=tmp_path)
    extractor.extract(creative(), root=tmp_path)
    assert extractor.calls == 2
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")
def test_frames_when_audio_outlasts_video(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=36x64:rate=10:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=9",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )
    assert frames.probe_duration(video, stream="v:0") == pytest.approx(6, abs=0.2)
    paths = frames.extract_frames(video, tmp_path / "frames")
    assert len(paths) == 8
    assert all(p.is_file() for p in paths)
