import asyncio
import io
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from creative_predictor.agent import graph
from creative_predictor.predict import schemas, service, worker


def creative(**overrides):
    fields = {
        "caption": "Look do dia",
        "planned_datetime": datetime(2026, 10, 20, 18, tzinfo=UTC),
        "sector": "moda",
        "channel": {"channel_id": "UCx"},
    }
    return schemas.CreativeInput(**(fields | overrides))


def prediction(creative_id, relative, flag=False):
    return schemas.Prediction(
        creative_id=creative_id,
        predicted_metric=100 * relative,
        interval_low=50,
        interval_high=300,
        relative_to_channel=relative,
        band="na_media",
        channel_median_likes=100,
        confidence_flag=flag,
        explanation="texto padrão",
    )


def test_input_requires_known_sector_and_caption():
    with pytest.raises(ValidationError):
        creative(sector="esportes")
    with pytest.raises(ValidationError):
        schemas.CreativeInput(
            planned_datetime=datetime.now(UTC), sector="moda", channel={}
        )


def test_channel_habits_use_most_common_value():
    history = pd.DataFrame(
        {"category_id": ["26", "26", "22"], "licensed_content": [True, True, False]}
    )
    assert service.channel_habits(history) == {
        "category_id": "26",
        "licensed_content": True,
    }
    assert service.channel_habits(pd.DataFrame()) == {}


def test_reference_median_falls_back_to_recent_posts_then_sector(monkeypatch):
    history = pd.DataFrame(
        {
            "published_at": pd.to_datetime(
                ["2026-10-01", "2026-10-10", "2026-11-01"], utc=True
            ),
            "likes": pd.array([10, 30, 999], dtype="Int64"),
        }
    )
    monkeypatch.setattr(service, "_sector_medians", lambda: {"moda": 77.0})
    value, source = service.reference_median(creative(), history)
    assert value == 20  # só os posts antes da data planejada
    assert "canal" in source
    value, reason = service.reference_median(creative(), history.iloc[:0])
    assert (value, "mediana do setor de moda" in reason) == (77.0, True)
    monkeypatch.setattr(service, "_sector_medians", dict)
    value, reason = service.reference_median(creative(), history.iloc[:0])
    assert value == 0.0
    assert "não vale" in reason


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Reply", (), {"content": "explicação do LLM"})()


def fake_service(monkeypatch):
    values = {"a": 0.8, "b": 1.6}
    seen = {}

    def predict(item, with_comparables=True, media=None):
        seen[item.creative_id] = media
        return prediction(item.creative_id, values[item.creative_id])

    monkeypatch.setattr(graph.service, "predict_creative", predict)
    monkeypatch.setattr(
        graph.service,
        "creative_media",
        lambda item: {"record": {"cuts": 3.0}} if item.creative else None,
    )
    monkeypatch.setattr(graph.service, "comparables", lambda item: [])
    return seen


def test_agent_extracts_video_once_and_passes_it_to_the_model(monkeypatch):
    seen = fake_service(monkeypatch)
    monkeypatch.setattr(graph, "_llm", lambda: None)
    graph.analyze(
        [creative(creative_id="a", creative="v.mp4"), creative(creative_id="b")]
    )
    assert seen == {"a": {"record": {"cuts": 3.0}}, "b": None}


def test_agent_ranks_and_explains_with_llm(monkeypatch):
    fake_service(monkeypatch)
    llm = FakeLLM()
    monkeypatch.setattr(graph, "_llm", lambda: llm)
    result = graph.analyze([creative(creative_id="a"), creative(creative_id="b")])
    assert [p.creative_id for p in result.predictions] == ["b", "a"]
    assert all(p.explanation == "explicação do LLM" for p in result.predictions)
    assert result.explanation == "explicação do LLM"
    # O LLM recebe os números já calculados pelo modelo.
    assert '"vezes_a_mediana_do_canal": 1.6' in llm.prompts[-1]


def test_agent_works_without_llm(monkeypatch):
    fake_service(monkeypatch)
    monkeypatch.setattr(graph, "_llm", lambda: None)
    result = graph.analyze([creative(creative_id="a")])
    assert result.predictions[0].explanation == "texto padrão"
    assert result.explanation == ""


def test_mcp_exposes_tools(monkeypatch):
    from fastmcp import Client

    from creative_predictor import mcp_server

    monkeypatch.setattr(
        mcp_server.service, "predict_creative", lambda c: prediction("x", 1.2)
    )

    async def call():
        async with Client(mcp_server.mcp) as client:
            names = {t.name for t in await client.list_tools()}
            result = await client.call_tool(
                "predict_creative",
                {"creative": json.loads(creative().model_dump_json())},
            )
            return names, result.structured_content

    names, data = asyncio.run(call())
    assert names == {"predict_creative", "rank_creatives", "analyze_creatives"}
    assert data["relative_to_channel"] == 1.2


def test_worker_protocol(monkeypatch):
    from creative_predictor.features import text

    monkeypatch.setattr(
        text,
        "load_encoder",
        lambda: lambda texts: np.array([[len(t), 1.0] for t in texts]),
    )
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        io.StringIO(json.dumps({"texts": ["ab", "abc"]}) + "\n" + "não é json\n"),
    )
    replies = io.StringIO()
    worker._serve(replies)
    first, second = (json.loads(line) for line in replies.getvalue().splitlines())
    assert first == {"vectors": [[2.0, 1.0], [3.0, 1.0]]}
    assert "error" in second


def test_worker_media_request(monkeypatch):
    from creative_predictor.predict import media

    calls = []

    class FakeMedia:
        def __call__(self, **request):
            calls.append(request)
            return {"video_id": "criativo_x", "record": {}, "vectors": {}, "notes": []}

    monkeypatch.setattr(media, "CreativeMedia", FakeMedia)
    request = {"video": "v.mp4", "sector": "moda", "title": "t", "use_vlm": False}
    monkeypatch.setattr(
        worker.sys, "stdin", io.StringIO(json.dumps({"media": request}) + "\n")
    )
    replies = io.StringIO()
    worker._serve(replies)
    assert json.loads(replies.getvalue())["video_id"] == "criativo_x"
    assert calls == [request]


def test_creative_id_reuses_base_id_or_hashes_content(tmp_path, monkeypatch):
    from creative_predictor.predict import media

    base = tmp_path / "media"
    (base / "abc123").mkdir(parents=True)
    (base / "abc123" / "video.mp4").write_bytes(b"x")
    monkeypatch.setattr(media, "MEDIA_DIR", base)
    assert media.creative_id(base / "abc123" / "video.mp4") == "abc123"
    other = tmp_path / "novo.mp4"
    other.write_bytes(b"x")
    copy = tmp_path / "copia.mp4"
    copy.write_bytes(b"x")
    assert media.creative_id(other).startswith("criativo_")
    assert media.creative_id(other) == media.creative_id(copy)


class FakeBundle:
    """Modelo mínimo para testar a montagem das colunas e os fatores."""

    def __init__(self):
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(0)
        self.media_pcas = {"frames": PCA(n_components=2).fit(rng.normal(size=(10, 4)))}
        self.blocks = {
            "metadados": ["duration_s", "title_pc0"],
            "sinais": ["cuts", "speech_language"],
            "embeddings": ["frames_pc0", "frames_pc1"],
            "extrator": ["vlm_forca_hook_0a10", "vlm_cor_dominante", "vlm_swatch"],
        }
        self.features = [f for b in self.blocks.values() for f in b]

    def contributions(self, rows):
        values = {f: 0.1 for f in self.features} | {"frames_pc0": 0.3}
        return pd.DataFrame([values], index=rows.index)


def creative_row():
    return pd.DataFrame({"video_id": [service.NEW_ID], "duration_s": [30.0]})


def test_media_columns_match_training_and_rubric_stays_categorical():
    model = FakeBundle()
    row = creative_row()
    media = {
        "record": {
            "cuts": 4.0,
            "speech_language": "es",
            "vlm_forca_hook_0a10": 7.0,
            "vlm_cor_dominante": "azul",
            "vlm_hook_primeiros_3s": "abre com o produto",
        },
        "vectors": {"frames": [1.0, 0.0, 0.0, 0.0]},
    }
    row = service.add_media(row, media, model)
    assert row["speech_language"].iloc[0] == "outro"
    assert isinstance(row["vlm_cor_dominante"].dtype, pd.CategoricalDtype)
    assert row[["frames_pc0", "frames_pc1"]].notna().all(axis=None)
    assert service.blocks_used(row, model) == [
        "metadados",
        "sinais",
        "embeddings",
        "extrator",
    ]
    assert service.attributes(media)["gancho nos 3 primeiros segundos"] == (
        "abre com o produto"
    )


def test_drivers_sum_embedding_components_and_name_missing_video():
    model = FakeBundle()
    row = creative_row()
    row = service.add_media(row, None, model)
    row["title_pc0"] = 0.5
    assert service.blocks_used(row, model) == ["metadados"]
    drivers = {d.feature: d for d in service._drivers(model, row)}
    # Sem vídeo: sinais e embeddings vazios somam num fator só (0,1 × 2 + 0,3 +
    # 0,1), e a rubrica não pedida em outro.
    assert drivers["__sem_video"].effect == pytest.approx(0.6)
    assert drivers["__sem_rubrica"].effect == pytest.approx(0.3)
    assert drivers["title_pc"].label == "tema do título"


def test_unreadable_video_falls_back_to_metadata(monkeypatch):
    def broken(*args):
        raise RuntimeError("subprocesso de PyTorch: arquivo corrompido")

    monkeypatch.setattr(service.worker, "media", broken)
    media = service.creative_media(creative(creative="quebrado.mp4"))
    assert media["record"] == {} and media["vectors"] == {}
    assert "falha na extração do vídeo" in media["notes"][0]
    row = service.add_media(creative_row(), media, FakeBundle())
    assert service.blocks_used(row, FakeBundle()) == ["metadados"]


def test_empty_field_of_a_real_video_is_not_missing_video():
    model = FakeBundle()
    media = {
        "record": {"cuts": 2.0, "speech_language": None},  # vídeo sem fala
        "vectors": {"frames": [1.0, 0.0, 0.0, 0.0]},
    }
    row = service.add_media(creative_row(), media, model)
    groups = service._groups(model, row)
    assert groups["speech_language"] == "__video_campos_vazios"
    assert groups["cuts"] == "cuts"


def test_boolean_attributes_read_as_yes_or_no():
    media = {"record": {"vlm_tem_rosto": 1.0, "face_in_hook": 0.0, "cuts": 3.0}}
    out = service.attributes(media)
    assert out["tem rosto"] == "sim"
    assert out["rosto nos 3 primeiros segundos"] == "não"
    assert out["cortes de cena"] == 3.0


def test_worker_reuses_the_media_model_for_text(monkeypatch):
    from creative_predictor.features import text
    from creative_predictor.predict import media

    loads = []
    monkeypatch.setattr(
        text,
        "load_encoder",
        lambda: loads.append("texto") or (lambda t: np.ones((1, 2))),
    )

    class FakeMedia:
        def __call__(self, **request):
            return {"video_id": "v", "record": {}, "vectors": {}, "notes": []}

        def encode_texts(self, texts):
            return np.zeros((len(texts), 2))

    monkeypatch.setattr(media, "CreativeMedia", FakeMedia)
    request = {"video": "v.mp4", "sector": "moda", "title": "t", "use_vlm": False}
    lines = [{"media": request}, {"texts": ["a"]}]
    monkeypatch.setattr(
        worker.sys, "stdin", io.StringIO("".join(json.dumps(x) + "\n" for x in lines))
    )
    replies = io.StringIO()
    worker._serve(replies)
    assert json.loads(replies.getvalue().splitlines()[1]) == {"vectors": [[0.0, 0.0]]}
    assert loads == []  # o codificador só de texto nem foi carregado


def no_base(monkeypatch):
    empty = pd.DataFrame(columns=["channel_id"])
    monkeypatch.setattr(service, "_base", lambda: (empty, empty))


def test_channel_history_given_by_hand(monkeypatch):
    no_base(monkeypatch)
    channel = schemas.ChannelInfo(
        history=[
            {"published_at": "2026-08-01T12:00:00Z", "likes": 40, "duration_s": 35.4},
            {"published_at": "2026-09-01T12:00:00Z", "likes": None},
        ]
    )
    history, info, warning = service.channel_frames(channel, "moda")
    assert warning is None
    assert history["duration_s"].tolist()[0] == 35
    assert pd.isna(history["likes"].iloc[1])
    assert info.loc[0, "sector"] == "moda"
    assert str(info.loc[0, "channel_created_at"].tz) == "UTC"


def test_channel_history_from_api(monkeypatch):
    no_base(monkeypatch)
    import creative_predictor.collect.shorts as shorts_mod

    monkeypatch.setattr(shorts_mod, "list_shorts", lambda *a, **k: ["v1"])
    monkeypatch.setattr(
        shorts_mod,
        "fetch_videos",
        lambda client, ids: [
            {
                "id": "v1",
                "snippet": {
                    "publishedAt": "2026-07-01T10:00:00Z",
                    "channelId": "UCnovo",
                    "title": "t",
                },
                "statistics": {"likeCount": "12"},
            }
        ],
    )
    monkeypatch.setattr(
        "creative_predictor.collect.youtube.YouTubeClient", lambda **k: None
    )
    history, info, _ = service.channel_frames(
        schemas.ChannelInfo(channel_id="UCnovo"), "beleza"
    )
    assert history["likes"].tolist() == [12]
    assert info.loc[0, "channel_id"] == "UCnovo"


def test_channel_history_api_failure_becomes_a_warning(monkeypatch):
    no_base(monkeypatch)

    def missing_key(channel_id):
        raise RuntimeError("Variável YOUTUBE_API_KEY ausente")

    monkeypatch.setattr(service, "_history_from_api", missing_key)
    history, info, warning = service.channel_frames(
        schemas.ChannelInfo(channel_id="UCnovo"), "moda"
    )
    assert history.empty
    assert info.loc[0, "channel_id"] == "UCnovo"
    assert "YOUTUBE_API_KEY" in warning


def test_api_error_reason_never_carries_the_key(monkeypatch):
    import httplib2
    from googleapiclient.errors import HttpError

    no_base(monkeypatch)

    def quota(channel_id):
        uri = "https://youtube.googleapis.com/youtube/v3/videos?key=SEGREDO&alt=json"
        raise HttpError(httplib2.Response({"status": 403}), b"{}", uri=uri)

    monkeypatch.setattr(service, "_history_from_api", quota)
    warning = service.channel_frames(
        schemas.ChannelInfo(channel_id="UCx"), "moda"
    ).warning
    assert "403" in warning
    assert "SEGREDO" not in warning


class PredictBundle(FakeBundle):
    """FakeBundle com o que `predict_creative` consulta."""

    wide_interval = 10.0
    channels_in_training: frozenset = frozenset()
    band_edges = np.array([-0.5, 0.5, 2.0])

    def predict(self, rows, risk=None):
        return np.array([0.1]), np.array([-0.5]), np.array([0.8])


def test_prediction_turns_api_failure_into_low_confidence(monkeypatch):
    history = pd.DataFrame(columns=["video_id", "published_at", "likes"])
    warning = "histórico do canal indisponível na API do YouTube (x)"
    row = creative_row().assign(
        title_pc0=0.0,
        history_risk=False,
        history_median_likes=np.nan,
        history_count=0,
        history_age_days=np.nan,
        channel_decline=False,
    )
    monkeypatch.setattr(
        service, "feature_row", lambda inp, media: (row, history, warning)
    )
    monkeypatch.setattr(service, "_bundle", PredictBundle)
    monkeypatch.setattr(service, "_sector_medians", dict)
    out = service.predict_creative(creative(), with_comparables=False, media={})
    assert out.confidence_flag
    assert warning in out.confidence_reasons
    assert any("não vale" in r for r in out.confidence_reasons)


def test_naive_planned_time_uses_channel_timezone():
    naive = creative(
        # Sem fuso de propósito: é o horário local que a pessoa digita.
        planned_datetime=datetime(2026, 10, 20, 18, 0),  # noqa: DTZ001
        channel={"channel_id": "UCx", "country": "BR"},
    )
    assert service.planned_time(naive) == pd.Timestamp("2026-10-20T21:00Z")
    aware = creative(planned_datetime=datetime(2026, 10, 20, 18, tzinfo=UTC))
    assert service.planned_time(aware) == pd.Timestamp("2026-10-20T18:00Z")
