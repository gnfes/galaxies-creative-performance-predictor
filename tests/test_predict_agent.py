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
    assert service.reference_median(creative(), history.iloc[:0]) == (
        77.0,
        "mediana do setor de moda",
    )
    monkeypatch.setattr(service, "_sector_medians", dict)
    assert service.reference_median(creative(), history.iloc[:0])[0] == 0.0


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Reply", (), {"content": "explicação do LLM"})()


def fake_service(monkeypatch):
    values = {"a": 0.8, "b": 1.6}
    monkeypatch.setattr(
        graph.service,
        "predict_creative",
        lambda item, with_comparables=True: prediction(
            item.creative_id, values[item.creative_id]
        ),
    )
    monkeypatch.setattr(graph.service, "comparables", lambda item: [])


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
    history, info = service.channel_frames(channel, "moda")
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
    history, info = service.channel_frames(
        schemas.ChannelInfo(channel_id="UCnovo"), "beleza"
    )
    assert history["likes"].tolist() == [12]
    assert info.loc[0, "channel_id"] == "UCnovo"


def test_naive_planned_time_uses_channel_timezone():
    naive = creative(
        # Sem fuso de propósito: é o horário local que a pessoa digita.
        planned_datetime=datetime(2026, 10, 20, 18, 0),  # noqa: DTZ001
        channel={"channel_id": "UCx", "country": "BR"},
    )
    assert service.planned_time(naive) == pd.Timestamp("2026-10-20T21:00Z")
    aware = creative(planned_datetime=datetime(2026, 10, 20, 18, tzinfo=UTC))
    assert service.planned_time(aware) == pd.Timestamp("2026-10-20T18:00Z")
