from datetime import date

import httplib2
import pandas as pd
import pytest
from googleapiclient.errors import HttpError
from pydantic import ValidationError

from creative_predictor.collect import channels
from creative_predictor.collect.channels import Candidate, Status
from creative_predictor.collect.youtube import YouTubeClient

CUTOFF = date(2026, 8, 28)
CHANNEL_ID = "UCabc123"


class FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeResource:
    def __init__(self, handler):
        self._handler = handler

    def list(self, **params):
        return FakeRequest(self._handler(params))


class FakeService:
    """Imita `build("youtube", "v3")`: cada recurso responde por uma função."""

    def __init__(self, **handlers):
        self.handlers = handlers
        self.calls = []

    def __getattr__(self, resource):
        handler = self.handlers[resource]

        def logged(params):
            self.calls.append((resource, params))
            return handler(params)

        return lambda: FakeResource(logged)


def not_found():
    return HttpError(httplib2.Response({"status": 404}), b"playlistNotFound")


def channel_item(channel_id=CHANNEL_ID, handle="@canal"):
    return {
        "id": channel_id,
        "snippet": {"customUrl": handle, "title": "Canal", "country": "BR"},
        "statistics": {"subscriberCount": "1200", "videoCount": "300"},
    }


def playlist_pages(dates, page_size=50):
    """Pagina uma lista de datas de publicação como a API de playlistItems."""
    pages = [dates[i : i + page_size] for i in range(0, len(dates), page_size)]

    def handler(params):
        index = int(params.get("pageToken", 0))
        response = {
            "pageInfo": {"totalResults": len(dates)},
            "items": [
                {
                    "contentDetails": {"videoId": f"v{index}-{i}"}
                    | ({"videoPublishedAt": f"{d}T12:00:00Z"} if d else {})
                }
                for i, d in enumerate(pages[index])
            ],
        }
        if index + 1 < len(pages):
            response["nextPageToken"] = str(index + 1)
        return response

    return handler


def videos_stats(params):
    """Views = 100 × posição do vídeo; o vídeo v0-0 esconde as curtidas."""
    items = []
    for n, video_id in enumerate(params["id"].split(",")):
        stats = {"viewCount": str(100 * (n + 1))}
        if video_id != "v0-0":
            stats["likeCount"] = str(n + 1)
        items.append({"id": video_id, "statistics": stats})
    return {"items": items}


def make_client(tmp_path, **handlers):
    service = FakeService(**handlers)
    return YouTubeClient(service=service, cache_dir=tmp_path), service


def candidate(**overrides):
    fields = {
        "handle": "canal",
        "name": "Canal",
        "sector": "beleza",
        "region": "br",
        "kind": "criador",
    }
    return Candidate(**(fields | overrides))


def test_candidate_normalizes_handle():
    assert candidate(handle="canal").handle == "@canal"
    assert candidate(handle="@canal").key == "@canal"


def test_candidate_requires_identifier():
    with pytest.raises(ValidationError, match="handle ou channel_id"):
        candidate(handle=None)


def test_candidate_rejects_unknown_sector():
    with pytest.raises(ValidationError):
        candidate(sector="games")


def test_shorts_playlist_id():
    assert channels.shorts_playlist_id("UCabc123") == "UUSHabc123"
    with pytest.raises(ValueError):
        channels.shorts_playlist_id("UUabc123")


def test_count_shorts_respects_cutoff_and_skips_private(tmp_path):
    dates = ["2026-09-20", "2026-08-29", "2026-08-28", None, "2026-01-01"]
    client, _ = make_client(tmp_path, playlistItems=playlist_pages(dates))
    scan = channels.count_shorts(client, CHANNEL_ID, CUTOFF)
    assert (scan.total, scan.until_cutoff, scan.latest) == (5, 2, date(2026, 9, 20))
    assert scan.recent_ids() == ["v0-2", "v0-4"]


def test_recent_ids_are_most_recent_first():
    scan = channels.ShortsScan(
        before_cutoff=[
            (date(2026, 1, 1), "a"),
            (date(2026, 3, 1), "c"),
            (date(2026, 2, 1), "b"),
        ]
    )
    assert scan.recent_ids(2) == ["c", "b"]


def test_count_shorts_stops_paging_at_limit(tmp_path):
    dates = ["2026-01-01"] * 200
    client, service = make_client(tmp_path, playlistItems=playlist_pages(dates))
    scan = channels.count_shorts(client, CHANNEL_ID, CUTOFF, limit=60)
    assert scan.until_cutoff == 60
    assert len(service.calls) == 2  # 2 páginas de 50 bastam para chegar a 60


def test_count_shorts_without_shorts_playlist(tmp_path):
    client, _ = make_client(tmp_path, playlistItems=lambda p: not_found())
    assert channels.count_shorts(client, CHANNEL_ID, CUTOFF) is None


def test_median_engagement_ignores_hidden_likes(tmp_path):
    client, _ = make_client(tmp_path, videos=videos_stats)
    views, likes = channels.median_engagement(client, ["v0-0", "v0-1", "v0-2"])
    assert views == 200
    assert likes == 2.5  # curtidas de v0-0 ocultas: mediana de [2, 3]


def test_median_engagement_without_videos(tmp_path):
    client, service = make_client(tmp_path)
    assert channels.median_engagement(client, []) == (None, None)
    assert service.calls == []


def test_count_shorts_propagates_other_errors(tmp_path):
    error = HttpError(httplib2.Response({"status": 403}), b"quotaExceeded")
    client, _ = make_client(tmp_path, playlistItems=lambda p: error)
    with pytest.raises(HttpError):
        channels.count_shorts(client, CHANNEL_ID, CUTOFF)


@pytest.mark.parametrize(
    ("n_shorts", "status"), [(50, Status.OK), (49, Status.FEW_SHORTS)]
)
def test_validate_candidate_applies_min_shorts(tmp_path, n_shorts, status):
    client, _ = make_client(
        tmp_path,
        channels=lambda p: {"items": [channel_item()]},
        playlistItems=playlist_pages(["2026-08-01"] * n_shorts),
        videos=videos_stats,
    )
    result = channels.validate_candidate(client, candidate(), CUTOFF)
    assert result.status == status
    assert result.channel_id == CHANNEL_ID
    assert result.subscribers == 1200
    assert result.active is True
    assert result.median_views == 100 * (n_shorts + 1) / 2


def test_validate_candidate_flags_inactive_channel(tmp_path):
    client, _ = make_client(
        tmp_path,
        channels=lambda p: {"items": [channel_item()]},
        playlistItems=playlist_pages(["2025-12-01"] * 60),
        videos=videos_stats,
    )
    result = channels.validate_candidate(client, candidate(), CUTOFF)
    assert result.status == Status.OK
    assert result.active is False


def test_validate_candidate_not_found(tmp_path):
    client, _ = make_client(tmp_path, channels=lambda p: {"items": []})
    result = channels.validate_candidate(client, candidate(), CUTOFF)
    assert result.status == Status.NOT_FOUND


def test_validate_candidate_by_channel_id(tmp_path):
    client, service = make_client(
        tmp_path,
        channels=lambda p: {"items": [channel_item()]},
        playlistItems=lambda p: not_found(),
    )
    result = channels.validate_candidate(
        client, candidate(handle=None, channel_id=CHANNEL_ID), CUTOFF
    )
    assert result.status == Status.NO_SHORTS
    assert service.calls[0] == (
        "channels",
        {"part": "snippet,statistics", "id": CHANNEL_ID},
    )


def test_validate_skips_excluded_candidates(tmp_path):
    client, service = make_client(tmp_path, channels=lambda p: {"items": []})
    result = channels.validate(
        [candidate(), candidate(handle="outro", exclude="fora do setor")],
        client,
        CUTOFF,
    )
    assert list(result["key"]) == ["@canal"]
    assert len(service.calls) == 1


def test_client_caches_responses(tmp_path):
    client, service = make_client(
        tmp_path, channels=lambda p: {"items": [channel_item()]}
    )
    first = client.list("channels", part="snippet", forHandle="@canal")
    second = client.list("channels", part="snippet", forHandle="@canal")
    assert first == second
    assert len(service.calls) == 1
    assert client.quota_used == 1

    client.list("channels", part="snippet", forHandle="@canal", refresh=True)
    assert len(service.calls) == 2


def validation_row(key, status="ok", holdout=False, likes=100.0, views=5000.0, **kw):
    return {
        "key": f"@{key}",
        "channel_id": f"UC{key}",
        "handle": f"@{key}",
        "name": key.upper(),
        "sector": "moda",
        "region": "br",
        "kind": "marca",
        "holdout": holdout,
        "cutoff": "2026-08-28",
        "status": status,
        "median_views": views,
        "median_likes": likes,
    } | kw


def test_export_channels_splits_training_and_holdout():
    validation = pd.DataFrame(
        [
            validation_row("b"),
            validation_row("a", sector="beleza"),
            validation_row("c", status="poucos_shorts"),
            validation_row("d", status="poucos_shorts", holdout=True),
            validation_row("e", status="sem_shorts", holdout=True),
        ]
    )
    content = channels.export_channels(validation)
    assert content["cutoff"] == "2026-08-28"
    assert [c["channel_id"] for c in content["channels"]] == ["UCa", "UCb"]
    assert [c["channel_id"] for c in content["holdout_channels"]] == ["UCd"]
    assert content["channels"][0] == {
        "channel_id": "UCa",
        "handle": "@a",
        "name": "A",
        "sector": "beleza",
        "region": "br",
        "kind": "marca",
        "low_engagement": False,
    }


def test_export_channels_rejects_mixed_cutoffs():
    validation = pd.DataFrame(
        [validation_row("a"), validation_row("b", cutoff="2026-08-31")]
    )
    with pytest.raises(ValueError, match="data de corte"):
        channels.export_channels(validation)


@pytest.mark.parametrize(
    ("views", "likes", "expected"),
    [
        (5000, 10, False),
        (5000, 9, True),
        (999, float("nan"), True),  # curtidas ocultas: decide pelas views
        (1000, None, False),
        (None, None, True),
    ],
)
def test_is_low_engagement(views, likes, expected):
    assert channels.is_low_engagement(views, likes) is expected


def test_candidates_file_is_valid():
    candidates = channels.load_candidates()
    keys = [c.key.lower() for c in candidates]
    assert len(keys) == len(set(keys)), "candidato duplicado"
    assert {c.sector for c in candidates} == set(channels.Sector)
