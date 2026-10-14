from datetime import date

import httplib2
import pandas as pd
import pytest
import yaml
from fakes import make_client, not_found
from googleapiclient.errors import HttpError
from pydantic import ValidationError

from creative_predictor.collect import channels
from creative_predictor.collect.channels import Candidate, Status
from creative_predictor.collect.youtube import YouTubeClient

START = date(2025, 1, 1)
CUTOFF = date(2026, 8, 28)
CHANNEL_ID = "UCabc123"


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


def test_count_shorts_keeps_only_the_window(tmp_path):
    dates = ["2026-09-20", "2026-08-28", None, "2025-01-01", "2024-12-31", "2024-06-01"]
    client, _ = make_client(tmp_path, playlistItems=playlist_pages(dates))
    scan = channels.count_shorts(client, CHANNEL_ID, START, CUTOFF)
    assert scan.total == 6
    assert scan.latest == date(2026, 9, 20)
    assert [d for d, _ in scan.in_window] == [date(2026, 8, 28), date(2025, 1, 1)]
    assert scan.recent_ids() == ["v0-1", "v0-3"]


def test_recent_ids_are_most_recent_first():
    scan = channels.ShortsScan(
        in_window=[
            (date(2026, 1, 1), "a"),
            (date(2026, 3, 1), "c"),
            (date(2026, 2, 1), "b"),
        ]
    )
    assert scan.recent_ids(2) == ["c", "b"]


def test_count_shorts_stops_after_passing_window_start(tmp_path):
    # 60 Shorts na janela e 100 anteriores: a 2ª página já passa do início,
    # então a 3ª e a 4ª não são pedidas.
    dates = ["2026-01-01"] * 60 + ["2024-01-01"] * 100
    client, service = make_client(tmp_path, playlistItems=playlist_pages(dates))
    scan = channels.count_shorts(client, CHANNEL_ID, START, CUTOFF)
    assert len(scan.in_window) == 60
    assert len(service.calls) == 2


def test_count_shorts_without_shorts_playlist(tmp_path):
    client, _ = make_client(tmp_path, playlistItems=lambda p: not_found())
    assert channels.count_shorts(client, CHANNEL_ID, START, CUTOFF) is None


def test_measure_engagement_ignores_hidden_likes(tmp_path):
    client, _ = make_client(tmp_path, videos=videos_stats)
    engagement = channels.measure_engagement(client, ["v0-0", "v0-1", "v0-2"])
    assert engagement.median_views == 200
    assert engagement.median_likes == 2.5  # curtidas de v0-0 ocultas: [2, 3]
    assert engagement.paid_share == 0
    assert engagement.hidden_likes_share == pytest.approx(1 / 3)


def test_measure_engagement_paid_share(tmp_path):
    def handler(params):
        stats = [
            {"viewCount": "100000", "likeCount": "100"},  # 0,1%: impulsionado
            {"viewCount": "1000", "likeCount": "50"},
            {"viewCount": "0", "likeCount": "0"},  # sem views: fica de fora
        ]
        return {"items": [{"id": str(i), "statistics": s} for i, s in enumerate(stats)]}

    client, _ = make_client(tmp_path, videos=handler)
    assert channels.measure_engagement(client, ["0", "1", "2"]).paid_share == 0.5


def test_measure_engagement_without_videos(tmp_path):
    client, service = make_client(tmp_path)
    assert channels.measure_engagement(client, []) == channels.Engagement()
    assert service.calls == []


def test_count_shorts_404_on_later_page_is_an_error(tmp_path):
    first_page = playlist_pages(["2026-01-01"] * 60)

    def handler(params):
        return not_found() if "pageToken" in params else first_page(params)

    client, _ = make_client(tmp_path, playlistItems=handler)
    with pytest.raises(HttpError):
        channels.count_shorts(client, CHANNEL_ID, START, CUTOFF)


def test_count_shorts_propagates_other_errors(tmp_path):
    error = HttpError(httplib2.Response({"status": 403}), b"quotaExceeded")
    client, _ = make_client(tmp_path, playlistItems=lambda p: error)
    with pytest.raises(HttpError):
        channels.count_shorts(client, CHANNEL_ID, START, CUTOFF)


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
    result = channels.validate_candidate(client, candidate(), START, CUTOFF)
    assert result.status == status
    assert result.channel_id == CHANNEL_ID
    assert result.subscribers == 1200
    assert result.shorts_in_window == n_shorts
    assert result.median_views == 100 * (n_shorts + 1) / 2
    assert (result.window_start, result.cutoff) == (START, CUTOFF)


def test_validate_candidate_counts_only_the_window(tmp_path):
    # 300 Shorts no total, mas só 20 na janela: não passa no mínimo.
    client, _ = make_client(
        tmp_path,
        channels=lambda p: {"items": [channel_item()]},
        playlistItems=playlist_pages(["2026-05-01"] * 20 + ["2023-01-01"] * 280),
        videos=videos_stats,
    )
    result = channels.validate_candidate(client, candidate(), START, CUTOFF)
    assert result.status == Status.FEW_SHORTS
    assert (result.shorts_total, result.shorts_in_window) == (300, 20)


def test_validate_candidate_not_found(tmp_path):
    client, _ = make_client(tmp_path, channels=lambda p: {"items": []})
    result = channels.validate_candidate(client, candidate(), START, CUTOFF)
    assert result.status == Status.NOT_FOUND


def test_validate_candidate_by_channel_id(tmp_path):
    client, service = make_client(
        tmp_path,
        channels=lambda p: {"items": [channel_item()]},
        playlistItems=lambda p: not_found(),
    )
    result = channels.validate_candidate(
        client, candidate(handle=None, channel_id=CHANNEL_ID), START, CUTOFF
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
        START,
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
        "window_start": "2024-10-15",
        "cutoff": "2026-08-28",
        "status": status,
        "shorts_in_window": 80 if status == "ok" else 10,
        "median_views": views,
        "median_likes": likes,
        "hidden_likes_share": 0.0,
    } | kw


def test_export_channels_splits_training_and_holdout():
    validation = pd.DataFrame(
        [
            validation_row("b"),
            validation_row("a", sector="beleza"),
            validation_row("c", status="poucos_shorts"),
            validation_row("d", status="poucos_shorts", holdout=True),
            validation_row(
                "e", status="sem_shorts", holdout=True, shorts_in_window=None
            ),
            validation_row(
                "f", status="poucos_shorts", holdout=True, shorts_in_window=0
            ),
        ]
    )
    content = channels.export_channels(validation)
    assert (content["window_start"], content["cutoff"]) == ("2024-10-15", "2026-08-28")
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


def test_export_channels_rejects_mixed_windows():
    validation = pd.DataFrame(
        [validation_row("a"), validation_row("b", window_start="2025-01-01")]
    )
    with pytest.raises(ValueError, match="mais de uma janela"):
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


def search_handler(params):
    return {
        "items": [{"snippet": {"channelId": "UCa"}}, {"snippet": {"channelId": "UCb"}}]
    }


def test_search_channels_filters_brazil_and_sorts(tmp_path):
    client, service = make_client(
        tmp_path,
        search=search_handler,
        channels=lambda p: {
            "items": [
                {**channel_item("UCa", "@a"), "statistics": {"subscriberCount": "5"}},
                {**channel_item("UCb", "@b"), "statistics": {"subscriberCount": "9"}},
            ]
        },
    )
    found = channels.search_channels(client, "maquiagem", br=True)
    assert list(found["channel_id"]) == ["UCb", "UCa"]
    assert service.calls[0][1]["regionCode"] == "BR"
    assert client.quota_used == 101


def test_search_channels_handles_removed_channels(tmp_path):
    client, _ = make_client(
        tmp_path, search=search_handler, channels=lambda p: {"items": []}
    )
    assert channels.search_channels(client, "maquiagem").empty


def test_apply_candidates_uses_current_list():
    validation = pd.DataFrame(
        [
            validation_row("a"),
            validation_row("b"),
            validation_row("c", holdout=True),
        ]
    )
    current = [
        candidate(handle="a", holdout=True),  # marcado depois do validate
        candidate(handle="c"),  # holdout retirado depois do validate
        candidate(handle="z"),  # ainda não validado
    ]
    aligned = channels.apply_candidates(validation, current)
    assert dict(zip(aligned["key"], aligned["holdout"])) == {"@a": True, "@c": False}


def test_apply_candidates_drops_excluded():
    validation = pd.DataFrame([validation_row("a"), validation_row("b")])
    current = [candidate(handle="a"), candidate(handle="b", exclude="fora do setor")]
    assert list(channels.apply_candidates(validation, current)["key"]) == ["@a"]


def test_summarize_without_approved_channels():
    assert channels.summarize(pd.DataFrame()).empty
    rejected = pd.DataFrame([validation_row("a", status="poucos_shorts")])
    assert channels.summarize(rejected).empty


def test_export_command_writes_yaml(tmp_path, monkeypatch):
    validation_path = tmp_path / "validation.csv"
    pd.DataFrame([validation_row("a"), validation_row("b")]).to_csv(
        validation_path, index=False
    )
    candidates_path = tmp_path / "candidates.yaml"
    candidates_path.write_text(
        "candidates:\n"
        "  - {handle: a, name: A, sector: moda, region: br, kind: marca}\n"
        "  - {handle: b, name: B, sector: moda, region: br, kind: marca,"
        " holdout: true}\n"
    )
    channels_path = tmp_path / "channels.yaml"
    monkeypatch.setattr(channels, "VALIDATION_PATH", validation_path)
    monkeypatch.setattr(channels, "CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(channels, "CHANNELS_PATH", channels_path)

    channels.main(["export"])

    text = channels_path.read_text()
    assert text.startswith("# Gerado por")
    content = yaml.safe_load(text)
    assert [c["channel_id"] for c in content["channels"]] == ["UCa"]
    assert [c["channel_id"] for c in content["holdout_channels"]] == ["UCb"]


def test_client_refresh_mode_and_atomic_cache(tmp_path):
    client, service = make_client(
        tmp_path, channels=lambda p: {"items": [channel_item()]}
    )
    client.list("channels", part="snippet", forHandle="@canal")

    fresh = YouTubeClient(service=service, cache_dir=tmp_path, refresh=True)
    fresh.list("channels", part="snippet", forHandle="@canal")
    assert len(service.calls) == 2
    assert not list(tmp_path.rglob("*.tmp"))


def test_validate_candidate_rejects_paid_media(tmp_path):
    def paid_stats(params):
        ids = params["id"].split(",")
        # 40% dos Shorts com 0,1% de curtidas por view
        return {
            "items": [
                {
                    "id": v,
                    "statistics": {
                        "viewCount": "100000",
                        "likeCount": "100" if n % 5 < 2 else "3000",
                    },
                }
                for n, v in enumerate(ids)
            ]
        }

    client, _ = make_client(
        tmp_path,
        channels=lambda p: {"items": [channel_item()]},
        playlistItems=playlist_pages(["2026-08-01"] * 60),
        videos=paid_stats,
    )
    result = channels.validate_candidate(client, candidate(), START, CUTOFF)
    assert result.status == Status.PAID_MEDIA
    assert result.paid_share == 0.4


def test_validate_candidate_rejects_hidden_likes(tmp_path):
    def hidden_stats(params):
        ids = params["id"].split(",")
        return {"items": [{"id": v, "statistics": {"viewCount": "500"}} for v in ids]}

    client, _ = make_client(
        tmp_path,
        channels=lambda p: {"items": [channel_item()]},
        playlistItems=playlist_pages(["2026-08-01"] * 60),
        videos=hidden_stats,
    )
    result = channels.validate_candidate(client, candidate(), START, CUTOFF)
    assert result.status == Status.HIDDEN_LIKES
    assert result.hidden_likes_share == 1


def test_export_channels_drops_holdout_with_hidden_likes():
    validation = pd.DataFrame(
        [
            validation_row("a"),
            validation_row(
                "d", status="poucos_shorts", holdout=True, hidden_likes_share=0.0
            ),
            validation_row(
                "e", status="poucos_shorts", holdout=True, hidden_likes_share=1.0
            ),
        ]
    )
    content = channels.export_channels(validation)
    assert [c["channel_id"] for c in content["holdout_channels"]] == ["UCd"]


def test_validate_rejects_inverted_window(tmp_path):
    client, _ = make_client(tmp_path)
    with pytest.raises(ValueError, match="janela invertida"):
        channels.validate([candidate()], client, CUTOFF, START)


def test_export_command_asks_to_revalidate_old_csv(tmp_path, monkeypatch):
    old = tmp_path / "validation.csv"
    pd.DataFrame([{"key": "@a", "status": "ok", "cutoff": "2026-08-28"}]).to_csv(
        old, index=False
    )
    monkeypatch.setattr(channels, "VALIDATION_PATH", old)
    with pytest.raises(SystemExit, match="Rode `validate` de novo"):
        channels.main(["export"])


def test_candidate_key_ignores_handle_case():
    assert candidate(handle="@Jequiti").key == "@jequiti"
    validation = pd.DataFrame([validation_row("jequiti")])
    aligned = channels.apply_candidates(validation, [candidate(handle="Jequiti")])
    assert list(aligned["key"]) == ["@jequiti"]


def test_export_channels_sorts_names_ignoring_case():
    validation = pd.DataFrame(
        [
            validation_row("b", name="adidas"),
            validation_row("a", name="Zara"),
            validation_row("c", name="Dior"),
        ]
    )
    names = [c["name"] for c in channels.export_channels(validation)["channels"]]
    assert names == ["adidas", "Dior", "Zara"]
