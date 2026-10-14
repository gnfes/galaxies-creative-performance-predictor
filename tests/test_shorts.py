from datetime import date

import pandas as pd
import pytest
from fakes import make_client, not_found

from creative_predictor.collect import shorts

START = date(2025, 1, 1)
CUTOFF = date(2026, 8, 28)
COLLECTED = date(2026, 10, 13)


def playlist(dates, page_size=50):
    """playlistItems com um Short por data (mais recente primeiro); id = data."""
    pages = [dates[i : i + page_size] for i in range(0, len(dates), page_size)]

    def handler(params):
        index = int(params.get("pageToken", 0))
        response = {
            "items": [
                {
                    "contentDetails": {"videoId": d}
                    | ({"videoPublishedAt": f"{d}T12:00:00Z"} if d else {})
                }
                for d in pages[index]
            ]
        }
        if index + 1 < len(pages):
            response["nextPageToken"] = str(index + 1)
        return response

    return handler


def video_item(video_id, channel_id="UCa", likes=True, duration="PT30S"):
    published = video_id if len(video_id) == 10 else "2026-01-01"
    stats = {"viewCount": "1000", "commentCount": "4"}
    if likes:
        stats["likeCount"] = "50"
    return {
        "id": video_id,
        "snippet": {
            "publishedAt": f"{published}T12:00:00Z",
            "channelId": channel_id,
            "title": f"Short {video_id}",
            "description": "desc",
            "categoryId": "26",
            "thumbnails": {"high": {"url": "h.jpg"}, "maxres": {"url": "m.jpg"}},
        },
        "contentDetails": {"duration": duration, "caption": "false"},
        "statistics": stats,
        "status": {"madeForKids": False},
        "paidProductPlacementDetails": {"hasPaidProductPlacement": True},
    }


def videos_handler(params):
    return {"items": [video_item(v) for v in params["id"].split(",")]}


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("PT32S", 32), ("PT1M5S", 65), ("PT3M", 180), ("PT1H2M3S", 3723), ("P0D", 0)],
)
def test_parse_duration(value, seconds):
    assert shorts.parse_duration(value) == seconds


def test_parse_duration_invalid():
    assert shorts.parse_duration(None) is None
    assert shorts.parse_duration("abc") is None
    assert shorts.parse_duration("PT") is None
    assert shorts.parse_duration("P") is None


def test_list_shorts_keeps_window_plus_history(tmp_path):
    days = pd.date_range("2026-09-30", "2024-09-01", freq="-5D").strftime("%Y-%m-%d")
    client, service = make_client(tmp_path, playlistItems=playlist(list(days)))
    ids = shorts.list_shorts(client, "UCa", START, CUTOFF, history=3)
    in_window = [d for d in days if str(START) <= d <= str(CUTOFF)]
    before = [d for d in days if d < str(START)][:3]
    assert ids == in_window + before
    # Parou antes de varrer a playlist inteira.
    assert len(service.calls) < -(-len(days) // 50)


def test_list_shorts_skips_private_videos(tmp_path):
    client, _ = make_client(tmp_path, playlistItems=playlist(["2026-05-01", None]))
    assert shorts.list_shorts(client, "UCa", START, CUTOFF) == ["2026-05-01"]


def test_parse_video_keeps_hidden_likes_null():
    row = shorts.parse_video(video_item("2026-03-01", likes=False, duration="PT1M5S"))
    assert row["likes"] is None
    assert row["views"] == 1000
    assert row["duration_s"] == 65
    assert row["has_paid_placement"] is True
    assert row["thumbnail_url"] == "m.jpg"
    assert row["url"] == "https://www.youtube.com/shorts/2026-03-01"


def test_sample_across_window_spreads_over_time():
    # 100 Shorts num único mês e 10 espalhados no resto da janela.
    burst = pd.date_range("2026-03-01", periods=100, freq="6h", tz="UTC")
    spread = pd.date_range("2025-01-15", "2026-08-15", periods=10, tz="UTC")
    published = pd.Series(burst.append(spread)).sort_values(ignore_index=True)
    picked = published[shorts.sample_across_window(published, 20, START, CUTOFF)]
    assert len(picked) == 20
    assert picked.is_unique
    # Todos os 10 Shorts fora do mês concentrado entram na amostra.
    assert set(spread) <= set(picked)


def test_sample_across_window_keeps_all_when_few():
    published = pd.Series(pd.date_range("2026-01-01", periods=5, tz="UTC"))
    assert len(shorts.sample_across_window(published, 120, START, CUTOFF)) == 5


def config_dict(n_media=3):
    channel = {"name": "A", "sector": "moda", "region": "br", "kind": "criador"}
    return {
        "window_start": str(START),
        "cutoff": str(CUTOFF),
        "media_shorts_per_channel": n_media,
        "channels": [channel | {"channel_id": "UCa", "handle": "@a"}],
        "holdout_channels": [],
    }


def test_collect_builds_tables(tmp_path):
    days = ["2026-09-01", "2026-08-01", "2026-05-01", "2026-01-01", "2025-06-01"]
    days += ["2024-12-01"]
    client, _ = make_client(
        tmp_path,
        channels=lambda p: {
            "items": [
                {
                    "id": "UCa",
                    "snippet": {"title": "A", "publishedAt": "2015-01-01T00:00:00Z"},
                    "statistics": {"subscriberCount": "9000", "viewCount": "1"},
                }
            ]
        },
        playlistItems=playlist(days),
        videos=videos_handler,
    )
    table, channels = shorts.collect(client, config_dict(), COLLECTED, progress=False)

    # 2026-09-01 é depois do corte; 2024-12-01 entra só como histórico.
    assert table["video_id"].tolist() == [
        "2024-12-01",
        "2025-06-01",
        "2026-01-01",
        "2026-05-01",
        "2026-08-01",
    ]
    assert table["in_window"].tolist() == [False, True, True, True, True]
    assert table["media_sample"].sum() == 3
    assert not table.loc[~table["in_window"], "media_sample"].any()
    assert table["age_days"].iloc[-1] == (COLLECTED - date(2026, 8, 1)).days
    assert (table["sector"] == "moda").all()
    assert str(table["likes"].dtype) == "Int64"
    assert channels.loc[0, "subscribers"] == 9000
    assert not channels.loc[0, "holdout"]


def test_collect_rejects_recent_cutoff(tmp_path):
    client, _ = make_client(tmp_path)
    with pytest.raises(ValueError, match="não estabilizadas"):
        shorts.collect(client, config_dict(), date(2026, 9, 10), progress=False)


def test_collect_sample_skips_hidden_likes_and_long_videos(tmp_path):
    hidden, long_video = "2026-02-01", "2026-03-01"

    def videos(params):
        items = []
        for v in params["id"].split(","):
            items.append(
                video_item(
                    v,
                    channel_id="UCa" if v < "2026-05-01" else "UCh",
                    likes=v != hidden,
                    duration="PT3M1S" if v == long_video else "PT30S",
                )
            )
        return {"items": items}

    def pages(params):
        if params["playlistId"] == "UUSHa":
            return playlist(["2026-03-01", "2026-02-01", "2026-01-01"])(params)
        if params["playlistId"] == "UUSHh":
            return playlist(["2026-06-01"])(params)
        raise not_found()

    cfg = config_dict(n_media=10)
    cfg["holdout_channels"] = [
        cfg["channels"][0] | {"channel_id": "UCh", "name": "H"},
        cfg["channels"][0] | {"channel_id": "UCx", "name": "X"},  # sem Shorts
    ]
    client, _ = make_client(
        tmp_path,
        channels=lambda p: {"items": []},
        playlistItems=pages,
        videos=videos,
    )
    table, channels = shorts.collect(client, cfg, COLLECTED, progress=False)

    sampled = table.loc[table["media_sample"], "video_id"].tolist()
    assert sampled == ["2026-01-01", "2026-06-01"]
    assert table.set_index("video_id")["holdout"].to_dict() == {
        "2026-01-01": False,
        "2026-02-01": False,
        "2026-03-01": False,
        "2026-06-01": True,
    }
    assert channels["holdout"].tolist() == [False, True, True]


def test_collect_without_shorts_is_an_error(tmp_path):
    def no_playlist(params):
        raise not_found()

    client, _ = make_client(
        tmp_path, channels=lambda p: {"items": []}, playlistItems=no_playlist
    )
    with pytest.raises(ValueError, match="nenhum Short"):
        shorts.collect(client, config_dict(), COLLECTED, progress=False)


def test_main_refuses_past_date_without_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(shorts, "COLLECTION_CACHE_DIR", tmp_path)
    with pytest.raises(SystemExit, match="Sem cache de 2026-10-01"):
        shorts.main(["--collected-on", "2026-10-01"])
