import io

import pytest

from creative_predictor.predict import source


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/shorts/abcDEF12345",
        "https://youtube.com/watch?v=abcDEF12345&t=3",
        "https://www.youtube.com/watch?feature=share&v=abcDEF12345",
        "https://youtu.be/abcDEF12345",
    ],
)
def test_youtube_links_resolve_to_the_video_id(url):
    assert source.youtube_id(url) == "abcDEF12345"


def test_other_sites_citing_youtube_are_not_youtube():
    url = "https://cdn.exemplo.com/v.mp4?ref=youtube.com/shorts/abcDEF12345"
    assert source.youtube_id(url) is None


def test_local_paths_pass_through():
    assert source.local_video(None) is None
    assert str(source.local_video("campanha/look.mp4")) == "campanha/look.mp4"
    assert not source.is_url("campanha/look.mp4")


def test_youtube_link_uses_the_collection_download(monkeypatch, tmp_path):
    from creative_predictor.collect import media

    calls = []
    monkeypatch.setattr(media, "MEDIA_DIR", tmp_path)
    monkeypatch.setattr(media, "read_status", lambda vid: None)

    def download(vid):
        calls.append(vid)
        return {"status": "ok"}

    monkeypatch.setattr(media, "download", download)
    monkeypatch.setattr(media, "media_dir", lambda vid: tmp_path / vid)
    path = source.local_video("https://youtu.be/abcDEF12345")
    assert calls == ["abcDEF12345"]
    assert path == tmp_path / "abcDEF12345" / "video.mp4"


def test_unavailable_youtube_video_raises(monkeypatch):
    from creative_predictor.collect import media

    monkeypatch.setattr(
        media, "read_status", lambda vid: {"status": "erro", "erro": "removido"}
    )
    with pytest.raises(RuntimeError, match="removido"):
        source.local_video("https://youtu.be/abcDEF12345")


def test_direct_link_is_downloaded_once_and_size_capped(monkeypatch, tmp_path):
    requests = []

    def urlopen(url, timeout):
        requests.append(url)
        return io.BytesIO(b"video")

    monkeypatch.setattr(source.urllib.request, "urlopen", urlopen)
    url = "https://cdn.exemplo.com/campanha/look.MP4?token=x"
    first = source.local_video(url, root=tmp_path)
    again = source.local_video(url, root=tmp_path)
    assert first == again and first.suffix == ".mp4"
    assert first.read_bytes() == b"video"
    assert requests == [url]

    monkeypatch.setattr(source, "MAX_BYTES", 2)
    with pytest.raises(ValueError, match="maior"):
        source.local_video("https://cdn.exemplo.com/outro.mp4", root=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [first.name]  # sem sobra
