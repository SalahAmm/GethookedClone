import pytest

from hooklab.scraper import (
    Video,
    build_search_url,
    download_audio,
    download_media,
    normalize_video,
    search_videos,
)


def test_build_search_url_for_youtube_shorts():
    url = build_search_url("YouTube Shorts", "cats", 5)
    assert url == "ytsearch5:cats"


def test_build_search_url_for_tiktok():
    url = build_search_url("TikTok", "dancing", 5)
    assert url.startswith("https://www.tiktok.com/tag/")


def test_build_search_url_for_reels():
    url = build_search_url("Instagram Reels", "fitness", 5)
    assert url.startswith("https://www.instagram.com/reels/")


def test_normalize_video_maps_fields():
    raw = {
        "id": "abc123",
        "title": "Big Brain Move",
        "view_count": 1200,
        "uploader": "Creator",
        "upload_date": "20260101",
        "webpage_url": "https://short.url/x",
        "thumbnail": "https://img/x.jpg",
        "duration": 27.0,
    }
    v = normalize_video(raw, "YouTube Shorts")
    assert v.id == "abc123"
    assert v.views == 1200
    assert v.duration == 27.0
    assert v.title == "Big Brain Move"


def test_normalize_video_missing_fields_safe():
    v = normalize_video({"id": "x"}, "YouTube Shorts")
    assert v.views == 0
    assert v.uploader == "Unknown"
    assert v.date == ""
    assert v.thumbnail == ""


class _FakeYoutubeDL:
    def __init__(self, entries=None):
        self.entries = entries or []

    def extract_info(self, url, download=False):
        if url == "http://nested":
            return {"id": "nested", "title": "Nested Title", "view_count": 3,
                    "duration": 10, "webpage_url": "http://nested"}
        return {"entries": self.entries}


def test_search_videos_filters_and_normalizes():
    fake = _FakeYoutubeDL(
        [
            {"id": "a", "title": "t", "view_count": 5, "duration": 30, "webpage_url": "https://x/a"},
            {"id": "b", "title": "t2", "view_count": 9, "duration": 120, "webpage_url": "https://x/b"},
        ]
    )
    videos = search_videos("cats", "YouTube Shorts", 5, ydl=fake)
    assert len(videos) == 2
    assert videos[0].id == "a"


def test_search_videos_skips_bad_entries():
    fake = _FakeYoutubeDL([{"_type": "playlist", "title": "b"}])
    videos = search_videos("cats", "YouTube Shorts", 5, ydl=fake)
    assert videos == []


def test_search_videos_resolves_nested_url_entries():
    fake = _FakeYoutubeDL([{"_type": "url", "url": "http://nested", "id": "x"}])
    videos = search_videos("cats", "YouTube Shorts", 5, ydl=fake)
    assert len(videos) == 1
    assert videos[0].id == "nested"


class _RecordingYDL(_FakeYoutubeDL):
    def __init__(self):
        super().__init__()
        self.called_url = None

    def extract_info(self, url, download=False):
        self.called_url = url
        return {"id": "m1", "title": "Media", "view_count": 1, "duration": 5,
                "webpage_url": "https://x/m1"}


def test_download_media_calls_extract_info(tmp_path):
    fake = _RecordingYDL()
    v = Video(id="m1", title="Media", views=1, uploader="", date="", url="https://x/m1",
              thumbnail="", duration=5)
    path = download_media(v, str(tmp_path), ydl=fake)
    assert fake.called_url == "https://x/m1"
    assert path == ""


def test_download_audio_calls_extract_info(tmp_path):
    fake = _RecordingYDL()
    v = Video(id="a1", title="Audio", views=1, uploader="", date="", url="https://x/a1",
              thumbnail="", duration=5)
    path = download_audio(v, str(tmp_path), ydl=fake)
    assert fake.called_url == "https://x/a1"
    assert path == ""