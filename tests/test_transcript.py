import pytest

from hooklab.scraper import Video
from hooklab.transcript import get_first_5s, slice_first_5s

VTT = """WEBVTT

00:00.000 --> 00:02.000
You will not believe this

00:02.500 --> 00:08.000
five second hook is huge and keeps going

00:20.000 --> 00:25.000
way past the window"""


def _v(**kw):
    defaults = dict(id="v1", title="Hooky title", views=0, uploader="", date="",
                    url="https://x/v1", thumbnail="", duration=60)
    defaults.update(kw)
    return Video(**defaults)


class _FakeWhisper:
    def __init__(self, segments):
        self._segments = list(segments)

    def transcribe(self, path, language=None):
        return iter(self._segments)


def _seg(ts, text):
    return {"start": ts, "text": text.strip()}


def test_slice_first_5s_keeps_cues_within_window():
    text = slice_first_5s(VTT, window_seconds=5)
    assert "believe" in text
    assert "keeps going" in text
    assert "way past" not in text


def test_slice_first_5s_strips_vtt_markup():
    marked = VTT + "\n00:05.000 --> 00:06.000\n<b>bold</b> bit\n"
    text = slice_first_5s(marked, window_seconds=10)
    assert "<b>" not in text


def test_get_first_5s_uses_subtitles_when_video_has_caption():
    text, source = get_first_5s(_v(), caption_provider=lambda v: VTT)
    assert source == "subtitles"
    assert "believe" in text


def test_get_first_5s_falls_back_to_whisper_without_captions():
    fake = _FakeWhisper([_seg(0.0, "Testing whisper hook line")])
    text, source = get_first_5s(_v(), audio_path="x.mp3", whisper_model=fake,
                                caption_provider=lambda v: "")
    assert source == "whisper"
    assert "whisper" in text


def test_get_first_5s_last_resort_is_title():
    text, source = get_first_5s(_v(title="Hooky title"), audio_path="",
                                whisper_model=None, caption_provider=lambda v: "")
    assert source == "title"
    assert text == "Hooky title"


def test_get_first_5s_empty_caption_falls_back():
    fake = _FakeWhisper([_seg(0.0, "Whispered only")])
    text, source = get_first_5s(_v(url="https://x/empty"), audio_path="a.mp3",
                                whisper_model=fake, caption_provider=lambda v: "")
    assert source == "whisper"
    assert "Whispered" in text