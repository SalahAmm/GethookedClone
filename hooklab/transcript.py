from __future__ import annotations

import re

_CUE_PATTERN = re.compile(
    r"(\d{2}):(\d{2})[.:](\d{3})\s+-->\s+(\d{2}):(\d{2})[.:](\d{3})"
)


def _parse_cue_seconds(cue: "re.Match") -> float:
    mm, ss = int(cue.group(1)), int(cue.group(2))
    return mm * 60 + ss


def slice_first_5s(vtt_text: str, window_seconds: int = 5) -> str:
    """Return the spoken text from VTT cues whose start time lands in the window."""
    lines = []
    cue = None
    for line in vtt_text.splitlines():
        m = _CUE_PATTERN.search(line)
        if m:
            cue = _parse_cue_seconds(m)
            continue
        if cue is None or cue > window_seconds:
            continue
        stripped = re.sub(r"<[^>]+>", "", line).strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines)


def _fetch_subtitles(video) -> str:
    from yt_dlp import YoutubeDL

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "noplaylist": True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video.url, download=False) or {}
        tracks = info.get("subtitles") or info.get("automatic_captions") or {}
        candidates = list(tracks.get("en") or []) + list(
            next(iter(tracks.values()), None) or []
        )
        for entry in candidates:
            if entry.get("url"):
                import urllib.request

                data = (
                    urllib.request.urlopen(entry["url"], timeout=10)
                    .read()
                    .decode("utf-8", "ignore")
                )
                if data:
                    return data
    except Exception:
        return ""
    return ""


def _transcribe_with_whisper(audio_path: str, model) -> str:
    if not audio_path:
        return ""
    try:
        segments = model.transcribe(audio_path, language="en")
        text = " ".join(seg["text"].strip() for seg in segments)
        return text.strip()
    except Exception:
        return ""


def _default_whisper():
    try:
        from faster_whisper import WhisperModel

        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        return None


def get_first_5s(video, audio_path="", whisper_model=None, caption_provider=None):
    vtt = (caption_provider or _fetch_subtitles)(video)
    if vtt:
        return slice_first_5s(vtt), "subtitles"
    text = ""
    if audio_path or whisper_model:
        model = whisper_model or _default_whisper()
        if model is not None:
            text = _transcribe_with_whisper(audio_path, model)
    if text:
        return text, "whisper"
    return video.title, "title"