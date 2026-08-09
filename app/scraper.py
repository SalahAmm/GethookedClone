from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

from yt_dlp import YoutubeDL

PLATFORM_SEARCH_URLS = {
    "YouTube Shorts": "ytsearch{limit}:{keyword}",
    "TikTok": "https://www.tiktok.com/tag/{keyword}",
    "Instagram Reels": "https://www.instagram.com/reels/{keyword}/",
}

_BASE_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "noplaylist": True,
    "extract_flat": True,
}


@dataclass
class Video:
    id: str
    title: str
    views: int
    uploader: str
    date: str
    url: str
    thumbnail: str
    duration: float
    platform: str = ""
    hook_text: str = ""
    hook_type: str = ""
    hook_score: int = 0
    reasons: list = field(default_factory=list)
    is_llm: bool = False
    media_path: str = ""


def build_search_url(platform: str, keyword: str, limit: int) -> str:
    template = PLATFORM_SEARCH_URLS[platform]
    return template.format(keyword=keyword.strip().replace(" ", "+"), limit=limit)


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def normalize_video(raw: dict, platform: str) -> Video:
    return Video(
        id=str(raw.get("id") or raw.get("url") or ""),
        title=str(raw.get("title") or "Untitled"),
        views=_int(raw.get("view_count")),
        uploader=str(raw.get("uploader") or raw.get("channel") or "Unknown"),
        date=str(raw.get("upload_date") or ""),
        url=str(raw.get("webpage_url") or raw.get("url") or ""),
        thumbnail=str(raw.get("thumbnail") or ""),
        duration=_float(raw.get("duration")),
        platform=platform,
    )


def _make_ydl(outdir: Optional[str] = None) -> YoutubeDL:
    opts = dict(_BASE_OPTS)
    if outdir:
        opts.update(
            {
                "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
                "skip_download": False,
            }
        )
    return YoutubeDL(opts)


def resolve_entry(entry: dict, ydl: YoutubeDL, platform: str) -> Video:
    if entry.get("_type") == "playlist":
        raise ValueError(f"{entry.get('title')!r} is not a single video")
    raw = entry
    url = entry.get("webpage_url") or entry.get("url")
    if entry.get("_type") == "url":
        raw = ydl.extract_info(url, download=False) or entry
    return normalize_video(raw, platform)


def search_videos(keyword: str, platform: str, limit: int, ydl=None) -> list:
    ydl = ydl or _make_ydl()
    url = build_search_url(platform, keyword, limit)
    info = ydl.extract_info(url, download=False) or {}
    entries = info.get("entries") or []
    videos = []
    for entry in entries[:limit]:
        try:
            videos.append(resolve_entry(entry, ydl, platform))
        except Exception:
            continue
    return videos


def download_media(video: Video, outdir: str, ydl=None) -> str:
    os.makedirs(outdir, exist_ok=True)
    ydl = ydl or _make_ydl(outdir)
    ydl.extract_info(video.url, download=True)
    matches = glob(os.path.join(outdir, f"{video.id}.*"))
    return matches[0] if matches else ""


def download_audio(video: Video, outdir: str, ydl=None) -> str:
    os.makedirs(outdir, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    ydl = ydl or YoutubeDL(opts)
    ydl.extract_info(video.url, download=True)
    matches = glob(os.path.join(outdir, f"{video.id}.*"))
    return matches[0] if matches else ""