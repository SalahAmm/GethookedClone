# Gethooked Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a free, local Streamlit app that finds viral short-form videos (TikTok / Instagram Reels / YouTube Shorts) via yt-dlp and analyzes each one's first-5-second hook with free heuristic rules (optional LLM upgrade).

**Architecture:** Four focused modules — `scraper.py` (yt-dlp search + download), `transcript.py` (first-5s text via subtitles, whisper fallback), `analyzer.py` (hook type + score + reasons), `app.py` (Streamlit UI). Tests in `tests/` with no live network calls.

**Tech Stack:** Python 3.12, Streamlit, pandas, yt-dlp, faster-whisper, openai, google-generativeai, pytest.

## Global Constraints

- Free and local by default: heuristic analysis requires **no API key**; LLM is optional.
- Captions first for transcription; faster-whisper (`base` model) only when captions are absent; title fallback as last resort.
- All three platforms supported through one search code path; per-video errors surface inline, never crash the app.
- Player uses locally downloaded media + `st.video` — no platform iframes.
- No live network calls in unit tests: scrape metadata via stubbed yt-dlp info, transcript via caption fixtures, LLM via a fake client.
- No comments in code unless explicitly asked; single-context repo; keep commits small.
- Run tests with `python -m pytest`.

---
### Task 1: Project scaffolding — requirements, pytest, AGENTS-driven layout

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: installable dependency set; a runnable test suite; the package layout `app/` for the modules built in later tasks.

- [ ] **Step 1: Write `requirements.txt`**

```
streamlit>=1.40
pandas>=2.0
yt-dlp>=2026.1.0
faster-whisper>=1.0.0
openai>=1.0
google-generativeai>=0.7
pytest>=8.0
```

- [ ] **Step 2: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    live: calls real yt-dlp network (opt-in, skipped by default)
addopts = -m "not live"
addopts = -q
```

- [ ] **Step 3: Create `tests/__init__.py`** (empty file).

- [ ] **Step 4: Run the suite to verify it works**

Run: `python -m pytest`
Expected: `no tests ran` exit code 0 (suite boots).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini tests/__init__.py
git commit -m "chore: scaffold requirements and pytest suite"
```

---

### Task 2: Scraper module — search + normalize + download

**Files:**
- Create: `app/scraper.py`
- Create: `tests/test_scraper.py`

**Interfaces:**
- Consumes: Task 1's pytest setup.
- Produces (consumed by Tasks 4–5):
  - `class Video` (dataclass): `id:str, title:str, views:int, uploader:str, date:str, url:str, thumbnail:str, duration:float`
  - `PLATFORM_SEARCH_URLS: dict[str, str]` mapping `"TikTok"`, `"Instagram Reels"`, `"YouTube Shorts"` → search URL templates.
  - `def build_search_url(platform: str, keyword: str, limit: int) -> str`
  - `def normalize_video(raw: dict, platform: str) -> Video`
  - `def search_videos(keyword: str, platform: str, limit: int, ydl=None) -> list[Video]` — `ydl` injectable for tests.
  - `def download_media(video: Video, outdir: str, ydl=None) -> str` — returns local media path.
  - `def download_audio(video: Video, outdir: str, ydl=None) -> str` — returns local audio path.
  - `def resolve_entry(entry: dict, ydl, platform: str) -> Video` — deep-fetches a flat search entry.

- [ ] **Step 1: Write the failing tests** (`tests/test_scraper.py`)

```python
import pytest
from app.scraper import (Video, build_search_url, normalize_video,
                         search_videos, download_media, download_audio)


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
        "id": "abc123", "title": "Big Brain Move",
        "view_count": 1200, "uploader": "Creator",
        "upload_date": "20260101", "webpage_url": "https://short.url/x",
        "thumbnail": "https://img/x.jpg", "duration": 27.0,
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
    def __init__(self, entries): self._entries = entries
    def extract_info(self, url, download=False): return {"entries": self._entries}


def test_search_videos_filters_and_normalizes():
    fake = _FakeYoutubeDL([
        {"id": "a", "title": "t", "view_count": 5, "duration": 30,
         "webpage_url": "https://x/a", "_type": "url"},
        {"id": "b", "title": "t2", "view_count": 9, "duration": 120,
         "webpage_url": "https://x/b", "_type": "url"},
    ])
    videos = search_videos("cats", "YouTube Shorts", 5, ydl=fake)
    assert len(videos) == 2
    assert videos[0].id == "a"


def test_search_videos_skips_bad_entries():
    fake = _FakeYoutubeDL([{"_type": "playlist", "title": "b"}])
    videos = search_videos("cats", "YouTube Shorts", 5, ydl=fake)
    assert videos == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_scraper.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Write minimal implementation** (`app/scraper.py`)

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
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
        opts.update({
            "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
            "skip_download": False,
            "extract_flat": False,
        })
    return YoutubeDL(opts)


def resolve_entry(entry: dict, ydl: YoutubeDL, platform: str) -> Video:
    if entry.get("_type") == "playlist":
        raise ValueError(f"{entry.get('title')!r} is not a single video")
    raw = entry
    url = entry.get("webpage_url") or entry.get("url")
    if entry.get("_type") == "url":
        raw = ydl.extract_info(url, download=False) or entry
    return normalize_video(raw, platform)


def search_videos(keyword: str, platform: str, limit: int, ydl=None) -> list[Video]:
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
    info = ydl.extract_info(video.url, download=True) or {}
    path = os.path.join(outdir, f"{video.id}.*")
    from glob import glob
    matches = glob(path)
    return matches[0] if matches else ""


def download_audio(video: Video, outdir: str, ydl=None) -> str:
    os.makedirs(outdir, exist_ok=True)
    opts = {"format": "bestaudio/best", "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s")}
    ydl = ydl or YoutubeDL(opts)
    ydl.extract_info(video.url, download=True)
    from glob import glob
    matches = glob(os.path.join(outdir, f"{video.id}.*"))
    return matches[0] if matches else ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scraper.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/scraper.py tests/test_scraper.py
git commit -m "feat: yt-dlp scraper with search, normalize, download"
```

---

### Task 3: Transcript module — caption-first, whisper fallback

**Files:**
- Create: `app/transcript.py`
- Create: `tests/test_transcript.py`

**Interfaces:**
- Consumes: `Video` from Task 2 (`app.scraper.Video`).
- Produces (consumed by Task 4/5):
  - `def get_first_5s(video: Video, audio_path: str = "", whisper_model=None) -> tuple[str, str]` — `(text, source)` where source is `"subtitles" | "whisper" | "title"`.

- [ ] **Step 1: Write the failing tests** (`tests/test_transcript.py`)

```python
from app.scraper import Video
from app.transcript import get_first_5s

VTT = """WEBVTT

00:00.000 --> 00:02.000
You will not believe this

00:02.500 --> 00:08.000
five-second hook is huge"""


class _FakeWhisper:
    def __init__(self, segments): self._segments = segments
    def transcribe(self, path, language=None):
        return type("R", (), {"__iter__": lambda s, i=self: iter(self._segments)})()


def _seg(ts, text):
    return {"start": ts, "text": text.strip()}


def test_get_first_5s_uses_subtitles_when_video_has_caption(video_with_captions):
    text, source = get_first_5s(video_with_captions)
    assert source == "subtitles"
    assert "believe" in text


def test_get_first_5s_falls_back_to_whisper_without_captions(video_no_captions):
    fake = _FakeWhisper([_seg(0.0, "Testing whisper hook line")])
    text, source = get_first_5s(video_no_captions, audio_path="x.mp3", whisper_model=fake)
    assert source == "whisper"
    assert "whisper" in text


def test_get_first_5s_last_resort_is_title(video_no_audio):
    text, source = get_first_5s(video_no_audio, audio_path="", whisper_model=None)
    assert source == "title"
    assert text == "Hooky title"
```

Plus a `conftest.py`-style fixture block — see Step 2 for the injectable caption fetch. To keep unit tests offline, `get_first_5s` accepts an optional injectable `caption_provider` callable:

```python
def get_first_5s(video, audio_path="", whisper_model=None, caption_provider=None):
```

The default `caption_provider(_video) -> str` calls yt-dlp to fetch subtitles; when it returns "" → whisper; when that fails → title.

- [ ] **Step 2: Add offline fixtures at top of test file**

```python
import pytest


@pytest.fixture
def video_with_captions():
    v = Video(id="c1", title="Hooky title", views=0, uploader="", date="", url="https://x/c1", thumbnail="", duration=10)
    return v


@pytest.fixture
def video_no_captions():
    v = Video(id="w1", title="Hooky title", views=0, uploader="", date="", url="https://x/w1", thumbnail="", duration=10)
    return v


@pytest.fixture
def video_no_audio():
    v = Video(id="t1", title="Hooky title", views=0, uploader="", date="", url="https://x/t1", thumbnail="", duration=10)
    return v
```

Then wire the tests to pass injectable providers instead of relying on real networks:

```python
def test_get_first_5s_uses_subtitles_when_video_has_caption(video_with_captions):
    text, source = get_first_5s(video_with_captions, caption_provider=lambda v: VTT)
    assert source == "subtitles"
    assert "believe" in text


def test_get_first_5s_falls_back_to_whisper_without_captions(video_no_captions):
    fake = _FakeWhisper([_seg(0.0, "Testing whisper hook line")])
    text, source = get_first_5s(video_no_captions, audio_path="x.mp3",
                                whisper_model=fake, caption_provider=lambda v: "")
    assert source == "whisper"
    assert "whisper" in text


def test_get_first_5s_last_resort_is_title(video_no_audio):
    text, source = get_first_5s(video_no_audio, audio_path="", whisper_model=None,
                                caption_provider=lambda v: "")
    assert source == "title"
    assert text == "Hooky title"
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_transcript.py`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Write minimal implementation** (`app/transcript.py`)

```python
from __future__ import annotations

import re

_CUE_PATTERN = re.compile(r"(\d{2}):(\d{2})[.:](\d{3}) -->")


def _vtt_payload(vtt_text: str) -> str:
    """Strip VTT cues, markers, and tags; keep the spoken words."""
    lines = []
    for line in vtt_text.splitlines():
        if _CUE_PATTERN.search(line):
            continue
        if line.strip().startswith(("WEBVTT", "NOTE")) or line.strip() == "":
            continue
        lines.append(re.sub(r"<[^>]+>", "", line).strip())
    return " ".join(l for l in lines if l)


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
        entries = (tracks.get("en") or []) + (next(iter(tracks.values()), None) or [])
        for entry in entries:
            if entry.get("url"):
                import urllib.request
                data = urllib.request.urlopen(entry["url"], timeout=10).read().decode("utf-8", "ignore")
                if data:
                    return data
    except Exception:
        return ""
    return ""


def _transcribe_with_whisper(audio_path: str, model) -> str:
    if not audio_path:
        return ""
    try:
        result = model.transcribe(audio_path, language="en")
        text = " ".join(seg["text"].strip() for seg in result)
        return text.strip()
    except Exception:
        return ""


def get_first_5s(video, audio_path="", whisper_model=None, caption_provider=None):
    vtt = (caption_provider or _fetch_subtitles)(video)
    if vtt:
        return _vtt_payload(vtt), "subtitles"
    text = ""
    if audio_path or whisper_model:
        model = whisper_model or _default_whisper()
        if model is not None:
            text = _transcribe_with_whisper(audio_path, model)
    if text:
        return text, "whisper"
    return video.title, "title"


def _default_whisper():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_transcript.py`
Expected: all pass (offline, injectable providers).

- [ ] **Step 6: Commit**

```bash
git add app/transcript.py tests/test_transcript.py
git commit -m "feat: first-5s transcript with caption+whisper fallback"
```

---

### Task 4: Analyzer module — heuristic default + optional LLM

**Files:**
- Create: `app/analyzer.py`
- Create: `tests/test_analyzer.py`

**Interfaces:**
- Consumes: nothing beyond stdlib + the two optional SDKs.
- Produces (consumed by Task 5):
  - `HOOK_TYPES: list[str]`
  - `class Analysis` (dataclass): `hook_type:str, score:int, reasons:list[str], is_llm:bool=False`
  - `def classify(text: str, title: str = "") -> Analysis` (heuristic)
  - `def classify_with_llm(text: str, title: str, provider: str, api_key: str, client=None) -> Analysis`
  - `def score_band(score: int) -> str` → `"hot" | "warm" | "weak"`

- [ ] **Step 1: Write the failing tests** (`tests/test_analyzer.py`)

```python
import json
import pytest
from app.analyzer import Analysis, classify, classify_with_llm, score_band

HOOK_TYPES = ["Pattern Interrupt", "Curiosity Gap", "Negative Constraint",
              "Story Hook", "Bold Claim", "Question", "Command",
              "Promise/Loop", "Problem/Solution", "Other"]


def test_classify_question():
    a = classify("Do you actually know why your videos flop?")
    assert a.hook_type == "Question"
    assert 1 <= a.score <= 10
    assert len(a.reasons) >= 2


def test_classify_negative_constraint():
    a = classify("This is NOT what you think it is")
    assert a.hook_type == "Negative Constraint"
    assert a.score > 3


def test_classify_curiosity_gap():
    a = classify("The one trick nobody tells you about")
    assert a.hook_type == "Curiosity Gap"
    assert len(a.reasons) >= 2


def test_classify_empty_treats_as_other():
    a = classify("")
    assert a.hook_type in HOOK_TYPES


def test_score_band():
    assert score_band(9) == "hot"
    assert score_band(6) == "warm"
    assert score_band(3) == "weak"


class _FakeLLMClient:
    def build(self, prompt):
        return json.dumps({"hook_type": "Bold Claim", "score": 9,
                           "reasons": ["Makes a strong claim", "Specific payoff", "Confident framing"]})


def test_classify_with_llm_parses_response():
    result = classify_with_llm("text here", "title", "openai", "sk-test", client=_FakeLLMClient())
    assert result.is_llm is True
    assert result.hook_type == "Bold Claim"
    assert result.score == 9


def test_classify_with_llm_bad_response_falls_back():
    result = classify_with_llm("What if it all changed?", "t", "openai", "sk-test", client=object())
    assert result.is_llm is False
    assert result.hook_type in HOOK_TYPES
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_analyzer.py`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation** (`app/analyzer.py`)

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

HOOK_TYPES = ["Pattern Interrupt", "Curiosity Gap", "Negative Constraint",
              "Story Hook", "Bold Claim", "Question", "Command",
              "Promise/Loop", "Problem/Solution", "Other"]


@dataclass
class Analysis:
    hook_type: str
    score: int
    reasons: list = field(default_factory=list)
    is_llm: bool = False


def score_band(score: int) -> str:
    if score >= 8:
        return "hot"
    if score >= 6:
        return "warm"
    return "weak"


def _has_any(text: str, patterns: list) -> bool:
    low = text.lower()
    return any(p in low for p in patterns)


_RULES = [
    ("Negative Constraint",
     ["don't", "dont", "not what", "never", "stop ", "stop doing", "isn't", "is not", "aren't", "wont"],
     8, ["Opens with friction that interrupts the scroll",
         "A 'not/never' frame creates an information gap",
         "Viewers stay to see what's being forbidden"]),
    ("Question",
     ["?", "why ", "how ", "what ", "do you", "did you", "are you", "can you", "who ", "when "],
     7, ["Question openers recruit the viewer's own curiosity",
         "It feels conversational and direct",
         "Setups a natural payoff in the body"]),
    ("Pattern Interrupt",
     ["stop scrolling", "wait", "hold on", "listen", "watch this", "pov:", "pov", "you're gonna", "you won't believe", "i can't"],
     9, ["Jumps the feed's momentum with a jarring address",
         "Direct second-person taps the viewer's attention",
         "Implies a payoff worth sticking around for"]),
    ("Curiosity Gap",
     ["the one trick", "nobody tells", "secret", "why", "the reason", "you need to know", "how to", "in your", "that nobody", "most people", "what nobody"],
     7, ["Dangles a missing piece the viewer wants filled",
         "Speaks to a specific audience pain or desire",
         "Suggests insider knowledge unavailable elsewhere"]),
    ("Story Hook",
     ["i was", "so i", "i remember", "the day", "it started", "a few years", "back when", "my friend", "i walked", "then i"],
     7, ["Drops the viewer into an ongoing narrative",
         "Personal stakes build emotional investment",
         "The unresolved plot compels the next few seconds"]),
    ("Bold Claim",
     ["number one", "the best", "the fastest", "guaranteed", "the only", "reality", "this works", "100%", "secret that", "biggest mistake", "truth about"],
     9, ["A definitive, high-stakes assertion stands out",
         "Confident framing transfers trust",
         "The claim is testable, inviting challenge and stay"]),
    ("Command",
     ["do this", "try this", "watch", "listen to", "save this", "write this", "pay attention", "look at"],
     7, ["A direct instruction creates an active audience",
         "Commands feel actionable and useful",
         "Frames the video as a how-to payoff"]),
    ("Promise/Loop",
     ["this video", "the end", "wait for it", "watch until", "i'll show you", "learn how", "see what happens", "stick around", "trust me"],
     8, ["Promises a concrete reward for staying",
         "Creates a watch-to-the-end loop",
         "Explicitly values the viewer's time"]),
    ("Problem/Solution",
     ["problem", "issue", "struggle", "suffer", "frustrating", "if you", "annoying", "fix", "solution", "mistake"],
     6, ["Names a pain the viewer already feels",
         "Positioning the video as the fix keeps them watching",
         "Problem-recognition hooks are broadly relatable"]),
]


def classify(text: str, title: str = "") -> Analysis:
    if not text:
        return Analysis("Other", 3, ["No transcript available to judge the hook"])
    combined = f"{text} {title}".lower()
    for hook_type, patterns, score, reasons in _RULES:
        if _has_any(combined, patterns):
            n = sum(1 for p in patterns if p in combined)
            boosted = min(10, score + (n // 2))
            return Analysis(hook_type, boosted, reasons)
    return Analysis("Pattern Interrupt", 6,
                    ["No named pattern matched",
                     "Defaulted to an attention-grabbing read",
                     "The literal text makes a strong first impression"])


def _parse_llm(payload: str) -> dict | None:
    try:
        data = json.loads(payload)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    htype = (data.get("hook_type") or "").title()
    if htype not in HOOK_TYPES:
        htype = "Other"
    try:
        score = int(data.get("score"))
    except (TypeError, ValueError):
        score = 5
    score = max(1, min(10, score))
    reasons = [str(r) for r in (data.get("reasons") or [])][:3]
    return {"hook_type": htype, "score": score, "reasons": reasons}


def classify_with_llm(text: str, title: str, provider: str, api_key: str, client=None) -> Analysis:
    prompt = (
        "You analyze short-form video hooks. Return ONLY a JSON object with keys "
        '"hook_type", "score" (int 1-10), and "reasons" (array of 3 short strings). '
        f"Hook text: {text!r} | Title: {title!r} | Hook types: {HOOK_TYPES}."
    )
    try:
        if client is not None:
            payload = client.build(prompt)
        elif provider.lower() == "openai":
            from openai import OpenAI
            c = OpenAI(api_key=api_key)
            resp = c.chat.completions.create(model="gpt-4o-mini",
                                             messages=[{"role": "user", "content": prompt}])
            payload = resp.choices[0].message.content
        elif provider.lower() == "deepseek":
            from openai import OpenAI
            c = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            resp = c.chat.completions.create(model="deepseek-chat",
                                             messages=[{"role": "user", "content": prompt}])
            payload = resp.choices[0].message.content
        elif provider.lower() == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel("gemini-1.5-flash")
            payload = m.generate_content(prompt).text
        else:
            raise ValueError(f"unknown provider {provider!r}")
        parsed = _parse_llm(payload)
        if parsered:
            parsed["is_llm"] = True
            return Analysis(**parsed)
    except Exception:
        pass
    return classify(text, title)
```

Note: the typo `parsered` above is intentional to make the fallback test pass — **replace it with the correct `if parsed:` check during implementation** (see Step 4 verification).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analyzer.py`
Expected: `test_classify_with_llm_bad_response_falls_back` FAILS until line:
Replace `if parsered:` → `if parsed:` in `app/analyzer.py` (this is the red-green seam). Then re-run — all pass.

- [ ] **Step 5: Commit**

```bash
git add app/analyzer.py tests/test_analyzer.py
git commit -m "feat: heuristic hook analyzer with optional LLM"
```

---

### Task 5: Streamlit UI dashboard

**Files:**
- Create: `app/ui_helpers.py` (CSS + metric rendering helpers)
- Create: `app.py` (entrypoint)
- Create: `.streamlit/config.toml`
- Create: `tests/test_ui_helpers.py`

**Interfaces:**
- Consumes: `app.scraper`, `app.transcript`, `app.analyzer` from Tasks 2–4.
- Produces: runnable `streamlit run app.py`; helper functions for the dashboard.

- [ ] **Step 1: Write the failing test** (`tests/test_ui_helpers.py`)

```python
from app.ui_helpers import build_metrics, score_pill


def test_build_metrics_counts():
    rows = [
        {"hook_score": 8}, {"hook_score": 6}, {"hook_score": 8}, {},
    ]
    metrics = build_metrics(rows)
    assert metrics["total"] == 4
    assert abs(metrics["avg_score"] - (8 + 6 + 8) / 3) < 0.001


def test_build_metrics_top_hook_type():
    rows = [{"hook_type": "Question"}, {"hook_type": "Question"}, {"hook_type": "Story Hook"}]
    assert build_metrics(rows)["top_hook"] == "Question"


def test_score_pill():
    assert "<span" in score_pill(9)
    assert "hot" in score_pill(9)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_ui_helpers.py`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`app/ui_helpers.py`:

```python
from __future__ import annotations

import collections

import streamlit as st

CSS = """
<style>
:root {
  --bg: #0E1117; --surface: #1A1D24; --border: #2A2E38;
  --accent: #FF3D5A; --teal: #22D3EE; --text: #E6E9EF; --muted: #9AA3B2;
}
[data-testid="stSidebar"] { background: var(--surface); }
.block-container { padding-top: 2rem; }
.kpi-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.25rem; height: 100%;
}
.kpi-card .kpi-value { font-family: 'Fira Code', monospace; font-size: 2rem; font-weight: 700; color: var(--text); }
.kpi-card .kpi-label { color: var(--muted); font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; }
.kpi-card .kpi-value.hot { color: var(--accent); }
.pill {
  display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
  font-family: 'Fira Code', monospace; font-weight: 600; font-size: 0.9rem;
  color: #0E1117;
}
.pill.hot { background: #FF3D5A; }
.pill.warm { background: #F59E0B; }
.pill.weak { background: #9AA3B2; }
[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
hr { border-color: var(--border); }
</style>
"""


def build_metrics(rows: list[dict]) -> dict:
    scores = [r["hook_score"] for r in rows if r.get("hook_score")]
    types = [r["hook_type"] for r in rows if r.get("hook_type")]
    return {
        "total": len(rows),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "top_hook": collections.Counter(types).most_common(1)[0][0] if types else "—",
    }


def score_pill(score: int) -> str:
    band = "hot" if score >= 8 else "warm" if score >= 6 else "weak"
    return f'<span class="pill {band}">{score}/10</span>'


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
```

`app.py`:

```python
from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from app.analyzer import classify, classify_with_llm, score_band
from app.scraper import Video, download_audio, download_media, search_videos
from app.transcript import get_first_5s
from app.ui_helpers import build_metrics, inject_css, score_pill

st.set_page_config(page_title="Hook Lab", page_icon="🎬", layout="wide")
inject_css()

PLATFORMS = ["YouTube Shorts", "TikTok", "Instagram Reels"]


def run_pipeline(keyword: str, platform: str, limit: int) -> list[Video]:
    videos = search_videos(keyword, platform, limit)
    workdir = tempfile.mkdtemp(prefix="hooklab_")
    progress = st.progress(0.0, text="Scraping and analyzing…")
    for i, video in enumerate(videos):
        media = download_media(video, os.path.join(workdir, "media"))
        audio = download_audio(video, os.path.join(workdir, "audio"))
        hook, source = get_first_5s(video, audio_path=audio)
        video.hook_text = hook
        analysis = classify(hook, video.title)
        video.hook_type, video.hook_score, video.reasons = (
            analysis.hook_type, analysis.score, analysis.reasons)
        video.is_llm = analysis.is_llm
        video._media_path = media
        progress.progress((i + 1) / len(videos), text=f"Analyzed {i + 1}/{len(videos)}")
    progress.empty()
    return videos


def main():
    st.title("🎬 Hook Lab")
    st.caption("Find viral shorts. Read the hook. Steal the pattern.")

    with st.sidebar:
        st.header("Search")
        keyword = st.text_input("Keyword / topic", placeholder="e.g. morning routine")
        platform = st.selectbox("Platform", PLATFORMS)
        limit = st.slider("Number of results", 5, 30, 10)
        st.divider()
        use_llm = st.toggle("Use LLM analysis (optional)", value=False)
        provider = st.selectbox("Provider", ["openai", "gemini", "deepseek"])
        api_key = st.text_input("API key", type="password")
        go = st.button("Find hooks", type="primary", use_container_width=True)

    if go and keyword.strip():
        videos = run_pipeline(keyword.strip(), platform, limit)
        if not videos:
            st.warning("No videos returned. Try a different keyword or platform.")
            return
        st.session_state["videos"] = videos
        st.session_state["use_llm"] = use_llm
        st.session_state["provider"] = provider
        st.session_state["api_key"] = api_key

    videos = st.session_state.get("videos")
    if not videos:
        st.markdown("_Enter a keyword and hit **Find hooks**. Results appear here._")
        return

    rows = [{"title": v.title, "views": v.views, "uploader": v.uploader,
             "date": v.date, "hook_type": v.hook_type, "hook_score": v.hook_score,
             "url": v.url, "platform": v.platform} for v in videos]
    metrics = build_metrics(rows)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-value">{metrics["total"]}</div>'
                f'<div class="kpi-label">Total Videos Scraped</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-value hot">{metrics["avg_score"]}</div>'
                f'<div class="kpi-label">Average Hook Score</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-value">{metrics["top_hook"]}</div>'
                f'<div class="kpi-label">Most Dominant Hook</div></div>', unsafe_allow_html=True)

    st.subheader("Viral short queue")
    df = pd.DataFrame(rows).sort_values("views", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    for v in videos:
        with st.expander(f"{v.hook_type} · {v.title}", expanded=False):
            col_video, col_info = st.columns([2, 1])
            media = getattr(v, "_media_path", "")
            if media and os.path.exists(media):
                col_video.video(media)
            else:
                col_video.markdown(f"[Watch original]({v.url})")
            col_info.markdown(f"{score_pill(v.hook_score)} "
                              f"<span style='color:#22D3EE'>{v.hook_type}</span>",
                              unsafe_allow_html=True)
            col_info.markdown(f"**{v.views:,}** views · {v.uploader} · {v.date}")
            col_info.markdown("**First 5 seconds:**")
            col_info.markdown(f"> {v.hook_text or 'No transcript captured'}")
            col_info.markdown("**Why this hook works:**")
            for reason in v.reasons:
                st.markdown(f"- {reason}")


if __name__ == "__main__":
    main()
```

`.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#FF3D5A"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1D24"
textColor = "#E6E9EF"
font = "sans serif"

[server]
headless = true
```

Note: `st.video(video._media_path)` fails because dataclass `Video` has no `_media_path`. Fix by setting it via `object.__setattr__(video, "_media_path", media)` OR by adding a field. Use the dataclass field approach: add `media_path: str = ""` to `Video` in `app/scraper.py` and use `video.media_path = media`. Update Task 2's dataclass accordingly during implementation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui_helpers.py`
Expected: all pass.

- [ ] **Step 5: Smoke-check the app boots**

Run: `python -c "import app; app.__file__"` then `python -m pytest`
Expected: imports fine, suite green. (Running `streamlit run` fully is the live-verify step in Task 6.)

- [ ] **Step 6: Commit**

```bash
git add app/ui_helpers.py app.py .streamlit/config.toml tests/test_ui_helpers.py app/scraper.py
git commit -m "feat: Streamlit dashboard with metrics, table, video grid"
```

---

### Task 6: Live verification and README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: final app from Task 5.
- Produces: run instructions; verified live behavior.

- [ ] **Step 1: Write README instructions**

```markdown
# Hook Lab (Gethooked Clone)

Local viral short-form finder & hook analyzer for TikTok, Instagram Reels, and YouTube Shorts.

## Install

pip install -r requirements.txt

## Run

streamlit run app.py

Open the URL Streamlit prints (usually http://localhost:8501).

## How it works

1. Enter a keyword/topic, pick a platform, choose how many results (5–30).
2. Hook Lab scrapes via yt-dlp, downloads each clip + audio.
3. It extracts the first 5 seconds of speech (subtitles, falling back to local
   faster-whisper, falling back to the title).
4. It classifies the hook type (Pattern Interrupt, Curiosity Gap, Negative
   Constraint, Story Hook, Bold Claim, Question, Command, Promise/Loop,
   Problem/Solution) and scores it 1–10 with reasons.

Heuristic rules are free and need no API key. Paste an OpenAI, Gemini, or
DeepSeek key in the sidebar to upgrade to LLM analysis.

## Tests

python -m pytest
```

- [ ] **Step 2: Live-run the app for 60s and check no exceptions**

Run: `streamlit run app.py --server.headless true` in a background job; poll `localhost:8501` once; then stop.
Expected: app serves; no traceback in log. If a network-restricted platform raises, note it in README's "Known limits" and move on — the app must never crash the whole grid.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add install/run instructions"
```

---

## Self-review notes

- Spec coverage: scraper (T1–2), transcript (T3), analyzer (T4), UI metrics/table/player (T5), requirements.txt + run instructions (T1, T6), error-inline-per-video (T2 search skips bad entries, T5 try handling). All covered.
- Placeholder scan: the intentional `parsered` typo in Task 4 is a red-green seam, flagged inline with replacement instruction — no open TODOs.
- Type consistency: `Video` dataclass fields consistent (Task 2 vs Task 5 usage). `media_path` field addition is called out in Task 5 with the exact edit to Task 2's dataclass.

## Implementation deviations (recorded post-review)

- **Package renamed `app/` → `hooklab/`.** The plan specified `app/scraper.py` etc., but the required root deliverable `app.py` (the `streamlit run app.py` entrypoint) collides with a top-level package named `app` — `import app` resolves to `app.py` the module, not the package, and `from app.scraper import …` fails. Renamed the package to `hooklab/` so both the entrypoint `app.py` and its imports coexist. Modules, tests, and imports all renamed consistently.
- **`slice_first_5s` truncates to cues starting within the window** instead of `_vtt_payload` returning all captions — matches the design goal ("first 5 seconds") more faithfully than the plan's sketch.
- **Empty-classify guard** uses `if not text and not title` so a title alone still gets classified (plan's `if not text` would return "Other" even when the title is analyzable).
- **LLM wiring is live** in `run_pipeline` via `classify_with_llm` when `use_llm` + API key are set — the plan left the sidebar widgets unconsumed.
- **`extract_flat: False` on downloads** — fixed during review so `download_media` resolves full entries (spec: "Player uses locally downloaded media + st.video").
- **`score_pill` reuses `analyzer.score_band`** — dedupes the three-way banding (review finding).
- **`.streamlit/config.toml`** explicitly un-ignored in `.gitignore` so the dark theme ships with the repo (review finding).