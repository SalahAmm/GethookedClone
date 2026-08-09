# Gethooked Clone — Design

Date: 2026-08-09
Status: Approved

## Purpose

A free, local Streamlit app that finds viral short-form videos (TikTok, Instagram Reels, YouTube Shorts) the way Gethooked does, and analyzes their opening hooks. Runs entirely on the user's PC — no paid central service, no mandatory API key.

## Success criteria

1. A keyword/topic search returns 5–30 viral short videos from any of the three platforms via yt-dlp, each with title, view count, uploader, upload date, direct video URL, and thumbnail.
2. Each video shows the transcript of its first 5 seconds (the hook).
3. Each hook is classified into a hook type (Pattern Interrupt, Curiosity Gap, Negative Constraint, Story Hook, Bold Claim, Question, Command, Problem/Solution, Promise/Loop) with a Hook Score 1–10 and a bulleted "why it works" breakdown.
4. Runs free out of the box (heuristic rules); an optional LLM API key upgrades analysis.
5. Never crashes on network/restriction/missing-data failures — per-video errors surface inline.

## Constraints

- Free & local by default — no API key required to use the app.
- Reliable on the user's PC: caption-first transcript with local whisper fallback only when captions are absent.
- Platform: all three (TikTok, Reels, YouTube Shorts) via yt-dlp search.
- Player: yt-dlp downloads media to a local `downloads/` dir; `st.video` plays it.

## Architecture

Four focused modules, each independently testable:

| File | Responsibility |
| ---- | -------------- |
| `app.py` | Streamlit UI: sidebar, metrics bar, sortable table, expandable video cards. |
| `scraper.py` | yt-dlp wrapper. Searches shorts by keyword per platform; extracts + normalizes metadata; downloads media & audio to `downloads/`. |
| `transcript.py` | Gets the first-5s transcript: subtitle/caption track first, `faster-whisper` fallback when captions are absent. |
| `analyzer.py` | Hook classifier: heuristic rules (default) + optional LLM (OpenAI / Gemini / DeepSeek). Returns hook type, score, reasons. |

## Data flow

1. Sidebar: keyword, platform selector, result count slider (5–30), optional API key.
2. `scraper.search(keyword, platform, n)` → list of normalized video dicts (id, title, views, uploader, date, url, thumbnail, duration).
3. For each video: `scraper.download_media()` gets the playable clip to `downloads/`; `scraper.download_audio()` gets audio for transcription when needed.
4. `transcript.get_first_5s(video)` → subtitle track → slice first 5s of text; fallback faster-whisper on audio; final fallback title+description.
5. `analyzer.classify(hook_text, thumb)` → (hook_type, hook_score, reasons[3], is_llm).
6. UI renders: 3 metric cards, sortable pandas table, expandable video card per video (`st.video`, transcript, breakdown).

## Hook types (heuristic vocabulary)

Pattern Interrupt, Curiosity Gap, Negative Constraint, Story Hook, Bold Claim, Question, Command, Promise/Loop, Problem/Solution, Other.

## Error handling

- yt-dlp exceptions (network, bot-check, geo): captured per-video, shown inline, rest of grid unaffected.
- Missing subtitles → whisper fallback → title fallback. Never raise to the user.
- Missing/invalid LLM key → silently degrade to heuristic scoring.
- Timeout guard on network calls; small batch + progress bar in UI.

## Testing

- `pytest` for: scraper metadata normalization (fixtures, no live network); transcript slicing logic; analyzer classification against fixture hooks per type.
- Live yt-dlp calls skipped in unit tests via a network stub. An opt-in `--live` marker runs real fetches.
- LLM path tested with a fake client; heuristic path is the deterministic default asserted in CI.

## Out of scope (v1)

- User accounts/auth, persistence/database layer (in-memory session only).
- Scheduling/automation, export, multi-keyword pipelines.
- Caption/transcript of the full video (only the hook window).