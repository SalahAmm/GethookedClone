# Hook Lab (Gethooked Clone)

Local viral short-form finder and hook analyzer for TikTok, Instagram Reels, and YouTube Shorts. Free, runs entirely on your machine, no paid central service.

## Install

```
pip install -r requirements.txt
```

## Run

```
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## How it works

1. Enter a keyword/topic, pick a platform, choose how many results (5–30).
2. Hook Lab scrapes viral shorts via yt-dlp and downloads each clip + audio.
3. It extracts the first 5 seconds of speech — captions first for speed, falling back to local faster-whisper transcription, then the video title.
4. It classifies the hook type (Pattern Interrupt, Curiosity Gap, Negative Constraint, Story Hook, Bold Claim, Question, Command, Promise/Loop, Problem/Solution) and scores it 1–10 with a short "why it works" breakdown.

Heuristic rules are free and need no API key. Paste an OpenAI, Gemini, or DeepSeek key in the sidebar to upgrade to LLM analysis.

## Tests

```
python -m pytest
```

Unit tests run offline — no live network, no downloads.

## Known limits

- TikTok and Instagram may rate-limit or bot-check scraper sessions; a video that fails to fetch is skipped inline and never crashes the grid. YouTube Shorts is the most reliable platform.
- A few platforms are stricter on some networks; if a search returns nothing, try a different keyword or switch platforms.