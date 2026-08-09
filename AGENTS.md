# AGENTS.md

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues. Use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Project

Gethooked Clone — a local Streamlit app that finds viral short-form videos (TikTok, Instagram Reels, YouTube Shorts) via yt-dlp and analyzes their opening hooks with an LLM or heuristic rules.

## Commands

- Install dependencies: `pip install -r requirements.txt`
- Run the app: `streamlit run app.py`
- Run tests: `python -m pytest`