from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from hooklab.analyzer import classify, classify_with_llm
from hooklab.scraper import Video, download_audio, download_media, search_videos
from hooklab.transcript import get_first_5s
from hooklab.ui_helpers import build_metrics, inject_css, score_pill

st.set_page_config(page_title="Hook Lab", page_icon="🎬", layout="wide")
inject_css()

PLATFORMS = ["YouTube Shorts", "TikTok", "Instagram Reels"]


def run_pipeline(keyword: str, platform: str, limit: int, use_llm=False,
                 provider="openai", api_key="") -> list[Video]:
    videos = search_videos(keyword, platform, limit)
    workdir = tempfile.mkdtemp(prefix="hooklab_")
    progress = st.progress(0.0, text="Scraping and analyzing…")
    for i, video in enumerate(videos):
        try:
            media = download_media(video, os.path.join(workdir, "media"))
        except Exception:
            media = ""
        try:
            audio = download_audio(video, os.path.join(workdir, "audio"))
        except Exception:
            audio = ""
        hook, _source = get_first_5s(video, audio_path=audio)
        video.hook_text = hook
        if use_llm and api_key:
            analysis = classify_with_llm(hook, video.title, provider, api_key)
        else:
            analysis = classify(hook, video.title)
        video.hook_type, video.hook_score, video.reasons = (
            analysis.hook_type,
            analysis.score,
            analysis.reasons,
        )
        video.is_llm = analysis.is_llm
        video.media_path = media
        progress.progress((i + 1) / len(videos), text=f"Analyzed {i + 1}/{len(videos)}")
    progress.empty()
    return videos


def main():
    st.title(":red[Hook] Lab")
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
        with st.spinner("Scraping viral shorts…"):
            videos = run_pipeline(
                keyword.strip(), platform, limit, use_llm=use_llm,
                provider=provider, api_key=api_key,
            )
        if not videos:
            st.warning("No videos returned. Try a different keyword or platform.")
            return
        st.session_state["videos"] = videos

    videos = st.session_state.get("videos")
    if not videos:
        st.markdown("_Enter a keyword, pick a platform, and hit **Find hooks**. Results appear here._")
        return

    rows = [
        {
            "title": v.title,
            "views": v.views,
            "uploader": v.uploader,
            "date": v.date,
            "hook_type": v.hook_type,
            "hook_score": v.hook_score,
            "url": v.url,
            "platform": v.platform,
        }
        for v in videos
    ]
    metrics = build_metrics(rows)
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{metrics["total"]}</div>'
        f'<div class="kpi-label">Total Videos Scraped</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="kpi-card"><div class="kpi-value hot">{metrics["avg_score"]:.1f}</div>'
        f'<div class="kpi-label">Average Hook Score</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{metrics["top_hook"]}</div>'
        f'<div class="kpi-label">Most Dominant Hook</div></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Viral short queue")

    @st.cache_data
    def _frame(rows):
        return pd.DataFrame(rows)

    df = _frame(rows).sort_values("views", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    for v in videos:
        with st.expander(f"{v.hook_type} · {v.title}", expanded=False):
            col_video, col_info = st.columns([2, 1])
            if v.media_path and os.path.exists(v.media_path):
                col_video.video(v.media_path)
            else:
                col_video.markdown(f"[Watch original]({v.url})", unsafe_allow_html=True)
            col_info.markdown(
                f"{score_pill(v.hook_score)} "
                f"<span style='color:#22D3EE'>{v.hook_type}</span>",
                unsafe_allow_html=True,
            )
            col_info.markdown(f"**{v.views:,}** views · {v.uploader} · {v.date}")
            col_info.markdown("**First 5 seconds:**")
            col_info.markdown(f"> {v.hook_text or 'No transcript captured'}")
            col_info.markdown("**Why this hook works:**")
            for reason in v.reasons:
                st.markdown(f"- {reason}")


if __name__ == "__main__":
    main()