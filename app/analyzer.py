from __future__ import annotations

import json
from dataclasses import dataclass, field

HOOK_TYPES = [
    "Pattern Interrupt",
    "Curiosity Gap",
    "Negative Constraint",
    "Story Hook",
    "Bold Claim",
    "Question",
    "Command",
    "Promise/Loop",
    "Problem/Solution",
    "Other",
]


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
    (
        "Negative Constraint",
        ["don't", "dont", "not what", "never", "stop doing", "isn't", "is not",
         "aren't", "wont", "stop guessing"],
        8,
        ["Opens with friction that interrupts the scroll",
         "A 'not/never' frame creates an information gap",
         "Viewers stay to see what is being forbidden"],
    ),
    (
        "Question",
        ["?", "why", "how", "what", "do you", "did you", "are you", "can you",
         "who", "when"],
        7,
        ["Question openers recruit the viewer's own curiosity",
         "It feels conversational and direct",
         "Sets up a natural payoff in the body"],
    ),
    (
        "Pattern Interrupt",
        ["stop scrolling", "wait", "hold on", "listen", "watch this", "pov:",
         "you're gonna", "you won't believe", "i can't", "psst"],
        9,
        ["Jumps the feed's momentum with a jarring address",
         "Direct second-person taps the viewer's attention",
         "Implies a payoff worth sticking around for"],
    ),
    (
        "Curiosity Gap",
        ["the one trick", "nobody tells", "secret", "the reason", "you need to know",
         "how to", "in your", "that nobody", "most people", "what nobody"],
        7,
        ["Dangles a missing piece the viewer wants filled",
         "Speaks to a specific audience pain or desire",
         "Suggests insider knowledge unavailable elsewhere"],
    ),
    (
        "Story Hook",
        ["i was", "so i", "i remember", "the day", "it started", "a few years",
         "back when", "my friend", "i walked", "then i"],
        7,
        ["Drops the viewer into an ongoing narrative",
         "Personal stakes build emotional investment",
         "The unresolved plot compels the next few seconds"],
    ),
    (
        "Bold Claim",
        ["number one", "the best", "the fastest", "guaranteed", "the only",
         "this works", "100%", "secret that", "biggest mistake", "truth about"],
        9,
        ["A definitive, high-stakes assertion stands out",
         "Confident framing transfers trust",
         "The claim is testable, inviting challenge and stay"],
    ),
    (
        "Command",
        ["do this", "try this", "watch", "listen to", "save this", "write this",
         "pay attention", "look at"],
        7,
        ["A direct instruction creates an active audience",
         "Commands feel actionable and useful",
         "Frames the video as a how-to payoff"],
    ),
    (
        "Promise/Loop",
        ["this video", "the end", "wait for it", "watch until", "i'll show you",
         "learn how", "see what happens", "stick around", "trust me"],
        8,
        ["Promises a concrete reward for staying",
         "Creates a watch-to-the-end loop",
         "Explicitly values the viewer's time"],
    ),
    (
        "Problem/Solution",
        ["problem", "issue", "struggle", "suffer", "frustrating", "if you",
         "annoying", "fix", "solution", "mistake"],
        6,
        ["Names a pain the viewer already feels",
         "Positioning the video as the fix keeps them watching",
         "Problem-recognition hooks are broadly relatable"],
    ),
]


def classify(text: str, title: str = "") -> Analysis:
    if not text and not title:
        return Analysis("Other", 3, ["No transcript available to judge the hook"])
    combined = f"{text} {title}".lower()
    for hook_type, patterns, score, reasons in _RULES:
        if _has_any(combined, patterns):
            n = sum(1 for p in patterns if p in combined)
            boosted = min(10, score + (n // 2))
            return Analysis(hook_type, boosted, reasons)
    return Analysis(
        "Pattern Interrupt",
        6,
        ["No named pattern matched",
         "Defaulted to an attention-grabbing read",
         "The literal text makes a strong first impression"],
    )


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
            resp = c.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            payload = resp.choices[0].message.content
        elif provider.lower() == "deepseek":
            from openai import OpenAI

            c = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            resp = c.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
            )
            payload = resp.choices[0].message.content
        elif provider.lower() == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            m = genai.GenerativeModel("gemini-1.5-flash")
            payload = m.generate_content(prompt).text
        else:
            raise ValueError(f"unknown provider {provider!r}")
        parsed = _parse_llm(payload)
        if parsed:
            parsed["is_llm"] = True
            return Analysis(**parsed)
    except Exception:
        pass
    return classify(text, title)