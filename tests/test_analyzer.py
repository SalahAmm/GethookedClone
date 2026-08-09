import json

from app.analyzer import HOOK_TYPES, Analysis, classify, classify_with_llm, score_band


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
    assert a.hook_type == "Other"


def test_classify_uses_title_when_text_empty():
    a = classify("", title="Why your morning routine is broken")
    assert a.hook_type == "Question"
    assert a.score > 3


def test_score_band():
    assert score_band(9) == "hot"
    assert score_band(6) == "warm"
    assert score_band(3) == "weak"


class _FakeLLMClient:
    def build(self, prompt):
        return json.dumps(
            {
                "hook_type": "Bold Claim",
                "score": 9,
                "reasons": [
                    "Makes a strong claim",
                    "Specific payoff",
                    "Confident framing",
                ],
            }
        )


def test_classify_with_llm_parses_response():
    result = classify_with_llm("text here", "title", "openai", "sk-test", client=_FakeLLMClient())
    assert result.is_llm is True
    assert result.hook_type == "Bold Claim"
    assert result.score == 9


def test_classify_with_llm_bad_response_falls_back():
    result = classify_with_llm("What if it all changed?", "t", "openai", "sk-test", client=object())
    assert result.is_llm is False
    assert result.hook_type in HOOK_TYPES