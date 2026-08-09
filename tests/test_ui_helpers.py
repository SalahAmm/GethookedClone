from hooklab.ui_helpers import build_metrics, score_pill, CSS


def test_build_metrics_counts():
    rows = [
        {"hook_score": 8},
        {"hook_score": 6},
        {"hook_score": 8},
        {},
    ]
    metrics = build_metrics(rows)
    assert metrics["total"] == 4
    assert abs(metrics["avg_score"] - (8 + 6 + 8) / 3) < 0.001


def test_build_metrics_top_hook_type():
    rows = [{"hook_type": "Question"}, {"hook_type": "Question"}, {"hook_type": "Story Hook"}]
    assert build_metrics(rows)["top_hook"] == "Question"


def test_build_metrics_empty():
    assert build_metrics([]) == {"total": 0, "avg_score": 0, "top_hook": "—"}


def test_score_pill():
    assert "<span" in score_pill(9)
    assert "hot" in score_pill(9)


def test_score_pill_bands():
    assert "warm" in score_pill(6)
    assert "weak" in score_pill(3)


def test_css_contains_theme_vars():
    assert "--accent: #FF3D5A" in CSS
    assert ".kpi-card" in CSS