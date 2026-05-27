"""Tests for the inline chart renderer.

Each case here pins down a property of the safe-eval / parse boundary, so
a future refactor that loosens those bounds gets caught immediately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import charts  # noqa: E402


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_function_series_renders() -> None:
    spec = json.dumps({
        "title": "parabola",
        "x": {"range": [-10, 10], "label": "x"},
        "y": {"label": "y"},
        "series": [
            {"type": "function", "expr": "x**2", "color": "violet",
             "label": "y = x²"},
        ],
    })
    fig, err = charts.render(spec)
    assert err is None
    assert fig is not None
    # One trace, lots of points
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 500
    # The trace's middle y should be near 0 since the domain is symmetric
    middle = list(fig.data[0].y)[250]
    assert abs(middle) < 1.0


def test_scatter_series_renders() -> None:
    spec = json.dumps({
        "x": {"range": [0, 10]},
        "y": {"label": "obs"},
        "series": [
            {"type": "scatter", "data": [[1, 2], [2, 4], [3, 6]],
             "color": "pink", "label": "pts"},
        ],
    })
    fig, err = charts.render(spec)
    assert err is None
    assert list(fig.data[0].x) == [1, 2, 3]
    assert list(fig.data[0].y) == [2, 4, 6]


def test_vlines_hlines_points_all_render() -> None:
    spec = json.dumps({
        "x": {"range": [-5, 5]},
        "y": {"label": "y"},
        "series": [{"type": "function", "expr": "x", "color": "violet"}],
        "vlines": [{"x": 0, "color": "orange", "dash": True}],
        "hlines": [{"y": 0, "color": "gray"}],
        "points": [{"x": 0, "y": 0, "color": "teal"}],
    })
    fig, err = charts.render(spec)
    assert err is None
    # 1 function trace + 1 point trace
    assert len(fig.data) == 2
    # vline + hline are layout shapes, not traces
    shapes = fig.layout.shapes or ()
    assert len(shapes) == 2


# ---------------------------------------------------------------------------
# Safety boundary — every one of these MUST be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwn')",
        "os.system('rm -rf /')",
        "open('/etc/passwd').read()",
        "exec('print(1)')",
        "eval('1+1')",
        "y + 1",                         # unknown symbol 'y'
        "x + a",                         # unknown symbol 'a'
        # Classic Python-sandbox-escape attempts. ast.Attribute is not in
        # _ALLOWED_NODES so all of these are rejected at validation time.
        "().__class__",
        "(1).__class__.__bases__[0].__subclasses__()",
        "x.__class__",
        "x.real",
        # Comprehensions / lambdas / starred args / kwargs are off the menu.
        "[x for _ in [0]]",
        "(lambda x: x)(x)",
        "sin(*[x])",
        "sin(x, base=2)",
        # Subscripting and starred unpacking — neither node type is allowed.
        "x[0]",
        # Walrus / named expression.
        "(y := x)",
    ],
)
def test_unsafe_or_unknown_expressions_rejected(expr: str) -> None:
    spec = json.dumps({
        "x": {"range": [-1, 1]},
        "series": [{"type": "function", "expr": expr}],
    })
    fig, err = charts.render(spec)
    # The contract is "unsafe input is rejected and never reaches eval";
    # the exact wording of the error message is allowed to evolve.
    assert fig is None
    assert err is not None and err


def test_x_range_magnitude_capped() -> None:
    spec = json.dumps({
        "x": {"range": [-1e9, 1e9]},
        "series": [{"type": "function", "expr": "x"}],
    })
    fig, err = charts.render(spec)
    assert fig is None
    assert "|x|" in err


def test_inverted_range_rejected() -> None:
    spec = json.dumps({
        "x": {"range": [5, -5]},
        "series": [{"type": "function", "expr": "x"}],
    })
    fig, err = charts.render(spec)
    assert fig is None
    assert err is not None


def test_scatter_data_cap() -> None:
    big = [[i, i] for i in range(1500)]
    spec = json.dumps({
        "x": {"range": [0, 100]},
        "series": [{"type": "scatter", "data": big}],
    })
    fig, err = charts.render(spec)
    assert fig is None
    assert "1000" in err or "exceeds" in err


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------


def test_nan_and_inf_masked() -> None:
    # log over a range that includes negatives -> NaN; 1/x at 0 -> Inf.
    spec = json.dumps({
        "x": {"range": [-1, 1]},
        "series": [{"type": "function", "expr": "log(x)"}],
    })
    fig, err = charts.render(spec)
    assert err is None
    ys = np.asarray(fig.data[0].y, dtype=float)
    # No raw +/-Inf reaches the trace; NaN is allowed and gets a gap.
    assert not np.any(np.isposinf(ys))
    assert not np.any(np.isneginf(ys))


def test_constant_expression_broadcasts() -> None:
    spec = json.dumps({
        "x": {"range": [0, 1]},
        "series": [{"type": "function", "expr": "pi"}],
    })
    fig, err = charts.render(spec)
    assert err is None
    ys = list(fig.data[0].y)
    assert len(ys) == 500
    assert all(abs(y - np.pi) < 1e-9 for y in ys)


# ---------------------------------------------------------------------------
# Parser plumbing
# ---------------------------------------------------------------------------


def test_split_text_and_charts_preserves_order() -> None:
    payload = (
        "Intro paragraph.\n\n"
        "```chart\n"
        "{\"x\":{\"range\":[-1,1]},\"series\":[{\"type\":\"function\",\"expr\":\"x\"}]}\n"
        "```\n\n"
        "Outro paragraph."
    )
    chunks = list(charts.split_text_and_charts(payload))
    kinds = [k for k, _ in chunks]
    assert kinds == ["text", "chart", "text"]
    assert "Intro paragraph" in chunks[0][1]
    assert "Outro paragraph" in chunks[2][1]


def test_text_without_chart_yields_single_text_chunk() -> None:
    chunks = list(charts.split_text_and_charts("just some prose"))
    assert chunks == [("text", "just some prose")]


def test_invalid_json_returns_structured_error() -> None:
    fig, err = charts.render("{not valid json}")
    assert fig is None
    assert "JSON invalid" in err


def test_chart_title_extracts_or_falls_back() -> None:
    assert charts.chart_title('{"title": "Parabola"}') == "Parabola"
    assert charts.chart_title("{}") == "untitled"
    assert charts.chart_title("nonsense") == "untitled"
