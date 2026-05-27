"""Inline chart rendering for Smart Teacher.

The teacher LLM may emit a fenced ``chart`` block inside its response when a
visualization clarifies a concept. This module parses those blocks and
returns a styled Plotly figure ready for ``st.plotly_chart``.

Block shape (JSON inside the fence)::

    ```chart
    {
      "title": "Projectile motion",
      "x": {"range": [-150, 150], "label": "horizontal position (m)"},
      "y": {"label": "height (m)"},
      "series": [
        {"type": "function", "expr": "-x**2/112.5 + 200",
         "color": "violet", "label": "trajectory"},
        {"type": "function", "expr": "-x/10 + 40",
         "color": "teal", "label": "tangent", "dash": true},
        {"type": "scatter", "data": [[-50, 175], [50, 175]],
         "color": "pink", "label": "sampled flight"}
      ],
      "vlines": [{"x": 0, "color": "orange", "dash": true}],
      "hlines": [{"y": 0, "color": "gray"}],
      "points": [{"x": 0, "y": 200, "color": "violet"}]
    }
    ```

Safety boundaries (all enforced unconditionally):

* Expressions are parsed by Python's stdlib ``ast`` in ``mode='eval'`` and
  then *every node* is checked against a tight whitelist before
  compilation. Anything outside arithmetic + the named-function whitelist
  is rejected — that includes attribute access, comprehensions, lambdas,
  imports, assignments, comparisons, dunder names, and starred args, so
  the classic Python sandbox escapes (``Integer.__class__.__bases__[0].
  __subclasses__()``, ``().__class__``, etc.) can't reach ``eval`` at all.
* Evaluation runs with ``globals={"__builtins__": None}`` plus a curated
  whitelist of numpy functions, so even if some new node type slipped
  past validation there'd be nothing useful in scope.
* Free names must be in ``{x, pi, e, sin, cos, tan, exp, log, sqrt, abs}``.
* Sample density is capped (`_MAX_POINTS`) and x-range magnitude is bounded
  (`_MAX_X_MAGNITUDE`) so a crafted block can't burn CPU/memory.
* Scatter data is capped at `_MAX_SCATTER` points.
* Non-finite (NaN / Inf) function outputs are masked so ``log(-1)`` and
  divide-by-zero don't blow up the plot.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Fenced ``` chart ``` blocks anywhere in a chat reply.
CHART_FENCE_RE = re.compile(r"```chart\s*\n(.*?)\n```", re.DOTALL)

# Palette tuned to the reference screenshot (dark canvas, soft glow). LLMs
# may emit color names *or* raw hex; both work.
PALETTE: dict[str, str] = {
    "violet": "#8B7FE8",
    "purple": "#8B7FE8",
    "indigo": "#7C7CE4",
    "teal":   "#4ED8A8",
    "green":  "#4ED8A8",
    "mint":   "#6EE7B7",
    "orange": "#E76F51",
    "red":    "#E76F51",
    "amber":  "#F0B86E",
    "yellow": "#F0B86E",
    "pink":   "#E879A0",
    "magenta": "#E879A0",
    "blue":   "#6EA8E8",
    "cyan":   "#6EE7E7",
    "gray":   "#9AA0A6",
    "grey":   "#9AA0A6",
    "white":  "#FAFAFA",
}

# AST node types that can appear in a math expression. Anything else —
# attribute access, comprehension, lambda, comparison, assignment,
# starred arg, named-expression (walrus), formatted-string, subscript —
# is rejected pre-compile. This is the *primary* safety boundary; the
# restricted globals below are belt-and-suspenders.
_ALLOWED_NODES: frozenset = frozenset({
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Name, ast.Call, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
})

# Names that may appear in an expression.
_ALLOWED_VARS: frozenset = frozenset({"x", "pi", "e"})
_ALLOWED_FUNCS: frozenset = frozenset({
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
})

# Eval namespace. __builtins__=None blocks all Python built-in name lookup,
# so even if validation ever lets something slip there's nothing reachable.
_EVAL_GLOBALS: dict[str, Any] = {
    "__builtins__": None,
    "pi": np.pi,
    "e": np.e,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}

_MAX_POINTS = 500          # per function series
_MAX_SCATTER = 1000        # per scatter series
_MAX_X_MAGNITUDE = 1.0e6   # absolute bound on either x-range endpoint


class ChartSpecError(ValueError):
    """Raised when a chart spec is structurally invalid or unsafe."""


def render(spec_text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse a chart spec and return ``(Figure, None)`` or ``(None, error)``.

    The renderer never raises out; structured errors come back as the
    second tuple element so the caller can decide whether to surface them
    in the UI (chat, PDF) or log them.
    """
    try:
        spec = json.loads(spec_text)
    except json.JSONDecodeError as e:
        return None, f"chart JSON invalid: {e.msg} (line {e.lineno})"
    if not isinstance(spec, dict):
        return None, "chart spec must be a JSON object"
    try:
        fig = _build_figure(spec)
    except ChartSpecError as e:
        return None, str(e)
    except Exception as e:                                  # defensive
        logger.exception("chart render failed unexpectedly")
        return None, f"chart render failed: {type(e).__name__}: {e}"
    return fig, None


def split_text_and_charts(text: str) -> Iterator[Tuple[str, str]]:
    """Yield ``("text", chunk)`` / ``("chart", spec_text)`` in order.

    Preserves the original interleaving so callers can rebuild the reply
    layout 1:1 in chat (markdown + figure + markdown) and in PDF.
    """
    pos = 0
    for m in CHART_FENCE_RE.finditer(text or ""):
        if m.start() > pos:
            yield "text", text[pos:m.start()]
        yield "chart", m.group(1)
        pos = m.end()
    if pos < len(text or ""):
        yield "text", text[pos:]


def figure_to_png(
    fig: Any,
    *,
    width: int = 900,
    height: int = 500,
    scale: float = 2.0,
) -> Optional[bytes]:
    """Render a Plotly Figure to PNG bytes via kaleido.

    Returns ``None`` on any failure (kaleido not installed, no Chrome,
    runtime error). Callers should treat ``None`` as "fall back to a
    text placeholder" rather than as an error.
    """
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        logger.exception("Plotly PNG export failed")
        return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _color(name: Any, fallback: str = "#8B7FE8") -> str:
    """Map a color name (or raw hex) to a hex string. Unknown → fallback."""
    if isinstance(name, str):
        cleaned = name.strip()
        if cleaned.startswith("#"):
            return cleaned
        if cleaned.lower() in PALETTE:
            return PALETTE[cleaned.lower()]
    return fallback


def _validate_ast(tree: ast.AST, source: str) -> None:
    """Walk every node in ``tree`` and reject anything outside the whitelist.

    This is the primary safety check. Once validation passes, evaluation
    can only touch arithmetic, the named whitelisted functions, and the
    variable ``x``. There is no path to attribute access, imports,
    subscripts, dunder names, comprehensions, or anything else.
    """
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ChartSpecError(
                f"expression {source!r} uses disallowed element: "
                f"{type(node).__name__}"
            )
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_VARS and node.id not in _ALLOWED_FUNCS:
                raise ChartSpecError(
                    f"expression uses unknown name {node.id!r}; "
                    f"allowed: x, {sorted(_ALLOWED_VARS | _ALLOWED_FUNCS)}"
                )
            # Belt-and-suspenders: never let a dunder name through, even
            # if our node whitelist somehow grew to include Attribute.
            if node.id.startswith("__"):
                raise ChartSpecError(
                    f"expression uses disallowed dunder name {node.id!r}"
                )
        if isinstance(node, ast.Call):
            # Only direct calls to a whitelisted function name. No
            # ``foo()()``, no ``getattr(...)(...)``, no starred args.
            if not isinstance(node.func, ast.Name):
                raise ChartSpecError(
                    "only direct function calls are allowed"
                )
            if node.func.id not in _ALLOWED_FUNCS:
                raise ChartSpecError(
                    f"function {node.func.id!r} is not allowed"
                )
            if node.keywords or any(
                isinstance(a, ast.Starred) for a in node.args
            ):
                raise ChartSpecError(
                    "function calls must use simple positional args"
                )


def _safe_eval(expr_str: str, x_values: np.ndarray) -> np.ndarray:
    """Evaluate ``expr_str`` at each value of ``x_values``.

    Uses ``ast`` validation + restricted ``eval`` globals. See module
    docstring for the full safety argument. NaN / Inf outputs are
    masked so plotting stays clean across discontinuities.
    """
    if not isinstance(expr_str, str) or not expr_str.strip():
        raise ChartSpecError("function 'expr' must be a non-empty string")

    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise ChartSpecError(
            f"could not parse expression {expr_str!r}: {e.msg}"
        ) from e

    _validate_ast(tree, expr_str)

    try:
        code = compile(tree, "<chart-expr>", "eval")
    except (SyntaxError, ValueError) as e:
        raise ChartSpecError(
            f"could not compile expression {expr_str!r}: {e}"
        ) from e

    with np.errstate(all="ignore"):
        try:
            raw = eval(code, _EVAL_GLOBALS, {"x": x_values})  # noqa: S307
        except (TypeError, NameError, ValueError, ZeroDivisionError) as e:
            raise ChartSpecError(
                f"could not evaluate expression {expr_str!r}: "
                f"{type(e).__name__}: {e}"
            ) from e

    # If the expression is a constant, eval returns a scalar — broadcast.
    y = np.broadcast_to(np.asarray(raw, dtype=float), x_values.shape).copy()
    y[~np.isfinite(y)] = np.nan
    return y


def _validate_range(x_range: Any) -> tuple[float, float]:
    if not (isinstance(x_range, (list, tuple)) and len(x_range) == 2):
        raise ChartSpecError("x.range must be a [min, max] list")
    try:
        x_min = float(x_range[0])
        x_max = float(x_range[1])
    except (TypeError, ValueError) as e:
        raise ChartSpecError(f"x.range values must be numeric: {e}") from e
    if not (x_min < x_max):
        raise ChartSpecError("x.range[0] must be strictly less than x.range[1]")
    if max(abs(x_min), abs(x_max)) > _MAX_X_MAGNITUDE:
        raise ChartSpecError(
            f"|x| must be <= {_MAX_X_MAGNITUDE:g}"
        )
    return x_min, x_max


def _build_figure(spec: dict) -> Any:
    import plotly.graph_objects as go  # lazy import keeps cold start cheap

    fig = go.Figure()

    x_cfg = spec.get("x") or {}
    y_cfg = spec.get("y") or {}
    if not isinstance(x_cfg, dict) or not isinstance(y_cfg, dict):
        raise ChartSpecError("'x' and 'y' must be JSON objects")

    # Determine the x-domain. Default to [-10, 10] if not specified — covers
    # most "show me sin(x)" cases the LLM might emit without explicit bounds.
    x_min, x_max = _validate_range(x_cfg.get("range", [-10, 10]))
    x_values = np.linspace(x_min, x_max, _MAX_POINTS)

    series = spec.get("series") or []
    if not isinstance(series, list):
        raise ChartSpecError("'series' must be a list")

    for idx, s in enumerate(series):
        if not isinstance(s, dict):
            raise ChartSpecError(f"series[{idx}] must be a JSON object")
        s_type = s.get("type", "function")
        color = _color(s.get("color", "violet"))
        dash = "dash" if s.get("dash") else None
        label = s.get("label")

        if s_type == "function":
            y_values = _safe_eval(s.get("expr"), x_values)
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values,
                mode="lines",
                line=dict(color=color, width=3, dash=dash),
                name=label if label else None,
                showlegend=bool(label),
                hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
            ))

        elif s_type == "scatter":
            data = s.get("data")
            if not isinstance(data, list):
                raise ChartSpecError(
                    f"series[{idx}] scatter 'data' must be a list of [x, y] pairs"
                )
            if len(data) > _MAX_SCATTER:
                raise ChartSpecError(
                    f"series[{idx}] scatter data exceeds {_MAX_SCATTER} points"
                )
            xs: list[float] = []
            ys: list[float] = []
            for j, pt in enumerate(data):
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    raise ChartSpecError(
                        f"series[{idx}].data[{j}] must be a [x, y] pair"
                    )
                try:
                    xs.append(float(pt[0]))
                    ys.append(float(pt[1]))
                except (TypeError, ValueError) as e:
                    raise ChartSpecError(
                        f"series[{idx}].data[{j}] not numeric: {e}"
                    ) from e
            mode = "lines+markers" if s.get("connect") else "markers"
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode=mode,
                marker=dict(color=color, size=10, line=dict(width=0)),
                line=dict(color=color, width=2, dash=dash),
                name=label if label else None,
                showlegend=bool(label),
                hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
            ))

        else:
            raise ChartSpecError(
                f"series[{idx}] unsupported type {s_type!r}; "
                "expected 'function' or 'scatter'"
            )

    for vl in spec.get("vlines") or []:
        if not isinstance(vl, dict):
            continue
        try:
            x_val = float(vl.get("x", 0))
        except (TypeError, ValueError):
            continue
        fig.add_vline(
            x=x_val,
            line=dict(
                color=_color(vl.get("color", "orange")),
                dash="dash" if vl.get("dash", True) else None,
                width=2,
            ),
        )

    for hl in spec.get("hlines") or []:
        if not isinstance(hl, dict):
            continue
        try:
            y_val = float(hl.get("y", 0))
        except (TypeError, ValueError):
            continue
        fig.add_hline(
            y=y_val,
            line=dict(
                color=_color(hl.get("color", "gray")),
                dash="dash" if hl.get("dash", True) else None,
                width=2,
            ),
        )

    for p in spec.get("points") or []:
        if not isinstance(p, dict):
            continue
        try:
            px = float(p["x"])
            py = float(p["y"])
        except (KeyError, TypeError, ValueError):
            continue
        fig.add_trace(go.Scatter(
            x=[px], y=[py],
            mode="markers",
            marker=dict(
                color=_color(p.get("color", "violet")),
                size=14, line=dict(width=0),
            ),
            showlegend=False,
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
        ))

    title = spec.get("title") or ""
    show_legend = any(
        isinstance(s, dict) and s.get("label")
        for s in series
    )
    fig.update_layout(
        title=title or None,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA", size=13),
        xaxis=dict(
            title=str(x_cfg.get("label", "")),
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.18)",
        ),
        yaxis=dict(
            title=str(y_cfg.get("label", "y")),
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.18)",
        ),
        margin=dict(l=55, r=20, t=50 if title else 20, b=45),
        showlegend=show_legend,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def chart_title(spec_text: str) -> str:
    """Best-effort title extraction for placeholders.

    Used by the PDF fallback when kaleido isn't available — gives the
    reader a hint of what was supposed to be drawn here.
    """
    try:
        spec = json.loads(spec_text)
        title = spec.get("title") if isinstance(spec, dict) else None
        if isinstance(title, str) and title.strip():
            return title.strip()
    except (json.JSONDecodeError, AttributeError):
        pass
    return "untitled"
