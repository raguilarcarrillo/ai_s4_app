"""Regression tests for pdf_export.

Each case here exists because a real user-visible failure happened. Don't
delete one without understanding which crash it pins down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``app/`` importable without installing the package.
_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from pdf_export import _flatten_markdown_links, markdown_to_pdf_bytes


@pytest.mark.parametrize(
    "body",
    [
        # The exact failure mode from the production traceback: an internal
        # anchor link with a numeric destination. fpdf2 2.8.7 turns this
        # into a placeholder named destination and crashes pdf.output()
        # unless we flatten it first.
        "Inline [1](#1) reference.",
        # Real LLM-shape variant — a section link.
        "See [the previous section](#section-1) for details.",
        # Multiple anchor links in one body, including bold around them.
        "Mix of **[ref a](#a)** and *[ref b](#b)* in one paragraph.",
        # An invalid-shape URL that previously also crashed.
        "Cited as [notes.pdf#3](notes.pdf#3) in body.",
    ],
)
def test_anchor_links_do_not_crash_output(body: str) -> None:
    """Generating a PDF must succeed even when the body contains
    ``[label](#anchor)``-style links. Without the flatten in
    ``pdf_export._prepare_text``, fpdf2 raises
    ``FPDFException: Named destination '...' was referenced but never set``.
    """
    out = markdown_to_pdf_bytes("Regression", body)
    assert out[:4] == b"%PDF", "expected a valid PDF header"


def test_flatten_keeps_label_drops_url() -> None:
    """Direct test of the helper: label survives, url is stripped."""
    assert _flatten_markdown_links("[hi](#x) world") == "hi world"
    assert _flatten_markdown_links("plain text") == "plain text"
    assert (
        _flatten_markdown_links("a [one](url1) b [two](url2) c")
        == "a one b two c"
    )


def test_bold_and_italic_still_render() -> None:
    """The flatten must not collide with surrounding markdown emphasis."""
    out = markdown_to_pdf_bytes(
        "Style check",
        "Some **bold** and *italic* text with a [link](#x).",
    )
    assert out[:4] == b"%PDF"
