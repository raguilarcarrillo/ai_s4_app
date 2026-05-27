"""PDF export helpers for Smart Teacher.

Two public entry points:

* :func:`markdown_to_pdf_bytes` — render a title + markdown body into an
  in-memory PDF (returned as ``bytes``).
* :func:`save_pdf_to_downloads` — write the same PDF under
  ``~/Downloads/`` with a short ``<topic>_<YYYY-MM-DD>.pdf`` filename and
  return the absolute :class:`Path`.

Rendering pipeline:

1. **Font selection** — at first use we probe ``/usr/share/fonts/truetype/``
   for Noto Sans (proportional, 4 styles) and DejaVu Sans Mono. When both
   are present we register them with fpdf2 and use them everywhere. When
   absent (non-Linux dev box), we fall back to the built-in Helvetica /
   Courier latin-1 fonts and apply a sanitization pass to swap unicode
   chars for ASCII look-alikes.
2. **Emoji handling** — color emojis can't render in fpdf2 (no CBDT/CBLC
   support), so they are replaced with short text tags ``[MAP]``,
   ``[EXERCISES]``, ``[OK]`` … or stripped if unmapped.
3. **Block parsing** — the body is walked line by line. Headings, bullets,
   numbered lists, blockquotes, horizontal rules, fenced code blocks and
   pipe-style markdown tables are each routed to a dedicated renderer.
   Anything else falls through to plain paragraphs so unknown markdown
   degrades gracefully rather than throwing.
4. **Inline formatting** — ``**bold**`` / ``*italic*`` ride through
   fpdf2's ``multi_cell(markdown=True)``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Final, Iterable

from fpdf import FPDF
from fpdf.enums import XPos, YPos

logger = logging.getLogger(__name__)

DOWNLOADS_DIR: Final[Path] = Path.home() / "Downloads"

# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

# System paths probed at runtime. First family with a complete 4-style set
# wins. Order matters: Noto Sans is preferred for its full italic coverage.
_FONT_SEARCH_DIRS: Final[tuple[str, ...]] = (
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/TTF",
    "/usr/local/share/fonts",
)

_BODY_FONT_CANDIDATES: Final[tuple[tuple[str, dict[str, str]], ...]] = (
    (
        "NotoSans",
        {
            "": "NotoSans-Regular.ttf",
            "B": "NotoSans-Bold.ttf",
            "I": "NotoSans-Italic.ttf",
            "BI": "NotoSans-BoldItalic.ttf",
        },
    ),
    (
        "DejaVuSans",
        {
            "": "DejaVuSans.ttf",
            "B": "DejaVuSans-Bold.ttf",
            "I": "DejaVuSans-Oblique.ttf",
            "BI": "DejaVuSans-BoldOblique.ttf",
        },
    ),
)

_MONO_FONT_CANDIDATES: Final[tuple[tuple[str, dict[str, str]], ...]] = (
    (
        "DejaVuSansMono",
        {
            "": "DejaVuSansMono.ttf",
            "B": "DejaVuSansMono-Bold.ttf",
            "I": "DejaVuSansMono-Oblique.ttf",
            "BI": "DejaVuSansMono-BoldOblique.ttf",
        },
    ),
)


def _find_font_files(file_map: dict[str, str]) -> dict[str, Path] | None:
    """Resolve every variant in ``file_map`` to a real path, or return None.

    Only returns a result when at least the regular and bold variants are
    present; italic/bold-italic are best-effort and silently dropped when
    missing.
    """
    resolved: dict[str, Path] = {}
    for style, filename in file_map.items():
        for base in _FONT_SEARCH_DIRS:
            candidate = Path(base) / filename
            if candidate.exists():
                resolved[style] = candidate
                break
    if "" not in resolved or "B" not in resolved:
        return None
    return resolved


def _pick_font(
    candidates: Iterable[tuple[str, dict[str, str]]],
) -> tuple[str, dict[str, Path]] | None:
    """Return ``(family, paths)`` for the first candidate with regular+bold."""
    for family, file_map in candidates:
        paths = _find_font_files(file_map)
        if paths:
            return family, paths
    return None


class _FontSet:
    """Resolved family names that will be passed to ``set_font``.

    Falls back to Helvetica / Courier (and the sanitization pass) when no
    unicode TTF was found.
    """

    def __init__(
        self,
        body_family: str,
        mono_family: str,
        unicode_ok: bool,
    ) -> None:
        self.body = body_family
        self.mono = mono_family
        self.unicode_ok = unicode_ok


def _install_fonts(pdf: FPDF) -> _FontSet:
    """Register the best available fonts on this fpdf instance.

    Returns the family names to use for body and monospace text plus a
    flag indicating whether unicode characters can be passed through
    unsanitized.
    """
    body = _pick_font(_BODY_FONT_CANDIDATES)
    mono = _pick_font(_MONO_FONT_CANDIDATES)
    if not body or not mono:
        logger.info("PDF export: unicode fonts not found, using Helvetica fallback")
        return _FontSet(body_family="Helvetica", mono_family="Courier", unicode_ok=False)
    body_family, body_paths = body
    mono_family, mono_paths = mono
    for style, path in body_paths.items():
        pdf.add_font(body_family, style=style, fname=str(path))
    for style, path in mono_paths.items():
        pdf.add_font(mono_family, style=style, fname=str(path))
    return _FontSet(body_family=body_family, mono_family=mono_family, unicode_ok=True)


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

# Common emojis Smart Teacher's prompts emit. Map them to short bracketed
# tags so meaning is preserved when the glyphs themselves can't render.
_EMOJI_TAGS: Final[dict[str, str]] = {
    "🎓": "[GRAD]",
    "📚": "[BOOKS]",
    "📖": "[BOOK]",
    "📝": "[NOTE]",
    "📊": "[CHART]",
    "📈": "[CHART]",
    "📉": "[CHART]",
    "📌": "[PIN]",
    "📍": "[PIN]",
    "📎": "[CLIP]",
    "📂": "[FOLDER]",
    "📁": "[FOLDER]",
    "🗺️": "[MAP]",
    "🗺": "[MAP]",
    "🖼️": "[IMAGE]",
    "🖼": "[IMAGE]",
    "🏋️": "[EXERCISES]",
    "🏋": "[EXERCISES]",
    "🎯": "[TARGET]",
    "🎲": "[RANDOM]",
    "🧠": "[THINK]",
    "💡": "[TIP]",
    "💬": "[CHAT]",
    "💭": "[THOUGHT]",
    "🔑": "[KEY]",
    "🔍": "[FIND]",
    "🔎": "[FIND]",
    "🔥": "[HOT]",
    "⭐": "[STAR]",
    "🌟": "[STAR]",
    "✨": "[NEW]",
    "✅": "[OK]",
    "✔️": "[OK]",
    "✔": "[OK]",
    "❌": "[X]",
    "❎": "[X]",
    "⚠️": "[WARN]",
    "⚠": "[WARN]",
    "❓": "[?]",
    "❗": "[!]",
    "⚡": "[FAST]",
    "🚀": "[GO]",
    "🛠️": "[TOOLS]",
    "🛠": "[TOOLS]",
    "🔧": "[TOOL]",
    "🔨": "[TOOL]",
    "🧪": "[LAB]",
    "🧮": "[CALC]",
    "🤖": "[BOT]",
    "👍": "[+1]",
    "👎": "[-1]",
    "🟢": "[GREEN]",
    "🔴": "[RED]",
    "🟡": "[YELLOW]",
    "🔵": "[BLUE]",
}

# Catch-all: any remaining codepoint inside the typical emoji blocks.
# Build a single compiled regex from the Unicode emoji blocks we care about
# so unmapped emojis are dropped instead of turning into tofu boxes.
_EMOJI_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    "(?:["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, etc.
    "\U0001F600-\U0001F64F"  # emoticons
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"
    "️"                 # variation selector-16 (emoji style)
    "‍"                 # zero-width joiner
    "]+ ?)+",                 # also eat an optional trailing space after the run
    flags=re.UNICODE,
)


def _emoji_to_tags(text: str) -> str:
    """Strip emojis entirely.

    Emojis don't render in fpdf2 (no color-bitmap support) and the
    tag-replacement variant (``[MAP]``, ``[OK]`` …) added visual noise to
    every heading. The headings themselves carry meaning, so we just drop
    the glyphs and let the bold heading text stand on its own.

    Both the explicit known-emoji list and the broader Unicode emoji
    blocks are scrubbed so unmapped pictographs don't render as tofu.
    """
    if not text:
        return text
    for src in _EMOJI_TAGS:
        if src in text:
            # Consume an immediately-following space too, so "🗺️ Course Map"
            # becomes "Course Map" instead of " Course Map".
            text = text.replace(src + " ", "")
            text = text.replace(src, "")
    return _EMOJI_BLOCK_RE.sub("", text)


def _sanitize_for_core_font(text: str) -> str:
    """ASCII fallback used when no unicode TTF is registered."""
    replacements = {
        "‘": "'", "’": "'", "‚": ",", "‛": "'",
        "“": '"', "”": '"', "„": '"',
        "–": "-", "—": "--", "−": "-",
        "•": "*", "·": "*", "▪": "*", "●": "*",
        "…": "...",
        " ": " ", " ": " ", "​": "",
        "→": "->", "←": "<-", "⇒": "=>",
        "×": "x",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


# fpdf2 2.8.7's markdown-link parser turns every ``[label](#anchor)`` into a
# placeholder named destination it expects us to resolve via set_link(name=...)
# later — and crashes ``pdf.output()`` if we don't. We have no way to make
# LLM-generated anchors stable (the model may emit ``[See §1](#1)``,
# ``[1](#1)``, etc. at any time), and PDF link clickability is moot for files
# read offline from Downloads. Flatten the construct to plain ``label`` before
# fpdf2 sees it. Also covers invalid-URL and links-inside-bold edge cases.
_MD_LINK_RE: Final[re.Pattern[str]] = re.compile(r"\[([^\[\]]+)\]\(([^()]+)\)")


def _flatten_markdown_links(text: str) -> str:
    """Replace ``[label](url)`` with ``label`` everywhere in ``text``."""
    return _MD_LINK_RE.sub(r"\1", text or "")


def _prepare_text(
    text: str, *, unicode_ok: bool, flatten_links: bool = True
) -> str:
    """Apply emoji-tagging, link-flattening, and (if needed) latin-1 sanitization.

    Args:
        text: Source string from the markdown body.
        unicode_ok: Whether the registered fonts cover non-latin-1 glyphs.
        flatten_links: Strip ``[label](url)`` markdown links to ``label``.
            Default ``True``; pass ``False`` only inside fenced code blocks
            where a literal ``[foo](bar)`` should be preserved.
    """
    text = _emoji_to_tags(text or "")
    if flatten_links:
        text = _flatten_markdown_links(text)
    if not unicode_ok:
        text = _sanitize_for_core_font(text)
    return text


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

_FIRST_H1_RE: Final[re.Pattern[str]] = re.compile(r"^\s*#\s+(.+)$", re.MULTILINE)
_FILENAME_SAFE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, *, max_len: int = 40) -> str:
    """Lowercase, hyphenated, filesystem-safe slug from arbitrary text."""
    cleaned = _emoji_to_tags(text or "")
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
    slug = _FILENAME_SAFE_RE.sub("-", cleaned.lower()).strip("-")
    if not slug:
        return "smart-teacher"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug


def _extract_topic(body_markdown: str, fallback_title: str) -> str:
    """Pick the cleanest available topic string.

    Prefers the first ``# Heading`` in the markdown body (that's the
    actual subject the LLM chose); falls back to the caller-supplied
    title (the user prompt) when no H1 exists.
    """
    if body_markdown:
        m = _FIRST_H1_RE.search(body_markdown)
        if m:
            return m.group(1).strip()
    return fallback_title


def _build_filename(topic: str, *, suffix: str = "pdf") -> str:
    """Compose ``<topic-slug>_<YYYY-MM-DD>.<suffix>``."""
    slug = _slugify(topic)
    date = datetime.now().strftime("%Y-%m-%d")
    return f"{slug}_{date}.{suffix}"


# ---------------------------------------------------------------------------
# Markdown block primitives
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^\s*```")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_HRULE_RE = re.compile(r"^\s*(-{3,}|_{3,}|\*{3,})\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _split_table_row(line: str) -> list[str]:
    """Split a ``| a | b | c |`` row into cell strings (trimmed)."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _try_consume_table(
    lines: list[str], start: int
) -> tuple[list[list[str]], int] | None:
    """If a markdown table starts at ``start`` return ``(rows, end_idx)``.

    ``end_idx`` points one past the last consumed line. Returns ``None``
    when the block at ``start`` is not a valid table.
    """
    if start + 1 >= len(lines):
        return None
    if not _TABLE_ROW_RE.match(lines[start]):
        return None
    if not _TABLE_SEP_RE.match(lines[start + 1]):
        return None
    header = _split_table_row(lines[start])
    rows: list[list[str]] = [header]
    i = start + 2
    while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
        row = _split_table_row(lines[i])
        # Pad / truncate to header width so the table renderer stays happy.
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        rows.append(row)
        i += 1
    return rows, i


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _para(
    pdf: FPDF,
    height: float,
    text: str,
    *,
    markdown: bool = True,
    fill: bool = False,
    align: str = "L",
) -> None:
    """Write a paragraph and move the cursor to the next line at left margin."""
    pdf.multi_cell(
        0,
        height,
        text,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        markdown=markdown,
        fill=fill,
        align=align,
    )


def _new_pdf(title: str, fonts_holder: list[_FontSet]) -> FPDF:
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=15, top=15, right=15)
    fonts = _install_fonts(pdf)
    fonts_holder.append(fonts)
    safe_title = _prepare_text(title, unicode_ok=fonts.unicode_ok)
    pdf.set_title(safe_title[:120])
    pdf.set_author("Smart Teacher")
    pdf.add_page()
    return pdf


def _write_title(pdf: FPDF, title: str, fonts: _FontSet) -> None:
    safe_title = _prepare_text(title, unicode_ok=fonts.unicode_ok)
    pdf.set_font(fonts.body, style="B", size=18)
    _para(pdf, 9, safe_title, markdown=False)
    pdf.set_font(fonts.body, style="I", size=9)
    pdf.set_text_color(110, 110, 110)
    subtitle = _prepare_text(
        f"Generated by Smart Teacher · {datetime.now():%Y-%m-%d %H:%M}",
        unicode_ok=fonts.unicode_ok,
    )
    _para(pdf, 5, subtitle, markdown=False)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def _render_table(pdf: FPDF, rows: list[list[str]], fonts: _FontSet) -> None:
    """Render a markdown table as a real bordered fpdf2 table."""
    if not rows:
        return
    prepared = [
        [_prepare_text(cell, unicode_ok=fonts.unicode_ok) for cell in row]
        for row in rows
    ]
    pdf.ln(2)
    pdf.set_font(fonts.body, size=10)
    pdf.set_draw_color(180, 180, 180)
    with pdf.table(
        text_align="LEFT",
        line_height=6,
        markdown=True,
        first_row_as_headings=True,
        headings_style=__import__(
            "fpdf.fonts", fromlist=["FontFace"]
        ).FontFace(emphasis="BOLD", fill_color=(240, 240, 240)),
    ) as table:
        for r in prepared:
            row = table.row()
            for cell in r:
                row.cell(cell)
    pdf.ln(2)


def _render_body(
    pdf: FPDF,
    markdown_text: str,
    fonts: _FontSet,
    *,
    skip_title: str | None = None,
) -> None:
    """Walk markdown lines and render them with fpdf2 primitives.

    When ``skip_title`` is provided, the first H1 line in the body that
    matches it is dropped — this avoids printing the same heading twice
    when the PDF title was extracted from that H1.
    """
    raw_lines = (markdown_text or "").splitlines()
    in_code = False
    i = 0
    title_skipped = False
    skip_title_norm = (skip_title or "").strip()

    while i < len(raw_lines):
        raw_line = raw_lines[i]
        line = raw_line.rstrip()

        if _CODE_FENCE_RE.match(line):
            in_code = not in_code
            pdf.ln(1)
            i += 1
            continue

        if in_code:
            prepared = _prepare_text(
                line or " ",
                unicode_ok=fonts.unicode_ok,
                flatten_links=False,
            )
            pdf.set_font(fonts.mono, size=9)
            pdf.set_fill_color(245, 245, 245)
            _para(pdf, 5, prepared, markdown=False, fill=True)
            i += 1
            continue

        if not line.strip():
            pdf.ln(3)
            i += 1
            continue

        # Markdown table?
        table = _try_consume_table(raw_lines, i)
        if table is not None:
            rows, end_idx = table
            _render_table(pdf, rows, fonts)
            i = end_idx
            continue

        if _HRULE_RE.match(line):
            y = pdf.get_y() + 2
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(5)
            i += 1
            continue

        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            raw_text = m.group(2).strip()
            if (
                not title_skipped
                and level == 1
                and skip_title_norm
                and raw_text == skip_title_norm
            ):
                title_skipped = True
                i += 1
                continue
            text = _prepare_text(raw_text, unicode_ok=fonts.unicode_ok)
            size = max(11, 18 - 2 * (level - 1))
            pdf.ln(1)
            pdf.set_font(fonts.body, style="B", size=size)
            _para(pdf, size * 0.55, text, markdown=False)
            pdf.ln(1)
            i += 1
            continue

        m = _BLOCKQUOTE_RE.match(line)
        if m:
            text = _prepare_text(m.group(1).strip(), unicode_ok=fonts.unicode_ok)
            pdf.set_font(fonts.body, style="I", size=11)
            pdf.set_text_color(90, 90, 90)
            _para(pdf, 6, "  " + text)
            pdf.set_text_color(0, 0, 0)
            i += 1
            continue

        m = _BULLET_RE.match(line)
        if m:
            text = _prepare_text(m.group(1).strip(), unicode_ok=fonts.unicode_ok)
            pdf.set_font(fonts.body, size=11)
            _para(pdf, 6, "  - " + text)
            i += 1
            continue

        m = _NUMBERED_RE.match(line)
        if m:
            text = _prepare_text(line.strip(), unicode_ok=fonts.unicode_ok)
            pdf.set_font(fonts.body, size=11)
            _para(pdf, 6, "  " + text)
            i += 1
            continue

        text = _prepare_text(line, unicode_ok=fonts.unicode_ok)
        pdf.set_font(fonts.body, size=11)
        _para(pdf, 6, text)
        i += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def markdown_to_pdf_bytes(title: str, body_markdown: str) -> bytes:
    """Render a title + markdown body into PDF bytes.

    Args:
        title: Heading shown at the top of the document. Also used as the
            PDF metadata title.
        body_markdown: The content. Supports headings, bullets, numbered
            lists, blockquotes, fenced code blocks, markdown tables, and
            inline ``**bold**`` / ``*italic*``.

    Returns:
        The encoded PDF as ``bytes``.
    """
    holder: list[_FontSet] = []
    pdf = _new_pdf(title, holder)
    fonts = holder[0]
    _write_title(pdf, title, fonts)
    _render_body(pdf, body_markdown, fonts, skip_title=title)
    out = pdf.output()
    return bytes(out)


def pdf_filename(title: str, body_markdown: str) -> str:
    """Return the filename a saved PDF would use, without writing the file.

    Useful for ``st.download_button(file_name=...)`` so the browser download
    matches what ``save_pdf_to_downloads`` would have produced.

    Args:
        title: Fallback title when the body has no H1.
        body_markdown: PDF body content.

    Returns:
        ``<topic-slug>_<YYYY-MM-DD>.pdf``.
    """
    topic = _extract_topic(body_markdown, title)
    return _build_filename(topic)


def save_pdf_to_downloads(
    title: str,
    body_markdown: str,
    *,
    downloads_dir: Path | None = None,
) -> Path:
    """Write the PDF under ``~/Downloads/`` and return its absolute path.

    Filename pattern: ``<topic>_<YYYY-MM-DD>.pdf`` where ``<topic>`` is
    the slugified first H1 in the response (or the supplied title if the
    body has no H1).

    Args:
        title: Used as the PDF cover heading and as a fallback for the
            filename topic when the body lacks an H1.
        body_markdown: PDF body content.
        downloads_dir: Override target directory (used by tests).

    Returns:
        Absolute path to the saved file.

    Raises:
        OSError: If the target directory cannot be created or written to.
    """
    target_dir = downloads_dir or DOWNLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    topic = _extract_topic(body_markdown, title)
    path = target_dir / _build_filename(topic)
    path.write_bytes(markdown_to_pdf_bytes(topic, body_markdown))
    return path
