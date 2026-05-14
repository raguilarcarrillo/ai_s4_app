# PDF export — Smart Teacher

This document covers the **"Save as PDF"** feature: what it does, where it
lives, how it's wired into the app, and how to tweak it.

It is the only place in the project where rendered chat / quiz output is
serialized outside Streamlit, so it's worth understanding before changing.

---

## What the feature does

Every assistant reply in the **Chat** tab and every quiz / grade report in
the **Quiz** tab is accompanied by a `📄 Save as PDF` button. When the
user clicks it:

1. The markdown content for that specific message is rendered into a PDF
   in-memory (no temp file).
2. The PDF is written to `~/Downloads/` with a short filename of the
   form `<topic-slug>_<YYYY-MM-DD>.pdf`.
3. A green Streamlit `success` banner shows the absolute path.

No network call, no third-party service.

---

## Where the code lives

| File | Role |
|---|---|
| `app/pdf_export.py` | Renderer. Public entry points and all conversion logic. |
| `app/app.py` | Wires the button into the chat history, the new chat reply, the active quiz, and the graded report. |
| `requirements.txt` | Pins `fpdf2>=2.8.7,<3.0.0` (vulnerability-audited 2026-05-14). |

The renderer has **no dependency on Streamlit, LangChain, or the rest of
the app** — `pdf_export.py` only imports stdlib + `fpdf2`. That makes it
trivial to unit-test or reuse in a CLI tool later.

---

## Public API (`app/pdf_export.py`)

```python
def markdown_to_pdf_bytes(title: str, body_markdown: str) -> bytes: ...
def save_pdf_to_downloads(
    title: str,
    body_markdown: str,
    *,
    downloads_dir: Path | None = None,
) -> Path: ...
```

- `markdown_to_pdf_bytes` returns raw PDF bytes — usable from
  `st.download_button` if you ever want a browser-streamed alternative.
- `save_pdf_to_downloads` writes the PDF under `~/Downloads/` (overridable
  for tests via `downloads_dir=`) and returns the absolute path.

Both accept the same content. `title` is used as the document heading and
as the **fallback** for the filename topic. If the body markdown starts
with an `# H1` heading, the topic is extracted from that H1 instead — so
the filename matches the subject the LLM actually wrote about, not the
raw user prompt.

---

## Rendering pipeline

```
markdown string
   │
   ▼
emoji → text tags ([MAP], [OK], [TIP] …)
   │
   ▼
unicode_ok?  ── yes → keep glyphs (Noto Sans + DejaVu Sans Mono)
   │
   └─ no  → strip non-latin-1 chars (Helvetica fallback)
   │
   ▼
line-by-line block parser
   ├── fenced code block ──► mono font + gray fill
   ├── markdown table     ──► fpdf2 `pdf.table()` with bold header
   ├── # heading          ──► bold, larger size (decays with depth)
   ├── > blockquote       ──► gray italic
   ├── - / * bullets      ──► indented bullets
   ├── 1. numbered        ──► indented numbered
   ├── --- horizontal rule──► thin gray line
   └── default            ──► plain paragraph, **bold** / *italic* respected
   │
   ▼
A4 page, 15 mm margins, left-aligned, auto page-break
```

The first H1 in the body is **suppressed** when its text matches the
document title — this prevents the same heading appearing twice (once as
the cover heading, once again as the body's first H1).

---

## Fonts

The renderer probes these directories at runtime, in order:

1. `/usr/share/fonts/truetype/noto`
2. `/usr/share/fonts/truetype/dejavu`
3. `/usr/share/fonts/TTF`
4. `/usr/local/share/fonts`

Body candidates (first match wins, regular + bold are required):

| Family | Files |
|---|---|
| **Noto Sans** *(preferred)* | `NotoSans-Regular.ttf`, `NotoSans-Bold.ttf`, `NotoSans-Italic.ttf`, `NotoSans-BoldItalic.ttf` |
| DejaVu Sans | `DejaVuSans.ttf`, `DejaVuSans-Bold.ttf` (+ oblique variants if present) |

Monospace candidate:

| Family | Files |
|---|---|
| **DejaVu Sans Mono** | `DejaVuSansMono.ttf`, `-Bold.ttf`, `-Oblique.ttf`, `-BoldOblique.ttf` |

If **either** family is unresolvable on the host (e.g. a Mac dev box
without these fonts), the renderer logs at `INFO` and falls back to the
fpdf2 built-in **Helvetica / Courier** latin-1 fonts plus a sanitization
pass that maps common unicode chars (smart quotes, em-dash, bullets,
arrows) to ASCII equivalents. Output still works; just less pretty.

To bundle fonts inside the repo for cross-platform parity, drop the TTF
files into a new `app/fonts/` directory and prepend that path to
`_FONT_SEARCH_DIRS` in `pdf_export.py`.

---

## Emoji handling

Color emojis can't render in fpdf2 (no CBDT/CBLC color-bitmap support),
so the renderer **strips them** before they reach fpdf2 (along with any
single space that immediately followed the emoji). The bold heading text
is enough to convey meaning without a glyph in front of it.

Both an explicit known-emoji list (`_EMOJI_TAGS` in `pdf_export.py`) and a
broader Unicode-emoji-block regex are scrubbed, so unmapped pictographs
don't render as tofu boxes either.

If you'd rather keep visual hierarchy with a text tag (e.g. `[MAP]`,
`[OK]`), swap the body of `_emoji_to_tags` to map each known emoji to a
tag instead of `""`. The `_EMOJI_TAGS` dict already holds a suggested
mapping for that variant.

---

## Filename scheme

```
<topic-slug>_<YYYY-MM-DD>.pdf
```

- **`<topic-slug>`** — derived from the first `# H1` in the body when one
  exists, otherwise from the caller-supplied title. The string is
  emoji-stripped, ASCII-folded, lower-cased, and hyphenated. Capped at
  40 characters.
- **`<YYYY-MM-DD>`** — local date at save time.

Example: a prompt like *"Teach me linear algebra, make sure of the
following…"* whose answer opens with `# Complete Linear Algebra Course —
From Zero to Confident` produces:

```
complete-linear-algebra-course-from-zero_2026-05-14.pdf
```

Filenames are deterministic for the same `(topic, day)`. Re-saving the
same message on the same day overwrites the previous file — by design,
so users aren't left with five near-identical PDFs.

---

## How it's wired into `app.py`

Three call sites, all using one small helper:

```python
def _render_pdf_save_button(title: str, content: str, *, key: str, label: str = "📄 Save as PDF") -> None:
    if not content or not content.strip():
        return
    if st.button(label, key=f"pdfbtn_{key}"):
        try:
            path = save_pdf_to_downloads(title=title, body_markdown=content)
            st.success(f"Saved to `{path}`")
        except Exception as e:
            st.warning(f"Could not save PDF: {e}")
            logger.exception("PDF save failed")
```

1. **Chat history loop** — each historical assistant message gets a button
   with key `chat_hist_<idx>`. The title is the user's prompt that
   preceded the message.
2. **Freshly rendered chat reply** — key `chat_new_<len(history)>`.
3. **Active quiz** — converted to markdown via `_quiz_to_markdown(quiz,
   include_answers=False)` and offered as a PDF (without the answer key,
   so it's printable for self-study).
4. **Grade report** — converted via `_grade_report_to_markdown(quiz,
   report)` and offered as a PDF (with answers, explanations, and source
   refs).

All four sites pass unique `key`s so Streamlit's widget identity stays
stable across reruns.

---

## Security audit (2026-05-14)

- **fpdf2 2.8.7** — no advisories on GitHub Advisory DB, Snyk DB, or NVD.
- **CVE-2025-65875** (FPDF AddFont RCE) targets the **PHP** library
  `Setasign/FPDF`, *not* the Python `fpdf2` package. Safe to ignore in
  this project.
- Transitive deps (`Pillow`, `defusedxml`, `fontTools`) — all currently
  clean. Keep `Pillow` patched routinely; it's the only one with
  historically frequent image-parsing CVEs.

When upgrading `fpdf2`, re-check the GitHub Advisory Database and Snyk
before bumping the pin in `requirements.txt`.

---

## Customization knobs

Easy edits in `pdf_export.py`:

| What | Where |
|---|---|
| Add / change an emoji tag | `_EMOJI_TAGS` dict |
| Add a font search directory | `_FONT_SEARCH_DIRS` tuple |
| Add a different body font family | `_BODY_FONT_CANDIDATES` (regular + bold required) |
| Change filename slug max length | `_slugify(max_len=...)` default |
| Change date format in filenames | `_build_filename()` |
| Change page size / margins | `_new_pdf()` (`format="A4"`, `set_margins(...)`) |
| Change default body font size | search for `size=11` in `_render_body()` |
| Tweak heading sizes | `size = max(11, 18 - 2 * (level - 1))` in `_render_body()` |
| Disable the title block at the top | comment out `_write_title(...)` in `markdown_to_pdf_bytes()` |
| Change Downloads target | `DOWNLOADS_DIR` constant, or pass `downloads_dir=` |

---

## Known limitations

- **No color emojis**. fpdf2 doesn't read the CBDT/CBLC color-bitmap
  tables. Tags are used instead.
- **No image rendering**. The current pipeline is pure text. If a
  markdown body contains `![alt](url)` it's rendered as the literal
  markdown.
- **No nested lists**. Indented sub-bullets render as plain bullets at
  the same level.
- **No syntax highlighting** in code blocks — just monospace + light fill.
- **HuggingFace Spaces compatibility**: the system fonts probed here are
  the Linux defaults, which Spaces typically has. If the Space image is
  ever swapped to a slim base without `fonts-noto` / `fonts-dejavu`, the
  renderer falls back to Helvetica + sanitization automatically — no
  crash, just less polished output.

---

## Testing

There are no committed unit tests yet. To smoke-test manually:

```python
from pdf_export import markdown_to_pdf_bytes, save_pdf_to_downloads
import pathlib

sample = pathlib.Path("docs/sample.md").read_text()  # bring your own
b = markdown_to_pdf_bytes("Sample title", sample)
assert b[:4] == b"%PDF"

path = save_pdf_to_downloads("Sample title", sample)
print("wrote", path)
```

Edge cases worth exercising when changing the renderer:

- Empty body → should still produce a valid header-only PDF.
- Very long line with no spaces → should wrap, not throw.
- Mixed unicode (Greek, math symbols, CJK if applicable) → should
  render when Noto Sans is found.
- Markdown table with mismatched column counts → renderer pads / truncates
  rows to the header width.

---

## Change log

- **2026-05-14** — initial feature: `📄 Save as PDF` button on chat, quiz,
  and grade-report outputs. Output under `~/Downloads/`.
- **2026-05-14** — improved rendering: switched body font to Noto Sans
  and mono to DejaVu Sans Mono, added markdown-table rendering, emoji →
  text-tag mapping, left-aligned paragraphs, and shortened filename to
  `<topic>_<YYYY-MM-DD>.pdf`. Suppressed duplicate H1 when title is
  derived from it.
- **2026-05-14** — changed emoji policy from text tags (`[MAP]`, `[OK]`
  …) to clean stripping. Headings now appear without leading glyphs or
  brackets.
