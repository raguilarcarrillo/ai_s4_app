# Smart Teacher — Architecture & Implementation Reference

A complete, self-contained reference for the `ai_s4_app` codebase: what it
is, how it's wired, why each part was built that way, and which packages
do what work. Cite this file when you need ground truth that doesn't live
in the README.

---

## 1. What it is

Smart Teacher is a Streamlit + LangChain pedagogical tutor. A user
uploads documents (PDF, TXT, MD, IPYNB) and either:

- **Chats** with a Smart Teacher persona that returns a 5-section
  pedagogical answer (Explanation → Recommended Learning Method → Study
  Plan → Practice Exercises → Self-check Questions), grounded in the
  uploaded chunks via inline `[source#index]` citations, with the
  option to emit a single inline Plotly chart.
- **Generates a quiz** (mixed types, calibrated difficulty), takes it,
  and is graded locally with per-question feedback and source refs.

Every assistant reply and graded report can be exported to PDF.

The app is LLM-agnostic and embeddings-agnostic — pluggable factories
abstract both. The target host is HuggingFace Spaces (free Streamlit
SDK tier), so all heavy primitives are chosen for an ephemeral,
shared, single-process container.

Entry point: `streamlit run app/app.py` → `app.main()`.

---

## 2. High-level architecture

```
                      ┌──────────────────────────────────────────────┐
                      │  Streamlit UI  (app/app.py)                  │
                      │  ┌─────────┐  ┌──────────┐  ┌────────────┐   │
                      │  │ Sidebar │  │ Chat tab │  │  Quiz tab  │   │
                      │  └─────────┘  └────┬─────┘  └─────┬──────┘   │
                      └─────────┬──────────┴──────────────┴──────────┘
                                │
            ┌───────────────────┼──────────────────────────┐
            ▼                   ▼                          ▼
   ┌──────────────────┐ ┌───────────────┐         ┌──────────────────┐
   │ llm_factory.py   │ │ rag.py        │         │ persistence.py   │
   │  (6 providers,   │ │  load → split │         │  opt-in disk     │
   │  lazy imports)   │ │  → embed →    │ ◄────►  │  caches:         │
   └────────┬─────────┘ │  FAISS →      │         │  faiss/<fp>/     │
            │           │  retrieve     │         │  sessions/<h(t)> │
            │           └──────┬────────┘         └──────────────────┘
            │                  ▲
            │                  │
            │           ┌──────┴─────────────┐
            │           │ embeddings_        │
            │           │ factory.py         │
            │           │  (3 backends)      │
            │           └────────────────────┘
            │
            ▼
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │ prompts.py       │ ─► │ quiz.py          │ ─► │ pdf_export.py    │
   │ teacher + quiz   │    │ Pydantic Quiz +  │    │ fpdf2 renderer   │
   │ system prompts   │    │ grade_quiz       │    │ + chart embed    │
   └──────────────────┘    └──────────────────┘    └──────────────────┘
                                                            ▲
                                                            │
                                              ┌─────────────┴────────┐
                                              │ charts.py            │
                                              │ AST-sandboxed eval   │
                                              │ Plotly + kaleido PNG │
                                              └──────────────────────┘
```

**Architectural principles:**

1. **Provider-pluggable, factory-mediated.** `llm_factory.get_llm` and
   `embeddings_factory.get_embeddings` are the only two coupling points
   to any vendor SDK. Everything downstream sees a LangChain
   `BaseChatModel` / `Embeddings` instance.
2. **Lazy imports inside factory branches.** Each provider package is
   imported only when its branch runs, so cold start is cheap and an
   uninstalled optional provider never crashes startup.
3. **Single Streamlit process per browser session.** No worker queue,
   no Celery, no API service. State that must survive a rerun lives in
   `st.session_state`; state that must survive a cold start lives in
   the opt-in on-disk layer.
4. **In-memory FAISS by default.** Opt-in persistence keys FAISS
   indexes by content fingerprint (sha256 of file bytes + chunk params)
   so byte-identical uploads across sessions reuse the same index.
5. **Defense in depth.** API-key redaction, sha256-hashed token
   directories, AST-validated chart eval, notebook-output stripping,
   markdown-link flattening — every dangerous primitive has a primary
   guard plus a backstop.

---

## 3. Tech stack and dependency rationale

The full set lives in `requirements.txt` (lower-bound pins; the file
notes you should re-pin with `pip freeze` for reproducible builds).

| Package | Pin | Role |
|---|---|---|
| `streamlit` | `>=1.40` | UI shell. `st.cache_resource` (process-wide), `st.session_state` (per session), `st.chat_message/chat_input`, `st.tabs`, `st.file_uploader`, `st.download_button`, `st.plotly_chart`. |
| `pydantic` | `>=2.9` | `Quiz`/`Question`/`GradeReport` schemas. Doubles as the JSON-schema source LangChain hands to `with_structured_output`. v2 specifically for `field_validator` and `model_validate`. |
| `python-dotenv` | `>=1.0` | Loads `.env` at `app.py` startup so provider keys land in `os.environ` before any factory reads them. Wrapped in `try/except ImportError` → genuinely optional. |
| `langchain` + `-core` + `-community` + `-text-splitters` | `>=0.3.x` | `Document`, `RecursiveCharacterTextSplitter`, `PyPDFLoader`, `TextLoader`, `FAISS`, `ChatPromptTemplate`. The 0.3.x split lets us pull loaders/vector-store from `-community` without dragging in every other integration. |
| `langchain-anthropic` / `-openai` / `-google-genai` / `-groq` / `-huggingface` / `-ollama` | various | One adapter per LLM provider. All imported lazily inside their `get_llm` branch. |
| `sentence-transformers` | `>=3.3` | Local CPU embeddings via `HuggingFaceEmbeddings`. Default model `all-MiniLM-L6-v2` (~80 MB, 384-dim, normalized) — chosen because it needs no API key and fits the HF Spaces free tier. |
| `faiss-cpu` | `>=1.9` | In-memory vector store via `langchain_community.vectorstores.FAISS`. CPU build matches the embeddings; no external service to run. Two-file on-disk format (`index.faiss`+`index.pkl`) fits the content-keyed cache layout. |
| `pypdf` | `>=5.1` | Pure-Python PDF text extraction via `PyPDFLoader`. No Poppler / OCR — the app only needs text. |
| `fpdf2` | `>=2.8.7,<3.0.0` | PDF generation. Pure-Python, no system deps. Audit note in requirements.txt: GHSA-rwmj-c32v-585v targets *PHP* FPDF, not fpdf2. |
| `huggingface-hub` / `transformers` / `torchvision` | various | Transitive runtime for `sentence-transformers`. `torchvision` is *not used by app code* — it exists because recent `transformers` eagerly imports zoedepth image processors. |
| `plotly` | `>=5.20,<7.0` | Inline charts. Audit note: no CVEs; no sympy needed because chart math is evaluated under a custom AST whitelist. |
| `kaleido` | `>=1.2.0,<2.0` | Rasterizes Plotly figures to PNG for PDF embedding. v1.x uses *system* Chrome (rather than a bundled binary) — patches apply normally, and a missing Chrome triggers a graceful placeholder. |

`numpy` is unpinned but imported directly in `charts.py` — it ships transitively via `sentence-transformers`/`faiss-cpu`.

---

## 4. End-to-end data flows

Each flow is a numbered trace. Shapes in `[brackets]`.

### 4.1 Chat / RAG flow

1. **Upload.** `app._render_sidebar` → `st.file_uploader` → `[List[UploadedFile]]`.
2. **Trigger.** `_render_chat_tab` calls `_get_vector_store_safe(cfg)`.
   If persistence is on or files were restored, `persistence.load_faiss(fp, embeddings)` is consulted first — cache hit short-circuits. Otherwise the `@st.cache_resource`-wrapped `_cached_vector_store` runs.
3. **Fingerprint.** `rag.fingerprint(files, chunk_size, chunk_overlap)` → sha256 over `cs=<>;co=<>;<basename><bytes>...` → `[str hex]`.
4. **Load.** `rag.load_uploaded_files` writes each `UploadedFile` into a `tempfile.TemporaryDirectory` (path-traversal guard: refuses any resolved path whose parents don't contain the tempdir), then `_read_file_to_documents` dispatches by suffix to `PyPDFLoader` / `TextLoader` / `_load_notebook`. Restored sessions take the `load_disk_files` branch instead. → `[List[Document]]`.
5. **Split.** `rag.split_documents` → `RecursiveCharacterTextSplitter(separators=["\n\n","\n",". "," ",""])` → each chunk gets `metadata.chunk_index` and `metadata.chunk_id = "<source>#<idx>"`. → `[List[Document] with metadata]`.
6. **Embed.** `_cached_embeddings(backend, model)` → `embeddings_factory.get_embeddings` → LangChain `Embeddings`. Default = `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", normalize_embeddings=True)`.
7. **Index.** `rag.build_vector_store(chunks, embeddings)` lazy-imports `FAISS` and returns `FAISS.from_documents(chunks, embeddings)` → `[FAISS]`.
8. **Question.** `st.chat_input` → `[str]`. Appended to `st.session_state["chat_history"]`.
9. **Retrieve.** `rag.retrieve(vs, query, k)` calls `similarity_search_with_score`, converts FAISS L2 distance to `1/(1+distance)` so higher = better. → `[List[RetrievedChunk(chunk_id, source, content, score)]]`.
10. **Context block.** `rag.format_context_block` joins chunks as `[chunk_id]\n<content>` separated by `---`. → `[str]`.
11. **Prompt.** `prompts.build_teacher_messages(question, context_block, history[-6:])` → `[List[Tuple["system"|"user"|"assistant", str]]]`. Only the last 3 exchanges are carried forward to bound token cost.
12. **Invoke.** `_get_llm_for_session(provider, model, temperature, api_key)` returns the cached LLM client (LRU of 4, keyed by `provider::model::temperature::sha256-hint(key)`). `llm.invoke(messages)` → `[str markdown]`.
13. **Render.** `_render_assistant_content` runs `charts.split_text_and_charts` to interleave `st.markdown` and `st.plotly_chart`. `_render_sources(chunks)` shows an expandable `RetrievedChunk` list with truncated previews. `_render_pdf_save_button` attaches the PDF download.

### 4.2 Quiz flow

1. **Inputs.** `_render_quiz_tab` collects `num_q`, `difficulty`, `qtype` (incl. `mixed`), `topic`, `grounded`. On click → `_generate_and_store_quiz`.
2. **Optional grounding.** If `grounded`, reuses `_get_vector_store_safe` and calls `rag.retrieve(vs, topic, k=max(cfg["k"], 8))` to pull a wider net for coverage. → `[List[RetrievedChunk]]`.
3. **Generate.** `quiz.generate_quiz`:
   - Builds the user message via `prompts.build_quiz_user_message` (toggles GROUNDED/UNGROUNDED mode marker based on `bool(context_block.strip())`).
   - **First attempt:** `llm.with_structured_output(Quiz).invoke(messages)` — native schema enforcement on Anthropic / OpenAI / Google / Groq.
   - **Fallback:** plain `llm.invoke` + `_parse_quiz_payload` (handles `Quiz` instance, `dict`, JSON string with fences stripped by `_strip_code_fences`, or LangChain message via `.content`) with up to `max_retries + 1 = 3` attempts; each failure appends a corrective user turn.
   - Returns a Pydantic-validated `Quiz(topic, difficulty, questions: List[Question])`.
4. **Stash.** `st.session_state["quiz" | "quiz_answers" | "quiz_report" | "quiz_sources"]`.
5. **Form.** `_render_active_quiz` walks `quiz.questions` inside `st.form("quiz_form")`. MCQ/TF use `st.radio` with integer options; short-answer uses `st.text_input`. Writes to `answers: dict[int, int|str]`.
6. **Grade.** `quiz.grade_quiz(quiz, answers)` calls `_grade_one` per question. MCQ/TF compare integers (type errors → incorrect, never crashes). Short-answer uses `_normalize_short_answer` (lowercase, collapse whitespace, strip edge punctuation, drop leading articles) and accepts either exact match or `\b<expected>\b` regex inside the user's normalized text. → `[GradeReport(score, total, percent, per_question: List[QuestionResult])]`.
7. **Render.** `_render_grade_report` picks `st.success/info/warning` by 80/50 thresholds; wraps each question in an `st.expander(expanded=not r.correct)`.
8. **Retry incorrect only.** `wrong_ids = [r.id for r in report.per_question if not r.correct]` stored in `st.session_state["retry_ids"]`; the form loop filters by membership.

### 4.3 PDF export flow

1. **Source markdown.** One of: chat reply, `_quiz_to_markdown(quiz, include_answers=False)`, or `_grade_report_to_markdown(quiz, report)`.
2. **Memoize.** `_get_pdf_bytes(title, content)` — `OrderedDict` in `st.session_state["_pdf_cache"]`, keyed by `f"{title}::{sha256(content)[:16]}"`, LRU bound 32.
3. **Render.** `pdf_export.markdown_to_pdf_bytes(title, body_markdown)`:
   - `_new_pdf` → A4, 15mm margins, `_install_fonts` probes `/usr/share/fonts/truetype/{noto,dejavu}` and registers NotoSans + DejaVuSansMono via `pdf.add_font`. Missing → Helvetica/Courier + `_sanitize_for_core_font` latin-1 fallback.
   - `_prepare_text` → `_emoji_to_tags` strips emojis (explicit dict + `_EMOJI_BLOCK_RE` regex backstop) → `_flatten_markdown_links` rewrites `[label](url)` to bare `label` (sidesteps fpdf2 2.8.7 unresolved-named-destination crash).
   - `_render_body` is a line-by-line state machine dispatching to: chart fence (`_embed_chart`), code fence (mono font, light fill, links *not* flattened), markdown table (`_try_consume_table` + `pdf.table()`), heading, blockquote, bullet, numbered list, horizontal rule, or plain paragraph. Unknown markdown falls through to plain paragraph — never raises.
   - `pdf.output()` → `[bytes]`.
4. **Filename.** `pdf_filename(title, body)` slugs the first body H1 (or the fallback title) via `_slugify` + `_build_filename` → `<topic-slug>_<YYYY-MM-DD>.pdf`.
5. **Deliver.** `st.download_button(data=pdf_bytes, file_name=..., mime="application/pdf")` — browser saves, no server filesystem write. Works on HF Spaces unchanged. `save_pdf_to_downloads` still exists but is only used by local CLI/test paths.

### 4.4 Opt-in persistence flow

1. **Toggle.** `_render_persistence_section` exposes `persist_enabled`. While unchecked, `_save_to_persistence` is never called and nothing leaves RAM.
2. **Mint.** `persistence.new_token()` → `uuid.uuid4().hex` (32 hex chars) → stored in `st.session_state["persist_token"]` and shown verbatim.
3. **Build / load index.** `_get_vector_store_safe` computes `fp = rag.fingerprint(...)`. Disk cache (`persistence.load_faiss(fp, embeddings)`) is consulted first; on miss the `@st.cache_resource` path runs.
4. **Gate.** A per-session marker `f"persist_marker::{token}::{fp}"` in `st.session_state` ensures `_save_to_persistence` runs once per `(token, fingerprint)` — Streamlit reruns this function on every interaction.
5. **Save FAISS.** `persistence.save_faiss(fp, vs)` → `vs.save_local(.cache/faiss/<fp>/)`. **Content-keyed**, shareable across sessions with identical bytes.
6. **Save session.** `persistence.save_session(token, [(safe_name, bytes), ...], **manifest_fields)`:
   - Directory `.cache/sessions/<sha256(token)>/`. **Raw token never lands on disk.**
   - `files/<safe_name>` + `manifest.json` (`SessionManifest` dataclass: filenames, fingerprint, splitter params, embed backend/model, `created_at`, `last_accessed_at` UTC ISO 8601).
7. **Restore.** User pastes token → `load_session` validates `^[0-9a-f]{32}$`, reads manifest, returns `(SessionManifest, list[Path])`, **rewrites `last_accessed_at`** so TTL is "days since last use".
8. **Restore pipeline.** Next rerun, `_get_vector_store_safe` sees `restored` paths with `is_restored=True`. Tries `persistence.load_faiss` first (sets `allow_dangerous_deserialization=True` — safe because we only load files we wrote). On miss, `_cached_vector_store` falls through to `rag.load_disk_files(paths)`. Manifest is **not** re-saved when `is_restored=True`.
9. **TTL sweep.** `_run_startup_cleanup` runs once per process via `@st.cache_resource(show_spinner=False)`. Calls `persistence.cleanup_expired()`, which removes sessions older than `SMART_TEACHER_SESSION_TTL_DAYS` (default 7) **and** any malformed dir missing `manifest.json` so the cache self-heals.

### 4.5 Inline chart flow

1. **Emit.** Per `TEACHER_SYSTEM_PROMPT`, the LLM may append a single fenced ` ```chart\n<JSON>\n``` ` block.
2. **Split.** `_render_assistant_content` iterates `charts.split_text_and_charts(content)` using `CHART_FENCE_RE = r"```chart\s*\n(.*?)\n```"` (DOTALL) → yields `("text", chunk)` and `("chart", spec_text)` in source order.
3. **Parse.** `charts.render(spec_text)` does `json.loads` then `_build_figure`. Never raises — failures come back as `(None, error_str)`.
4. **Build figure.** `_build_figure`:
   - Validates `x.range` via `_validate_range` (`[min, max]`, `min < max`, `max(|min|,|max|) <= _MAX_X_MAGNITUDE = 1e6`).
   - `x_values = np.linspace(x_min, x_max, _MAX_POINTS = 500)`.
   - Each `series`: `function` → `_safe_eval` → `go.Scatter(mode="lines")`; `scatter` → caps at `_MAX_SCATTER = 1000`, `mode = "lines+markers" if connect else "markers"`.
   - `vlines` / `hlines` / `points` → corresponding Plotly primitives.
   - Layout: `plotly_dark` template, `paper_bgcolor="#0E1117"` (matches Streamlit dark canvas).
5. **AST-sandboxed eval.** `_safe_eval(expr, x_values)`:
   - `ast.parse(mode="eval")` → `_validate_ast` walks every node and rejects any type not in `_ALLOWED_NODES = {Expression, BinOp, UnaryOp, Constant, Name, Call, Load, Add, Sub, Mult, Div, Pow, Mod, FloorDiv, USub, UAdd}`. Names must be in `{x, pi, e}` ∪ `{sin, cos, tan, exp, log, sqrt, abs}`. Dunder names rejected explicitly. Calls must be direct (`ast.Name` callee), no keywords, no `Starred`.
   - `compile(tree, "<chart-expr>", "eval")` then `eval(code, _EVAL_GLOBALS, {"x": x_values})` with `_EVAL_GLOBALS = {"__builtins__": None, pi, e, sin, cos, tan, exp, log, sqrt, abs}` — belt-and-suspenders.
   - `y[~np.isfinite(y)] = np.nan` so discontinuities (`log(-1)`, divide-by-zero) plot cleanly with gaps.
6. **Chat render.** `st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_prefix}_{i}")` for interactive figures.
7. **PDF render.** `pdf_export._embed_chart` calls the same `charts.render(spec_text)`, then `charts.figure_to_png(fig)` → `fig.to_image(format="png", ...)` via kaleido. **If kaleido or Chrome is missing**, returns `None` and `_placeholder_chart` writes an italic gray `[Chart: <title>]` line — graceful degradation.

---

## 5. Module reference

### 5.1 `app/app.py` — Streamlit UI shell

**Purpose.** The only module that knows Streamlit exists. Composes
factories, the RAG pipeline, the chart renderer, the PDF exporter, and
the quiz module into a sidebar + Chat tab + Quiz tab.

**Public entry points.** `main()`.

**Notable internals.**

- `_cached_embeddings(backend, model)` — `@st.cache_resource`; process-wide because no user secret participates (local model or deploy-time env key). Avoids reloading the ~80 MB MiniLM per session.
- `_cached_vector_store(index_fingerprint, chunk_size, chunk_overlap, backend, model, _files, _embeddings)` — `@st.cache_resource` keyed by the fingerprint. Branches on `isinstance(_files[0], Path)` between `load_disk_files` (restored) and `load_uploaded_files` (fresh). Leading-underscore params are unhashable; the fingerprint already covers content.
- `_run_startup_cleanup()` — one-shot TTL sweep, memoized so it fires exactly once per process; wrapped in try/except so a misconfigured cache dir never blocks startup.
- `_get_llm_for_session(provider, model, temperature, api_key)` — session-state LRU (size 4) keyed by `provider::model::temperature::_key_hint(api_key)`. **Per-session** specifically to prevent the cross-tenant leak you'd get from `@st.cache_resource`.
- `_key_hint(value)` — 16-char sha256 prefix. Collision-resistant and unrecoverable.
- `_redact_secrets(text)` — sweeps every `st.session_state[f"key::{provider}"]` literal occurrence in `text` and replaces with `***REDACTED***`. Applied to every user-facing error string before render.
- `_friendly_llm_error(exc)` — translates raw exceptions into actionable Markdown: `UnicodeEncodeError` → "you pasted prompt text into the API key field"; `429` / `rate limit` → retry guidance; `401` / `auth` → recheck key; `connection`/`timeout`/`network` → try Ollama. Final fallback shows only exception class + first line; full traceback only goes to server logs.
- `_get_pdf_bytes(title, content)` — same LRU pattern as the LLM cache, bound 32.
- `_render_assistant_content(content, key_prefix)` — splits via `charts.split_text_and_charts`; markdown chunks → `st.markdown`; chart chunks → `st.plotly_chart` with stable per-message Streamlit key, fallback to `st.code` + caption on parse failure.
- `_render_persistence_section()` — strict opt-in; mints a fresh token on enable; restore path calls `persistence.load_session` and forces `st.rerun()`.
- `_get_vector_store_safe(cfg)` — source precedence fresh uploads > restored paths > None. Disk writes gated on `persist_marker::<token>::<fp>` session flag.

### 5.2 `app/llm_factory.py` — Pluggable LLM providers

**Purpose.** One `get_llm()` returns a LangChain `BaseChatModel` for any of six providers.

**Surface.**

- `ProviderSpec` (frozen dataclass): `key, label, default_model, env_var, needs_key, notes`.
- `PROVIDERS` — anthropic (`claude-sonnet-4-6`), openai (`gpt-4o-mini`), google (`gemini-2.5-flash`), groq (`llama-3.3-70b-versatile`, free dev tier), huggingface (`Meta-Llama-3-8B-Instruct`), ollama (`llama3.2`, no key).
- `resolve_api_key(provider, user_key)` — precedence sidebar > env > `st.secrets`; the secrets branch is wrapped in bare try/except so it works outside Streamlit.
- `get_llm(provider, model=None, temperature=0.2, api_key=None, **kwargs)` — lazy-imports the matching `langchain_*` package inside each branch. HuggingFace clamps `temperature` to `max(temp, 0.01)` (endpoint rejects exact 0). Ollama reads `OLLAMA_BASE_URL` env (default `http://localhost:11434`).

### 5.3 `app/embeddings_factory.py` — Pluggable embeddings

**Purpose.** Mirrors `llm_factory` for the embedding side.

**Surface.**

- `EmbeddingSpec`, `EMBEDDINGS` — `sentence-transformers` (default, no key, `all-MiniLM-L6-v2`), `huggingface` (HF Inference API), `openai` (`text-embedding-3-small`).
- `get_embeddings(backend="sentence-transformers", model=None, api_key=None, **kwargs)`.
- The default backend uses `HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True, ...})` — normalization on by default so cosine-equivalent similarity is well-behaved.

### 5.4 `app/rag.py` — Load → split → index → retrieve

**Purpose.** Owns IO and indexing; supports PDF, TXT, MD/MARKDOWN, IPYNB.

**Surface.**

- `RetrievedChunk` dataclass — `chunk_id` (`source#index`), `source`, `content`, `score`.
- `load_uploaded_files(uploaded)` — writes each Streamlit `UploadedFile` into a `tempfile.TemporaryDirectory`. Defense-in-depth: strips path components via `Path(raw_name).name`, refuses any resolved path whose parents don't contain the tempdir, checks extension is in `SUPPORTED_SUFFIXES = {.pdf, .txt, .md, .markdown, .ipynb}`. Per-file errors logged and skipped — one bad file doesn't abort the upload.
- `load_disk_files(paths)` — same dispatch for restored sessions (files already under `.cache/sessions/<id>/files/`).
- `_load_notebook(path, source_label)` — markdown + code cells only, **outputs and "raw" cells are skipped entirely**. The code-fence language is taken from `metadata.kernelspec.language` only if `str.isidentifier()` (prevents markup injection); NUL bytes stripped.
- `split_documents(docs, chunk_size=1000, chunk_overlap=150)` — `RecursiveCharacterTextSplitter(separators=["\n\n","\n",". "," ",""])`. Stamps `chunk_index` and `chunk_id = "<source>#<idx>"`.
- `build_vector_store(chunks, embeddings)` — lazy-imports FAISS, returns `FAISS.from_documents(...)`.
- `retrieve(vs, query, k=4)` — `similarity_search_with_score` with fallback to `similarity_search` on any failure. Converts L2 distance to `1/(1+distance)` so higher = better.
- `format_context_block(chunks)` — joins each chunk as `[chunk_id]\n<content>` with `---` separators.
- `fingerprint(files, chunk_size, chunk_overlap)` — sha256 over `cs=<>;co=<>;<basename><bytes>...`. Accepts either `UploadedFile` or `Path` and produces the same digest for byte-identical inputs — a restored session collide-hits the same cache.

### 5.5 `app/prompts.py` — System prompts

**Purpose.** Single home for the teacher persona, the quiz JSON contract, and message-builder helpers.

**Surface.**

- `TEACHER_SYSTEM_PROMPT` — five mandatory sections (Explanation, Recommended Learning Method, Study Plan, Practice Exercises, Self-check Questions) + optional Sources. Strict rules: bare `[source#index]` citations (no markdown links — that crashes the PDF renderer); math symbols must be wrapped in inline backticks (sans-serif fonts drop them in bold/italic); ungrounded answers begin with `> Note: this answer is **ungrounded**...`; "I don't know based on the provided material." is the canonical refusal. Includes a chart-block schema (function | scatter | vlines | hlines | points) with allowed operators/functions/constants matching `charts._ALLOWED_*`.
- `QUIZ_SYSTEM_PROMPT` — JSON-only output, no fences. MCQ: exactly 4 options, integer index. T/F: `options = ["True","False"]`, `correct_answer ∈ {0,1}`. Short-answer: `options=null`, 1–6-word canonical string. Grounded mode: every `correct_answer` MUST be justified by context; `source_refs` populated; no fabricated refs.
- `build_teacher_messages(question, context_block, history=None)` — assembles `[("system", ...), ...history, ("user", question + context_section)]`.
- `build_quiz_user_message(...)` — toggles GROUNDED/UNGROUNDED mode and writes the user-turn payload.

### 5.6 `app/quiz.py` — Schema + generation + grading

**Surface.**

- Pydantic models: `Question`, `Quiz`, `QuestionResult`, `GradeReport`. `Question._validate_options` rejects single-option lists.
- `generate_quiz(llm, topic, num_questions=5, difficulty="medium", question_types="mixed", context_chunks=None, max_retries=2)` — clamps count to `[1, 50]`. First tries `llm.with_structured_output(Quiz).invoke(messages)`; on failure logs a warning and falls into a parse-retry loop with up to `max_retries + 1` iterations, appending a corrective user nudge each time. Raises `ValueError` after exhausting attempts.
- `grade_quiz(quiz, answers)` — per-question `_grade_one`. MCQ/TF integer compare (type errors → incorrect, never crashes). Short-answer uses `_normalize_short_answer` (lowercase, collapse ws, strip edge punctuation, drop leading articles `"the "`, `"a "`, `"an "`) and word-boundary regex `\b<expected>\b` so single-letter expected answers can't match arbitrary substrings.

### 5.7 `app/charts.py` — Sandboxed inline charts

**Purpose.** Parse LLM-emitted fenced `chart` JSON blocks into Plotly figures with **multiple safety layers**.

**Surface & internals.**

- `CHART_FENCE_RE`, `PALETTE` (named-color → hex), `ChartSpecError(ValueError)`.
- `render(spec_text) -> (Figure|None, error|None)` — never raises; JSON errors, top-level non-dict, spec errors, and unexpected exceptions all surface as `(None, str)`.
- `split_text_and_charts(text)` — iterator yielding `("text", chunk)` / `("chart", spec)` in source order.
- `figure_to_png(fig, width, height, scale)` — wraps `fig.to_image(...)`; returns `None` on any failure (kaleido absent, no Chrome, runtime error). Callers must treat `None` as "fall back to placeholder".
- `_ALLOWED_NODES`: `Expression, BinOp, UnaryOp, Constant, Name, Call, Load, Add, Sub, Mult, Div, Pow, Mod, FloorDiv, USub, UAdd`. **No** `Attribute`, `Subscript`, `Compare`, `Lambda`, `ListComp`, `Starred`, `NamedExpr`, etc.
- `_ALLOWED_VARS = {x, pi, e}`, `_ALLOWED_FUNCS = {sin, cos, tan, exp, log, sqrt, abs}`.
- `_EVAL_GLOBALS = {"__builtins__": None, ...curated numpy funcs}` — even if a node type slipped past validation, name lookup has no `__builtins__` so escapes like `().__class__.__bases__[0].__subclasses__()` can't attach.
- `_MAX_POINTS = 500` per function series, `_MAX_SCATTER = 1000` per scatter, `_MAX_X_MAGNITUDE = 1e6` — DoS bounds.
- `_validate_ast` rejects dunder names redundantly (`startswith("__")`), refuses indirect calls (`node.func` must be `ast.Name`), refuses kwargs and starred args.
- `_safe_eval` masks non-finite outputs (`y[~np.isfinite(y)] = np.nan`) so `log(-1)` and divide-by-zero produce gaps, not crashes.

### 5.8 `app/pdf_export.py` — Markdown → PDF

**Public.** `markdown_to_pdf_bytes(title, body_markdown) -> bytes`, `pdf_filename(title, body) -> str`, `save_pdf_to_downloads(...)` (CLI/test path only).

**Pipeline.**

1. `_new_pdf` — A4, 15mm margins, `_install_fonts` probes `/usr/share/fonts/truetype/{noto,dejavu}`, falls back to Helvetica/Courier with `_sanitize_for_core_font` latin-1 mapping (smart quotes, dashes, bullets, arrows, ×).
2. `_prepare_text` — `_emoji_to_tags` strips emojis (explicit dict + `_EMOJI_BLOCK_RE` covering `\U0001F300-\U0001FAFF`, `\U0001F600-\U0001F64F`, `\U00002600-\U000027BF`, `\U0001F900-\U0001F9FF`, plus VS-16 and ZWJ) → `_flatten_markdown_links` rewrites `[label](url)` to bare `label` (sidesteps fpdf2 2.8.7 crash; flattening is skipped inside code fences via `flatten_links=False`).
3. `_render_body` — line-by-line state machine: chart fence (`_embed_chart` → kaleido PNG or `_placeholder_chart`), code fence (mono + light fill), markdown table (`_try_consume_table` + `pdf.table()` with `FontFace` headings), heading (size = `max(11, 18 - 2*(level-1))`), blockquote (italic grey), bullet (`"  - "` prefix), numbered list (original prefix), horizontal rule (1-line gray), plain paragraph (`multi_cell(markdown=True)`). Skips the first H1 that equals the supplied title to avoid duplicate cover heading.
4. `pdf.output()` → bytes.

### 5.9 `app/persistence.py` — Opt-in disk persistence

**Threat model.** The on-disk directory name is `sha256(token)`; a filesystem listing alone never reveals an active token. Only the user holding the raw token can construct the path. Sharing is share-link semantics, not user accounts.

**Layout.**

```
.cache/
├── faiss/<content_fingerprint>/   # shared, content-keyed
│   ├── index.faiss
│   └── index.pkl
└── sessions/<sha256(token)>/
    ├── files/<safe_name>
    └── manifest.json
```

**Surface.**

- `new_token()` → `uuid.uuid4().hex`.
- `_validate_token(s)` → regex `^[0-9a-f]{32}$`.
- `_hash_token(s)` → sha256 hex of validated token.
- `_safe_name(s)` → `Path(s).name`, rejects `""`/`.`/`..`.
- `SessionManifest` dataclass — `filenames, fingerprint, chunk_size, chunk_overlap, embed_backend, embed_model, created_at, last_accessed_at` (all UTC ISO 8601).
- `save_session` / `load_session` / `delete_session`. Load rewrites `last_accessed_at` so TTL is "days since last use".
- `save_faiss(fp, vs)` → `vs.save_local(...)`.
- `load_faiss(fp, embeddings) -> Optional[Any]` — `None` if `index.faiss` is absent; otherwise `FAISS.load_local(..., allow_dangerous_deserialization=True)`. **Safe only because we only load files we wrote ourselves under our own cache dir.**
- `cleanup_expired() -> int` — TTL = `SMART_TEACHER_SESSION_TTL_DAYS` (default 7). Removes expired *and* malformed dirs (missing `manifest.json` or unparseable `last_accessed_at`) so the cache self-heals.

---

## 6. Design rationale (the non-obvious why)

Every entry below is documented in code; citations point to the exact location.

| # | Decision | Why | Where |
|---|---|---|---|
| 1 | Pluggable provider factory + lazy imports | Adding a provider is a 2-step edit (table + branch); uninstalled providers don't crash startup. | `app/llm_factory.py:1-21, 201-264` |
| 2 | In-memory FAISS default; persistence opt-in | Matches HF Spaces' ephemeral container; opt-in keeps disk writes off by default. | `app/persistence.py:1-24`, `app/app.py:349-372` |
| 3 | Local `sentence-transformers` default | Free, no key, runs CPU, ~80 MB fits Spaces free tier. | `app/embeddings_factory.py:1-12, 40-62` |
| 4 | `with_structured_output` first, parse-retry fallback | Native schema enforcement on the providers that support it; fallback rescues HF/Ollama models that don't. | `app/quiz.py:213-243` |
| 5 | `@st.cache_resource` for embeddings/index, session-state LRU for LLM clients | Process-wide cache safe only when no user secret participates; LLM clients bound to per-session keys to avoid cross-tenant leak. | `app/app.py:81-219` |
| 6 | Key resolution sidebar > env > `st.secrets`; session-state only; `_redact_secrets` | Keys never logged; SDK errors that embed bearer tokens are stripped before render. | `app/llm_factory.py:113-153`, `app/app.py:222-240` |
| 7 | `sha256(token)` on-disk directory naming | Filesystem listing never reveals active tokens; raw token never lands on disk. | `app/persistence.py:106-107`, `_validate_token`, `_hash_token` |
| 8 | Notebook outputs skipped entirely | Defense in depth (text/html, application/javascript, base64 images) + prevents printed-secret leakage into the embedding index. | `app/rag.py:86-167` |
| 9 | AST whitelist *before* compile + `__builtins__=None` eval | Primary boundary blocks classic Python-sandbox escapes (`__class__`, `__subclasses__`, comprehensions, attribute access, subscripts, dunder names) at parse time; restricted globals are backstop. | `app/charts.py:85-122, 204-287` |
| 10 | PDF anchor-link flattening | fpdf2 2.8.7 crashes `pdf.output()` on `[label](#anchor)` because of unresolved named destinations; LLM emits these unpredictably; PDF link clickability is moot for offline reads. | `app/pdf_export.py:284-316` |
| 11 | PDF emoji stripping | fpdf2 lacks CBDT/CBLC color-bitmap support; prior `[MAP]/[OK]` tags were visual noise; clean strip + broad Unicode regex eliminates tofu glyphs. | `app/pdf_export.py:244-263`, `docs/pdf-export.md:141-156` |
| 12 | Math-symbol backtick rule in teacher prompt | Sans-serif fonts drop `∈`, `ℝ`, `Σ` etc. in bold/italic; inline-code wrapping renders reliably. | `app/prompts.py:69-77` |
| 13 | Stable `source#index` citation ids | Threaded splitter → retrieval → context → LLM → sources expander, end-to-end. | `app/rag.py:242-355`, `app/prompts.py:23-29` |
| 14 | `st.download_button` for PDFs + graceful kaleido fallback | No server-side filesystem write — works identically on local dev and HF Spaces; missing Chrome triggers `[Chart: title]` placeholder. | `app/app.py:606-634`, `app/charts.py:168-185`, `app/pdf_export.py:498-554` |
| 15 | sha256 fingerprint as cache key | `(chunk params, filenames, file bytes)` digest — re-uploads + restored sessions hit the same FAISS entry transparently. | `app/rag.py:358-387` |
| 16 | `_friendly_llm_error` translation | `UnicodeEncodeError` = key/prompt mix-up (per-provider key prefixes shown to help spot the swap); rate/auth/network errors get short actionable Markdown; full traceback only to server logs. | `app/app.py:811-868` |

---

## 7. Deployment

**Target.** HuggingFace Spaces, Streamlit SDK.

**README YAML front matter:**

```yaml
---
title: Smart Teacher
emoji: 🎓
sdk: streamlit
sdk_version: 1.40.2
python_version: "3.11"
app_file: app/app.py
---
```

**Provider keys.** Configured in the Space's "Variables and secrets"
(loaded via `st.secrets`). Users without configured keys can still
paste their own into the sidebar.

**First build.** ~5 min while it installs faiss + sentence-transformers.

**Chrome for chart PNGs in PDF.** Not installed on the default Space
image — charts fall back to `[Chart: <title>]` placeholders in the PDF.
To enable full-fidelity chart embedding, add `chromium` to a top-level
`packages.txt`.

**Persistent storage.** The default Space container is ephemeral.
Upgrade to persistent storage and point
`SMART_TEACHER_CACHE_DIR=/data/smart-teacher` to make opt-in
persistence survive restarts.

---

## 8. Known limitations

- **In-memory FAISS** rebuilt on cold start unless persistence is on. Fine for ≤100 MB corpora; for larger use a managed vector DB.
- **Single-document session** without persistence. The opt-in model uses share-link semantics (anyone with the token can restore), not user accounts.
- **Citation fidelity** depends on the underlying LLM. Smaller models sometimes fabricate chunk ids — prefer Groq Llama-3.3-70B or Claude.
- **Short-answer grading** uses normalized substring + word-boundary regex, not semantic similarity. Phrase canonical answers tightly.
- **No streaming** — chat waits for the full reply before rendering.
- **Ollama is local-dev only** — the Space deployment cannot reach `localhost:11434`.

---

## 9. Extension recipes

### Add a new LLM provider

1. Append a `ProviderSpec(...)` entry to `PROVIDERS` in `app/llm_factory.py`.
2. Add an `if provider == "<key>": ...` branch inside `get_llm()` that lazy-imports the package and constructs the chat model.
3. Pin the provider's `langchain-<x>` package in `requirements.txt`.

### Add a new embedding backend

Same shape, in `app/embeddings_factory.py`.

### Add a new file format

1. Add the suffix to `SUPPORTED_SUFFIXES` in `app/rag.py`.
2. Add a branch to `_read_file_to_documents` that returns `List[Document]` with `metadata["source"]` set to the original safe filename.

### Add a new chart series type

1. Extend `_build_figure` in `app/charts.py` to dispatch on `s_type`.
2. Document the new shape inside `TEACHER_SYSTEM_PROMPT` so the LLM knows it can emit it.

---

## 10. File index

```
ai_s4_app/
├── app/
│   ├── app.py                 # Streamlit UI + caches + secret hygiene
│   ├── llm_factory.py         # 6-provider LLM factory
│   ├── embeddings_factory.py  # 3-backend embeddings factory
│   ├── rag.py                 # load → split → embed → FAISS → retrieve
│   ├── prompts.py             # teacher + quiz system prompts
│   ├── quiz.py                # Pydantic Quiz + structured-output + grading
│   ├── charts.py              # AST-sandboxed Plotly chart renderer
│   ├── pdf_export.py          # fpdf2 markdown → PDF + chart embed
│   └── persistence.py         # opt-in sha256(token) disk cache
├── tests/
│   ├── test_charts.py         # AST sandbox boundary tests
│   └── test_pdf_export.py     # fpdf2 anchor-link regression tests
├── docs/
│   ├── architecture.md        # this file
│   └── pdf-export.md          # PDF feature deep-dive
├── .streamlit/config.toml     # server + theme
├── .env.example               # env-var template
├── requirements.txt           # pinned deps with audit notes
└── README.md                  # user-facing intro + provider table
```
