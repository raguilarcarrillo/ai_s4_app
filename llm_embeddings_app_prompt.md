# User Prompt — Build a Messy-Data Correlation App

> Copy everything below the horizontal rule into your conversation with
> Claude (Code or API) when you're ready to start building. The prompt
> is self-contained — it specifies the architecture pattern, the file
> formats to support, the correlation engine, the UI, the schemas, the
> security model, and the acceptance criteria.

---

You are going to build **DataCorrelator**, a Streamlit + LangChain
application that ingests messy, heterogeneous data files and helps a
user discover relationships, duplicates, and inconsistencies across
them. The architecture must follow the exact same foundations as the
Smart Teacher reference app described below, then specialize for the
data-correlation use case.

## 1. Reference foundation (copy these patterns exactly)

The architecture is taken from a working app called **Smart Teacher**.
You will replicate the following patterns verbatim, only swapping the
domain layer:

### 1.1 Pluggable LLM factory

A single `llm_factory.py` module exposes a frozen `ProviderSpec`
dataclass (`key, label, default_model, env_var, needs_key, notes`), a
`PROVIDERS: Dict[str, ProviderSpec]` table covering at minimum
**Anthropic Claude, OpenAI, Google Gemini, Groq, HuggingFace, and
Ollama**, and a `get_llm(provider, model=None, temperature=0.2,
api_key=None, **kwargs)` entry point. Each provider SDK is imported
**lazily inside the matching branch** so cold start is cheap and an
uninstalled optional provider never crashes startup.

Key resolution order is **sidebar input > `os.environ` >
`st.secrets`**. Wrap the `st.secrets` lookup in a bare try/except so
the factory works outside Streamlit (CLI, tests). Anthropic default
model is `claude-sonnet-4-6`; Groq default is `llama-3.3-70b-versatile`
(free dev tier — recommended for first-run smoke tests).

### 1.2 Pluggable embeddings factory

Mirror of the LLM factory in `embeddings_factory.py`. Backends:

- `sentence-transformers` (DEFAULT, no key, `all-MiniLM-L6-v2` ~80 MB,
  CPU, `normalize_embeddings=True`). Use this via
  `langchain_huggingface.HuggingFaceEmbeddings`.
- `huggingface` (HF Inference API, needs `HUGGINGFACEHUB_API_TOKEN`).
- `openai` (`text-embedding-3-small`).

### 1.3 Streamlit shell with two-layer caching

- `@st.cache_resource` for things with **no per-user secret** (the
  embedding model, the FAISS index). Process-wide; safe.
- `st.session_state` LRU dicts for **per-tenant** things (the LLM
  client, rendered PDF bytes). Key the LLM cache by
  `provider::model::temperature::sha256-hint(key)`; bound to 4
  entries. Key the PDF cache by `title::sha256(content)[:16]`; bound
  to 32 entries. **Never put LLM clients in `@st.cache_resource` —
  that leaks the first user's key to subsequent users.**

### 1.4 RAG-style indexing pipeline

The reference app's `rag.py` does load → split → embed → FAISS →
retrieve. You will refactor this for records (not text chunks) — see
section 2. Reuse:

- `RecursiveCharacterTextSplitter` with
  `separators=["\n\n","\n",". "," ",""]` **only for free-form text
  fields** that are too long to embed as a single record.
- FAISS in-memory vector store via
  `langchain_community.vectorstores.FAISS`. CPU build.
- Stable id format `<source>#<record_path>` (Smart Teacher uses
  `<source>#<chunk_index>`).
- Convert FAISS L2 distance to a similarity score `1/(1+distance)` so
  higher = better.
- `fingerprint(files, chunk_size, chunk_overlap)` — sha256 over chunk
  params + filenames + file bytes, accepting either Streamlit
  `UploadedFile` or `pathlib.Path` and producing the **same digest for
  byte-identical content** so cache hits work across fresh uploads and
  restored sessions.

### 1.5 Opt-in disk persistence

Replicate `persistence.py` exactly:

- Two on-disk caches under `DATA_CORRELATOR_CACHE_DIR` (default
  `./.cache/`):
  - `faiss/<content_fingerprint>/` — shared, content-keyed FAISS index.
  - `sessions/<sha256(token)>/{files/<safe_name>, manifest.json}` —
    per-session blob. **The raw token never lands on disk** — only its
    sha256 hash becomes the directory name. Filesystem listing alone
    does not reveal active tokens.
- `new_token()` → `uuid.uuid4().hex` (32 hex chars).
- `_validate_token` enforces regex `^[0-9a-f]{32}$`.
- `_safe_name` rejects `""`, `.`, `..`, strips path components.
- `SessionManifest` dataclass with UTC ISO 8601 timestamps.
  `load_session` rewrites `last_accessed_at` so TTL is **days since
  last use**, not days since creation.
- `save_faiss(fp, vs)` uses `vs.save_local(...)`.
- `load_faiss(fp, embeddings)` returns `None` if `index.faiss` is
  absent, otherwise `FAISS.load_local(..., allow_dangerous_deserialization=True)`. The flag is safe here ONLY because we load files we wrote ourselves. Document this inline.
- `cleanup_expired()` removes sessions older than
  `DATA_CORRELATOR_SESSION_TTL_DAYS` (default 7) **and** any malformed
  dir missing `manifest.json` so the cache self-heals.
- Persistence toggle is **strict opt-in** in the sidebar; nothing
  touches disk unless the user actively checks the box.

### 1.6 Structured output with Pydantic + fallback

For every schema-bound LLM call:

- First attempt: `llm.with_structured_output(MyModel).invoke(messages)`.
- On exception, log a warning and fall through to a parse-retry loop
  (up to 3 attempts) using plain `invoke` + a `_parse_payload` helper
  that handles `MyModel` instance, `dict`, JSON string with
  `_strip_code_fences` applied, or LangChain message via `.content`.
  Append a corrective user turn each retry: *"Your previous output
  was not valid JSON matching the schema. Return ONLY the JSON
  object, no prose, no fences."*

### 1.7 PDF export

A `pdf_export.py` module with `markdown_to_pdf_bytes(title, body) ->
bytes` and `pdf_filename(title, body) -> str`. Use `fpdf2>=2.8.7,<3.0`.
Probe `/usr/share/fonts/truetype/{noto,dejavu}` for unicode TTFs, fall
back to Helvetica + a `_sanitize_for_core_font` latin-1 pass. Strip
emojis (explicit dict + broad Unicode-block regex including VS-16 and
ZWJ). **Flatten `[label](url)` markdown links to bare `label`** before
fpdf2 sees them — fpdf2 2.8.7 crashes on unresolved named
destinations. Skip flattening inside fenced code blocks. Render to
`bytes` and hand to `st.download_button` so the browser saves with no
server-side filesystem write (HF Spaces friendly). Filename pattern
`<topic-slug>_<YYYY-MM-DD>.pdf`.

### 1.8 Defense in depth

- `_redact_secrets(text)` sweeps every `st.session_state[f"key::{provider}"]` literal occurrence out of any user-facing error string before render.
- `_friendly_llm_error(exc)` translates raw exceptions into actionable Markdown: `UnicodeEncodeError` = the user pasted prompt text into the API key field; rate-limit/429, auth/401, connection/timeout/network each get a short hint. Full traceback only to server logs.
- Filename sanitization in every file-IO function (basename only, no `..`, tempdir-confinement check via resolved-path parent check).
- Use `tempfile.TemporaryDirectory` for uploaded file IO; refuse any resolved path whose parents don't contain the tempdir.

### 1.9 Deployment target

**HuggingFace Spaces**, Streamlit SDK, Python 3.11. README YAML front
matter declares `app_file: app/app.py`. All heavy primitives must work
on the free-tier ephemeral container: no GPU, no Chrome by default
(graceful fallback paths required), provider keys readable from
`st.secrets`, default embeddings local and free.

## 2. Domain specialization — DataCorrelator

The reference app teaches from documents. DataCorrelator **correlates
records across heterogeneous files**.

### 2.1 Supported file formats

| Format | Extension(s) | Loader plan |
|---|---|---|
| JSON | `.json` | `json.load`. Walk the structure; emit one **record** per leaf object (dict). Use `jsonpath-ng` or a hand-rolled recursive walk to track each record's path (e.g. `users[3].address`). |
| Excel | `.xlsx`, `.xls` | `openpyxl` (preferred over pandas — lighter, no NumPy/Pandas reload cost). Emit one record per data row, per sheet. The header row supplies field names. |
| DNS BIND zone | `.zone`, `.txt` | Use `dnspython` (`dns.zone.from_text`) to parse. Emit one record per RRset (or per RR if multi-record). Capture origin, owner name, TTL, class, type, rdata. **Support `$ORIGIN`, `$TTL`, `$INCLUDE` directives, owner-name inheritance, multi-line records in parens, and comments (`;`).** Record types to support: A, AAAA, CNAME, MX, NS, TXT, SOA, PTR, SRV, CAA. |
| YAML | `.yaml`, `.yml` | `pyyaml` (`yaml.safe_load`!). Same walk as JSON; emit one record per leaf object. |

Pin all loaders in `requirements.txt`. Lazy-import each loader inside
its branch in the format-dispatch function — same pattern as Smart
Teacher's `rag._read_file_to_documents`.

### 2.2 Record model

Replace `RetrievedChunk` with a `DataRecord` dataclass:

```python
@dataclass
class DataRecord:
    record_id: str          # stable: "<source>#<path>"
                            # e.g. "users.json#users[3]"
                            # e.g. "acme.zone#www.A"
                            # e.g. "inventory.xlsx#Sheet1!A7"
    source: str             # original filename (sanitized basename)
    record_type: str        # "json_object" | "excel_row" | "dns_rr" | "yaml_object"
    fields: dict[str, Any]  # normalized field map
    text_repr: str          # serialized form for embedding + LLM context
    raw: Any                # original parsed object (for export)
    score: float | None = None   # set only by retrieve()
```

The `text_repr` is what gets embedded. Keep it deterministic and
human-readable, e.g.:

```
[users.json#users[3]]
name=Alice
email=alice@example.com
hostname=alice-laptop.example.com
```

For DNS RRs:

```
[acme.zone#www.A]
owner=www.acme.com.
type=A
ttl=3600
rdata=192.0.2.1
```

### 2.3 Indexing pipeline

`rag.py` becomes `indexer.py`:

1. `load_uploaded_files(files)` writes each upload into a
   `tempfile.TemporaryDirectory` with path-traversal guards (resolved
   path must be under the tempdir).
2. `extract_records(path, source_label)` dispatches by suffix and
   returns `List[DataRecord]`.
3. `build_record_index(records, embeddings)` returns a `FAISS`
   instance built from each record's `text_repr` (use
   `FAISS.from_texts(texts, embedding=embeddings, metadatas=[...])`
   so each FAISS document carries the `record_id` and `record_type`).
4. `retrieve(index, query, k=5)` — semantic search, same
   distance-to-similarity conversion as Smart Teacher.
5. `fingerprint(files, ...)` — same sha256 design as Smart Teacher,
   includes any user-configurable normalization params.

### 2.4 Three retrieval / correlation layers

Build all three. They are complementary:

**Layer A — exact-match index** (`exact_index.py`).
Build inverted indexes over normalized field values:
`field_value → set[record_id]`. Normalize by trimming and
lower-casing strings, parsing IPv4/IPv6 into canonical form
(`ipaddress.ip_address(...)`), normalizing FQDNs (strip trailing
dot, lowercase). Useful for "find every record whose IP is X" or
"find every record referencing this hostname".

**Layer B — semantic FAISS index** (already covered above).
Useful for fuzzy similarity ("records similar to this one" — even when
field names differ).

**Layer C — LLM-explained correlations** (`correlator.py`).
A Pydantic-bound LLM call that takes the user's question + the
top-k records (from A and/or B) and returns a structured
`CorrelationReport` explaining the relationships.

### 2.5 Pydantic schemas

```python
class CorrelationFinding(BaseModel):
    kind: Literal[
        "duplicate",         # two records describe the same entity
        "reference",         # one record references another (e.g. CNAME → A)
        "conflict",          # records disagree on a value (e.g. two A records, different IPs)
        "missing_reference", # a reference points to no defined record
        "orphan",            # a record nothing references
        "related",           # weak semantic similarity worth noting
    ]
    record_ids: list[str]
    explanation: str
    confidence: Literal["low", "medium", "high"]
    suggested_action: str | None = None

class CorrelationReport(BaseModel):
    question: str
    summary: str             # one-paragraph human-readable summary
    findings: list[CorrelationFinding] = Field(..., min_length=0)

class CorrelationAnalysis(BaseModel):
    """Output of a 'find all correlations' batch run, not a single Q."""
    summary: str
    duplicates: list[CorrelationFinding] = Field(default_factory=list)
    references: list[CorrelationFinding] = Field(default_factory=list)
    conflicts: list[CorrelationFinding] = Field(default_factory=list)
    orphans: list[CorrelationFinding] = Field(default_factory=list)
```

Use `with_structured_output(CorrelationReport)` first, then the
parse-retry fallback. **Every `record_ids` value must reference an
actually-indexed record** — the system prompt must forbid
fabrication, exactly like Smart Teacher forbids fabricated
`source_refs`.

### 2.6 System prompt for the correlator LLM

Write `prompts.py` with a `CORRELATOR_SYSTEM_PROMPT` that:

- Identifies the assistant as a data steward / asset inventory
  analyst.
- Instructs the LLM to cite `record_id` values inline as bare
  `[source#record_path]` (NOT `[label](url)` — Smart Teacher learned
  this crashes fpdf2 and we use the same PDF renderer).
- Defines the six finding kinds with examples.
- Forbids fabrication: "If no records in the provided context support
  a claim, say so. Do not invent record ids."
- Requires confidence calibration (low/medium/high) and instructs the
  LLM to use "low" when retrieval is sparse.

Write a `build_correlator_messages(question, context_records, history)`
helper that mirrors Smart Teacher's `build_teacher_messages`.

### 2.7 UI — three tabs

- **🔍 Ask** — natural-language search over the records. Like Smart
  Teacher's chat tab. User asks a question, the app retrieves top-k
  records (semantic), the LLM produces a `CorrelationReport`, the
  findings render with expandable per-record cards plus a `📄 Download
  as PDF` button.
- **🧬 Correlate** — batch correlation analysis. User picks scope
  (all records | specific file | specific record type). App runs the
  exact-match passes (Layer A) to surface duplicates, conflicts,
  missing references, and orphans **deterministically**, then asks
  the LLM (Layer C) to produce a `CorrelationAnalysis` that
  summarizes and prioritizes the deterministic findings. Render with
  a `📄 Download as PDF` for the full report.
- **📋 Records** — flat browser of all indexed records with search +
  filter by source / record_type. Lets the user verify what was
  ingested.

A sidebar identical in structure to Smart Teacher: provider/model/key,
temperature, top-k, embedding backend/model, file upload (multi-file,
size cap matched to Streamlit's 50 MB), opt-in persistence panel,
clear-history button.

### 2.8 Deterministic correlation passes (Layer A details)

Implement these as plain Python — no LLM. They produce
`CorrelationFinding[]` that feeds into Layer C.

| Pass | What it detects | Example |
|---|---|---|
| `find_exact_duplicates` | Records whose normalized `fields` dict is identical | Two A records for the same owner with the same IP, in two different zone files |
| `find_value_collisions` | Records that share a canonical key (e.g. hostname) but disagree on a value | Excel row says `alice@old.example` but YAML says `alice@example.com` |
| `find_missing_references` | Records that reference a name/IP that no record defines | `mail.example.com IN MX` → no A record for `mail.example.com` anywhere |
| `find_orphans` | Records nothing references and that themselves reference nothing | An A record for a host not mentioned in any other file |
| `find_ip_cross_refs` | Same IP appears in records from different files | `192.0.2.10` in `acme.zone` and in an Excel `inventory.xlsx` row |
| `find_hostname_cross_refs` | Same canonical FQDN appears in records from different files | `mail.example.com.` in zone + `mail.example.com` (no trailing dot) in YAML |

Each pass returns `CorrelationFinding(kind=..., record_ids=[...],
explanation=..., confidence="high")`. Cite confidence "high" because
these passes are exact, not heuristic.

### 2.9 Tests

A `tests/` directory with at minimum:

- `test_indexer.py` — per-format extractors. Ship small fixture files
  (`tests/fixtures/sample.json`, `sample.yaml`, `sample.zone`,
  `sample.xlsx`). Assert record counts and stable record ids.
- `test_correlator.py` — exact-match passes against known-input fixtures
  (a duplicate, a missing reference, an orphan, an IP cross-ref).
- `test_pdf_export.py` — regression test for `[label](#anchor)` link
  flattening (mirror Smart Teacher's `tests/test_pdf_export.py`).
- `test_persistence.py` — round-trip save/load with a fresh token;
  TTL cleanup removes a manifest older than the cutoff; invalid token
  rejected.
- `test_safety.py` — uploaded files with `../etc/passwd` style names
  are rejected by `_safe_name`; YAML loader uses `safe_load`; JSON
  loader rejects gigabyte inputs gracefully.

### 2.10 Pinned dependencies

```
streamlit>=1.40
pydantic>=2.9
python-dotenv>=1.0

langchain>=0.3.13
langchain-core>=0.3.28
langchain-community>=0.3.13
langchain-text-splitters>=0.3.4

langchain-anthropic>=0.3
langchain-openai>=0.2.14
langchain-google-genai>=2.0
langchain-groq>=0.2
langchain-huggingface>=0.1.2
langchain-ollama>=0.2

sentence-transformers>=3.3
faiss-cpu>=1.9

# Format loaders
openpyxl>=3.1          # Excel
PyYAML>=6.0            # YAML (use safe_load)
dnspython>=2.6         # BIND zone files
jsonpath-ng>=1.6       # JSON path navigation

# PDF export — audit equivalent to Smart Teacher
fpdf2>=2.8.7,<3.0.0

# HuggingFace runtime (transitive but pin defensively)
huggingface-hub>=0.27
transformers>=4.47
torchvision>=0.20      # transformers needs this for zoedepth imports
```

Add audit comments above any package with known historical advisories
(see Smart Teacher's `requirements.txt` for the format).

## 3. Module layout

```
data_correlator/
├── app/
│   ├── app.py                    # Streamlit shell + caches + secret hygiene
│   ├── llm_factory.py            # 6-provider LLM factory (copy from Smart Teacher)
│   ├── embeddings_factory.py     # 3-backend embeddings factory (copy)
│   ├── indexer.py                # load → extract → embed → FAISS → retrieve
│   ├── extractors/
│   │   ├── __init__.py           # extract_records(path, source_label) dispatcher
│   │   ├── json_extractor.py
│   │   ├── excel_extractor.py
│   │   ├── bind_extractor.py     # uses dnspython
│   │   └── yaml_extractor.py     # uses safe_load
│   ├── exact_index.py            # Layer A — inverted indexes + deterministic passes
│   ├── correlator.py             # Layer C — LLM-bound correlation report
│   ├── prompts.py                # CORRELATOR_SYSTEM_PROMPT + builders
│   ├── pdf_export.py             # fpdf2 markdown → PDF (copy + adapt)
│   └── persistence.py            # opt-in sha256(token) disk cache (copy)
├── tests/
│   ├── fixtures/
│   │   ├── sample.json
│   │   ├── sample.yaml
│   │   ├── sample.zone
│   │   └── sample.xlsx
│   ├── test_indexer.py
│   ├── test_correlator.py
│   ├── test_pdf_export.py
│   ├── test_persistence.py
│   └── test_safety.py
├── docs/
│   ├── architecture.md
│   └── bind-zone-handling.md     # parsing details + supported RR types
├── .streamlit/config.toml
├── .env.example
├── requirements.txt
└── README.md
```

## 4. Security and accuracy requirements

These are non-negotiable. Implement them from day one, not as a "we'll
harden later" pass.

1. **YAML must use `yaml.safe_load`** — never `yaml.load`. Constructed
   objects are an arbitrary-code-execution vector.
2. **JSON loader must reject inputs larger than a configured cap**
   (default 50 MB). Stream-parse if larger formats are needed later.
3. **BIND parser must isolate per-file `$ORIGIN`** — `dnspython` does
   this correctly when you use `dns.zone.from_text(..., origin=...)`.
   Never let one file's directives leak into another's parse.
4. **FAISS `load_local` must only be called on directories we wrote
   ourselves** under the cache dir. Document this inline next to the
   `allow_dangerous_deserialization=True` flag, exactly as the
   reference app does.
5. **API keys never log, never echo to UI** — same `_redact_secrets`
   sweep applied to every rendered error.
6. **Filename sanitization at every IO boundary** — uploads,
   restored files, manifest filenames. `_safe_name` rejects `.`,
   `..`, empty, anything with path separators.
7. **Opt-in persistence is OFF by default.** Even local dev shouldn't
   write to disk without the toggle.
8. **Token format strictly `^[0-9a-f]{32}$`.** Reject anything else
   in `_validate_token` with a clear error.
9. **TTL cleanup removes malformed dirs** so the cache self-heals.
10. **No emoji glyphs in PDF body text** — they don't render in
    fpdf2 (no CBDT/CBLC). Strip them; the bold heading text carries
    meaning on its own.

## 5. Accuracy contract — "do not invent"

This is the single most important behavioral requirement. The
correlator LLM **must not fabricate record ids**. Bake the rule into:

- The system prompt: "If no records in the provided context support a
  claim, say so. Do not invent record ids. Set confidence to 'low'
  when retrieval is sparse."
- A post-processing validator: after parsing the
  `CorrelationReport`, verify that every `record_id` in every
  finding actually exists in the index. Drop findings that reference
  non-existent ids and log a warning. (Smart Teacher does the
  equivalent at the citation level.)
- A unit test that feeds a tiny corpus + a question and asserts the
  validator catches an injected hallucination.

## 6. Acceptance criteria

The build is complete when:

1. **Provider parity.** A user can switch between Anthropic, Groq, and
   Ollama via the sidebar and run the same question with consistent
   structured output. (HF and Gemini and OpenAI also work but are
   tested manually.)
2. **All four loaders work.** Upload one JSON, one Excel (multi-sheet),
   one BIND zone (with `$ORIGIN`, `$TTL`, MX/A/CNAME records), and
   one YAML file. The Records tab shows every record with a stable id.
3. **Exact-match passes.** Insert a known duplicate, a known missing
   reference, a known IP cross-reference into the fixtures. The
   Correlate tab surfaces each as a `CorrelationFinding` with
   `confidence="high"` before the LLM is involved.
4. **LLM correlation.** The Ask tab returns a `CorrelationReport`
   whose findings cite real record ids only (no hallucinated paths).
5. **PDF export.** Both Ask and Correlate tabs offer working PDF
   downloads delivered via `st.download_button` — no server
   filesystem writes.
6. **Opt-in persistence round-trip.** Toggle persistence → upload
   files → note the token → close the tab → reopen → paste the token →
   files restored and the FAISS index is reused (verify by absence of
   the "Indexing documents…" spinner).
7. **Safety tests pass.** Every test in `tests/test_safety.py` is
   green: bad filenames rejected, YAML safe-loaded, FAISS
   round-trip works.
8. **HF Spaces deploy works.** Push to a fresh Space → first build
   ≤10 min → app loads → default Groq provider answers a question
   end-to-end with no extra configuration once `GROQ_API_KEY` is set
   in Space secrets.

## 7. Concrete example fixtures

Include these in `tests/fixtures/` so the acceptance criteria are
testable on day one.

### 7.1 `sample.zone`

```
$ORIGIN example.com.
$TTL 3600
@        IN  SOA  ns1.example.com. admin.example.com. (
                 2024010101 ; serial
                 7200       ; refresh
                 3600       ; retry
                 1209600    ; expire
                 86400 )    ; minimum
@        IN  NS   ns1.example.com.
@        IN  NS   ns2.example.com.
@        IN  MX   10 mail.example.com.
ns1      IN  A    192.0.2.1
ns2      IN  A    192.0.2.2
www      IN  A    192.0.2.10
mail     IN  A    192.0.2.20
api      IN  A    192.0.2.30
docs     IN  CNAME www.example.com.
legacy   IN  CNAME old.removed.example.com.   ; intentional missing reference
```

Expected findings on this file alone:

- 1 `missing_reference` — `legacy → old.removed.example.com` is unresolved.
- 1 `reference` chain — `docs → www → 192.0.2.10`.
- 5 A records as plain records, no conflict.

### 7.2 `sample.json`

```json
{
  "hosts": [
    { "name": "www", "ip": "192.0.2.10", "owner": "team-web" },
    { "name": "api", "ip": "192.0.2.30", "owner": "team-platform" },
    { "name": "mail", "ip": "192.0.2.99", "owner": "team-ops" }
  ]
}
```

Expected when run alongside `sample.zone`:

- 1 `conflict` — `mail.example.com` is `192.0.2.20` in the zone but
  `192.0.2.99` in JSON.
- 2 `ip_cross_refs` — `192.0.2.10` and `192.0.2.30` appear in both files.

### 7.3 `sample.yaml`

```yaml
inventory:
  - hostname: www.example.com
    asset_tag: A0001
  - hostname: api.example.com
    asset_tag: A0002
  - hostname: ghost.example.com   # orphan
    asset_tag: A0099
```

Expected:

- 1 `orphan` — `ghost.example.com` has no matching record anywhere.

### 7.4 `sample.xlsx`

A single sheet with header row `hostname, ip, owner, location` and
five rows. At least one row should duplicate `www → 192.0.2.10` to
exercise the cross-source duplicate finder, and at least one row
should disagree on the owner field versus the JSON to exercise
`find_value_collisions`.

## 8. Style and quality

- Match the reference app's docstring discipline: every public
  function has a Google-style docstring with Args/Returns/Raises.
- Inline comments explain *why*, not *what*. The reference app uses
  comments effectively for documenting security-sensitive choices
  (FAISS deserialization, AST whitelist, sha256 token directories) —
  do the same.
- Frozen dataclasses for static metadata (`ProviderSpec`,
  `EmbeddingSpec`, `ExtractorSpec` if you build one).
- Lazy imports inside dispatch branches keep cold start cheap.
- No global mutable state outside `st.session_state` and the cache
  decorators.

## 9. First milestone

Aim for a vertical slice in this order:

1. `llm_factory.py` + `embeddings_factory.py` (copy from Smart
   Teacher, add `DATA_CORRELATOR_*` env knobs).
2. `extractors/json_extractor.py` only; `indexer.py` that returns
   `List[DataRecord]` and builds a FAISS index.
3. `app/app.py` with just the sidebar + Records tab — upload a JSON,
   see records.
4. Add `extractors/bind_extractor.py` and `yaml_extractor.py` and
   `excel_extractor.py` one at a time, each with a smoke test.
5. `exact_index.py` + the Correlate tab with deterministic
   passes only.
6. `correlator.py` + the Ask tab + `pdf_export.py`.
7. `persistence.py` + sidebar persistence panel.
8. Tests, then HF Spaces deploy.

Do not add features outside this list until the milestone is done.

---

**When you start, confirm you've read this prompt by stating: which
file you'll create first, what you'll defer, and any clarifying
question you have about the BIND parser configuration (the trickiest
loader). Then begin.**
