---
title: Smart Teacher
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.2
python_version: "3.11"
app_file: app/app.py
pinned: false
license: mit
---

# 🎓 Smart Teacher

**Smart Teacher** is an LLM-agnostic, retrieval-augmented AI tutor built with
Streamlit + LangChain. Give it a topic and/or some documents (PDF, TXT, MD)
and it returns a pedagogically-structured answer — explanation, recommended
learning method for *that* subject, a step-by-step study plan, practice
exercises, and self-check questions — with inline citations to your source
material. It can also generate, grade, and review quizzes on demand.

Switch between Anthropic, OpenAI, Gemini, Groq, HuggingFace, or local Ollama
models from the sidebar at runtime — no code changes required.

---

## Why it matters

Generic chatbots tell you things. Smart Teacher *teaches* you things:

- It picks the **right learning method** for the topic (spaced repetition for
  vocabulary, Feynman for theory, deliberate practice for skills, etc.).
- It **grounds** every claim in your uploaded materials and shows you the
  exact chunks it used.
- It **assesses** you with quizzes whose correct answers are justified by the
  same source material, so learning + evaluation share one source of truth.
- It's **LLM-agnostic** — start free on Groq or local Ollama, scale up to
  Claude Sonnet for higher-stakes work, without touching code.

---

## Tech stack

| Layer | Choice |
|---|---|
| App shell | **Streamlit** (`st.chat_message`, `st.tabs`, `st.cache_resource`) |
| Orchestration | **LangChain** (loaders, splitters, retriever, structured output) |
| LLM providers | Anthropic · OpenAI · Google Gemini · Groq · HuggingFace · Ollama |
| Embeddings (default) | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | **FAISS** (in-memory) |
| Schemas | **Pydantic v2** |
| Deploy | **HuggingFace Spaces** (Streamlit SDK) |

---

## Supported providers

| Provider | Required env var / secret | Default model | Notes |
|---|---|---|---|
| Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | Paid; best reasoning. |
| OpenAI GPT | `OPENAI_API_KEY` | `gpt-4o-mini` | Paid; widely supported. |
| Google Gemini | `GOOGLE_API_KEY` | `gemini-2.5-flash` | Generous free tier on Flash. Also try `gemini-2.5-pro` or `gemini-3.1-flash-lite`. |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Free developer tier; very fast. |
| HuggingFace | `HUGGINGFACEHUB_API_TOKEN` | `meta-llama/Meta-Llama-3-8B-Instruct` | Free tier with rate limits. |
| Ollama (local) | _(none)_ | `llama3.2` | Needs Ollama running locally. |

Keys are resolved in this order: **sidebar input → `os.environ` →
`st.secrets`**. They live only in `st.session_state` and are never logged.

---

## How to add a new provider

Editing `app/llm_factory.py` is the only required change:

1. Add a `ProviderSpec(...)` entry to the `PROVIDERS` dict.
2. Add a branch inside `get_llm()` that imports the package lazily and
   instantiates the chat model.
3. Pin the provider's `langchain-<x>` package in `requirements.txt`.
4. (Optional) Document it in the table above.

```python
# app/llm_factory.py
PROVIDERS["mistral"] = ProviderSpec(
    key="mistral", label="Mistral", default_model="mistral-large-latest",
    env_var="MISTRAL_API_KEY", needs_key=True, notes="Paid API.",
)

# inside get_llm():
if provider == "mistral":
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(model=model, temperature=temperature,
                         mistral_api_key=resolved_key, **kwargs)
```

That's it — the rest of the app (RAG, prompts, quiz, UI) is provider-unaware.

---

## Project structure

```
app/
├── app.py                  # Streamlit entry point + UI
├── llm_factory.py          # Pluggable LLM provider abstraction
├── embeddings_factory.py   # Pluggable embedding provider abstraction
├── rag.py                  # Loading, splitting, indexing, retrieval
├── prompts.py              # Teacher + quiz system prompts and templates
├── quiz.py                 # Pydantic schema, generation, grading
└── pdf_export.py           # 📄 Save as PDF — markdown → PDF renderer

docs/
└── pdf-export.md           # Architecture + customization guide for PDF export

requirements.txt            # Pinned dependencies for all providers
.env.example                # Template for local env-var setup
.streamlit/config.toml      # Streamlit server + theme config
```

### PDF export

Every assistant answer and quiz result has a `📄 Save as PDF` button that
writes a `<topic>_<YYYY-MM-DD>.pdf` file into `~/Downloads/`. See
[`docs/pdf-export.md`](docs/pdf-export.md) for the rendering pipeline,
font story, emoji-to-tag map, customization knobs, and security notes.

---

## Local install + run

```bash
# 1. Clone & create a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (covers ALL supported providers)
pip install -r requirements.txt

# 3. Configure keys (any subset — only the provider you use is required)
cp .env.example .env
# edit .env and set the keys you have

# 4. Run
streamlit run app/app.py
```

The first run downloads the local embedding model (~80 MB) once.

### Provider-specific tips

- **Groq** (recommended for first run): free, fast — just create a key at
  `console.groq.com` and paste it in the sidebar.
- **Ollama**: install Ollama, run `ollama pull llama3.2`, then choose
  *Ollama (local)* in the sidebar. Set `OLLAMA_BASE_URL` if running on a
  non-default host.
- **HuggingFace**: small models on the free tier may rate-limit; switch to
  Groq if you hit limits.

---

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Streamlit   │ →  │  rag.py          │ →  │  FAISS index │
│  app.py      │    │  load → split →  │    │  (in-memory) │
└──────┬───────┘    │  embed → search  │    └──────┬───────┘
       │            └──────────────────┘           │
       │                                           ▼
       │            ┌──────────────────┐    ┌──────────────┐
       │            │ embeddings_      │    │ Retrieved    │
       │            │ factory.py       │    │ chunks +     │
       │            └──────────────────┘    │ citations    │
       │                                    └──────┬───────┘
       ▼                                           │
┌──────────────┐    ┌──────────────────┐           │
│  prompts.py  │ →  │  llm_factory.py  │ ← ────────┘
│  teacher +   │    │  get_llm(prov.)  │
│  quiz        │    └────────┬─────────┘
└──────┬───────┘             │
       │                     ▼
       │            ┌──────────────────┐
       └─────────→  │  quiz.py         │
                    │  Pydantic schema │
                    │  + grading       │
                    └──────────────────┘
```

- **`llm_factory` + `embeddings_factory`** are the *only* points coupled to
  specific providers. Everything else flows through LangChain interfaces.
- The RAG flow: upload → split into chunks (with stable `source#index`
  ids) → embed → index in FAISS → on each question, retrieve top-k →
  format as a context block → prepend to the teacher prompt.
- The quiz flow reuses the same retrieval to produce a grounded quiz whose
  every `correct_answer` references chunk ids from the index.

---

## Deployment to HuggingFace Spaces

1. Create a **new Space** → Streamlit SDK → Python 3.10.
2. Push the repo (or upload files via the web UI). Keep the same layout —
   HF Spaces runs `streamlit run app/app.py` by convention if you set the
   `app_file` in the Space's `README.md` metadata header, or rename
   `app/app.py` to a top-level `app.py`. The simplest fix is to add this
   to a top-level `app.py`:
   ```python
   from app.app import main
   main()
   ```
3. In the Space **Settings → Variables and secrets**, add any provider keys
   you want pre-configured (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, …). They
   load via `st.secrets` automatically. **Never commit them.**
4. The first build takes ~5 min while it installs faiss + sentence-transformers.
5. Users without pre-configured keys can still paste their own in the sidebar.

> Deployed URL: _add your HuggingFace Space URL here once deployed._

---

## Screenshots

> Add three screenshots to a `docs/` folder and link them here:
>
> - `docs/chat-with-sources.png` — chat reply with the *Sources used* expander
>   open showing chunk ids and similarity scores.
> - `docs/quiz-in-progress.png` — quiz form with a mix of MCQ / TF / short
>   answer questions.
> - `docs/quiz-results.png` — graded quiz with per-question feedback,
>   expansions, and the *Retry incorrect only* button.

---

## Test plan

### Sample topics

| Domain | Topic | Expected behavior |
|---|---|---|
| Technical | *Teach me gradient descent.* | Picks **worked examples → faded guidance**, includes a numbered study plan with math exercises and a checkpoint. |
| Conceptual | *Explain the Theseus paradox.* | Picks **Feynman technique**, frames practice as writing 1-paragraph explanations. |
| Language | *Help me start learning Italian.* | Picks **immersion + comprehensible input** + spaced repetition for vocabulary, with weekly milestones. |

### Provider parity

Run the same prompt through two providers and confirm the structure
matches:

- **Groq** (`llama-3.3-70b-versatile`) — fast smoke test.
- **Anthropic Claude** (`claude-sonnet-4-6`) — quality baseline.

Both should produce the same five Markdown sections and (when grounded)
populate inline `[source#index]` citations.

### Quiz round-trip

1. Upload a small PDF.
2. Generate a 5-question, **mixed**, **medium** quiz with grounding ON.
3. Answer all questions (mix correct + wrong).
4. Submit → confirm score banner, per-question feedback, and `source_refs`
   for at least the MCQs.
5. Click **Retry incorrect only** → confirm only the wrong questions are
   shown again.
6. Click **New quiz** → confirm session state is cleared.

---

## Known limitations

- **In-memory FAISS** is rebuilt on every Streamlit cold start. Fine for
  ≤ ~100 MB of documents; for larger corpora persist the index to disk.
- **Single-document session** — there's no cross-session knowledge base; each
  user's uploads stay in their own `st.session_state`.
- **Citation fidelity** depends on the underlying LLM. Smaller models
  (HuggingFace free tier, small Ollama models) sometimes hallucinate chunk
  ids — prefer Groq Llama-3.3-70B or Claude for reliable grounding.
- **Short-answer grading** uses substring + canonical-form matching, not
  semantic similarity. Phrase your `correct_answer` tightly when writing
  custom quizzes.
- **No streaming** of responses yet — the chat waits for the full reply
  before rendering.
- **Ollama provider** assumes the Ollama daemon is reachable; the Space
  deployment cannot reach `localhost:11434`, so Ollama is local-dev only.

---

## License

MIT — see `LICENSE` if provided, otherwise treat as MIT.
