"""Retrieval-Augmented Generation pipeline for Smart Teacher.

Responsibilities:
    * Load uploaded documents (PDF, TXT, MD).
    * Split them into overlapping chunks.
    * Embed and index chunks in a FAISS in-memory vector store.
    * Retrieve top-k chunks with similarity scores and stable citation ids.

The module is intentionally small: it owns IO and indexing; the LLM and
prompt orchestration live elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".ipynb"}


@dataclass
class RetrievedChunk:
    """A chunk surfaced by the retriever.

    Attributes:
        chunk_id: Stable identifier of the form ``<source>#<index>`` used as a
            citation reference.
        source: Source filename or label.
        content: Text of the chunk.
        score: Similarity score (higher is better). May be ``None`` if the
            backend doesn't expose one.
    """

    chunk_id: str
    source: str
    content: str
    score: Optional[float]


def _read_file_to_documents(path: Path, source_label: str) -> List[Document]:
    """Load a single file into one or more LangChain Documents.

    Args:
        path: Path to the file on disk.
        source_label: Original filename used as the ``source`` metadata.

    Returns:
        Loaded ``Document`` objects (one per page for PDFs, one per text
        file, one per notebook).

    Raises:
        ValueError: If the file extension is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path))
        docs = loader.load()
    elif suffix in {".txt", ".md", ".markdown"}:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    elif suffix == ".ipynb":
        docs = _load_notebook(path, source_label)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    for d in docs:
        d.metadata["source"] = source_label
    return docs


def _load_notebook(path: Path, source_label: str) -> List[Document]:
    """Parse a Jupyter notebook into a single Document.

    Only ``markdown`` and ``code`` cells contribute to the output; ``raw``
    cells and *all* cell outputs are skipped on purpose:

    * Notebook outputs can include arbitrary ``text/html`` /
      ``application/javascript`` payloads and multi-MB base64-encoded
      images. We don't render anything as raw HTML today, but skipping
      outputs is defense-in-depth — if a future surface ever does, ipynb
      outputs won't suddenly become an XSS or noise vector.
    * Outputs frequently echo printed secrets (API tokens, env vars) the
      author didn't intend to share. Skipping keeps those out of the
      embedding index.

    The code-fence language is taken from the notebook's
    ``metadata.kernelspec.language`` but only when it matches Python's
    identifier rules — otherwise we fall back to ``python``. This stops a
    crafted notebook from injecting markup via the language string.

    Args:
        path: Path to the ``.ipynb`` file on disk.
        source_label: Original filename, used in the error message and
            attached to the returned Document by the caller.

    Returns:
        A list with a single Document, or an empty list if the notebook
        had no usable cells.

    Raises:
        ValueError: If the file isn't valid JSON or isn't a notebook.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ValueError(f"Could not read notebook {source_label}: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{source_label} is not valid JSON: {e}"
        ) from e
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise ValueError(
            f"{source_label} does not look like a Jupyter notebook"
        )

    kernel = data.get("metadata", {}).get("kernelspec", {})
    raw_lang = kernel.get("language", "") if isinstance(kernel, dict) else ""
    lang = raw_lang if isinstance(raw_lang, str) and raw_lang.isidentifier() else "python"

    parts: list[str] = []
    for cell in data["cells"]:
        if not isinstance(cell, dict):
            continue
        cell_type = cell.get("cell_type")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(s for s in source if isinstance(s, str))
        if not isinstance(source, str):
            continue
        # Strip NULL bytes defensively — some downstream consumers (DBs,
        # certain loaders) refuse them and there's no legitimate reason
        # to embed one in a notebook source cell.
        source = source.replace("\x00", "").strip()
        if not source:
            continue
        if cell_type == "markdown":
            parts.append(source)
        elif cell_type == "code":
            parts.append(f"```{lang}\n{source}\n```")
        # "raw" cells (template / nbconvert directives) are intentionally
        # ignored; "outputs" are not consulted at all.

    if not parts:
        return []
    return [
        Document(
            page_content="\n\n".join(parts),
            metadata={"source": source_label},
        )
    ]


def load_disk_files(paths: Iterable[Path]) -> List[Document]:
    """Load already-on-disk files into LangChain Documents.

    Used by the persistence-restore path, where files are read back from
    ``./.cache/sessions/<id>/files/``. Mirrors :func:`load_uploaded_files`
    minus the tempdir dance.

    Args:
        paths: Iterable of absolute filesystem paths to load.

    Returns:
        Flat list of loaded ``Document`` objects across all inputs.
    """
    documents: List[Document] = []
    for path in paths:
        safe_name = path.name
        if not safe_name or safe_name in {".", ".."}:
            logger.warning("Skipping suspicious filename %r", path)
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            logger.warning("Skipping unsupported file %s", safe_name)
            continue
        try:
            documents.extend(_read_file_to_documents(path, safe_name))
        except Exception:
            logger.exception("Failed to load %s", safe_name)
    return documents


def load_uploaded_files(uploaded_files: Iterable[Any]) -> List[Document]:
    """Load Streamlit ``UploadedFile`` objects into LangChain Documents.

    Each uploaded file is written to a temp path so the underlying loaders
    can read it from disk.

    Args:
        uploaded_files: Iterable of objects exposing ``name`` and
            ``getbuffer()`` (Streamlit's ``UploadedFile`` interface).

    Returns:
        Flat list of loaded ``Document`` objects across all inputs.
    """
    documents: List[Document] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_resolved = Path(tmpdir).resolve()
        for up in uploaded_files:
            raw_name = getattr(up, "name", "uploaded")
            # Strip any path components a hostile client might send. The
            # basename is also what flows into chunk metadata and therefore
            # into LLM prompts, so we want it clean.
            safe_name = Path(raw_name).name
            if not safe_name or safe_name in {".", ".."}:
                logger.warning("Skipping suspicious filename %r", raw_name)
                continue
            suffix = Path(safe_name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                logger.warning("Skipping unsupported file %s", safe_name)
                continue
            tmp_path = (tmpdir_resolved / safe_name).resolve()
            # Defense-in-depth: refuse to write outside the temp directory.
            if tmpdir_resolved not in tmp_path.parents:
                logger.warning("Refusing to write %s outside tempdir", safe_name)
                continue
            tmp_path.write_bytes(up.getbuffer())
            try:
                documents.extend(_read_file_to_documents(tmp_path, safe_name))
            except Exception:
                logger.exception("Failed to load %s", safe_name)
    return documents


def split_documents(
    docs: Sequence[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into overlapping chunks and stamp them with stable IDs.

    Args:
        docs: Documents to split.
        chunk_size: Target chunk length in characters.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        Chunked ``Document`` list. Each chunk's metadata carries:

        * ``source`` — original filename.
        * ``chunk_index`` — sequential int within ``source``.
        * ``chunk_id`` — ``"<source>#<chunk_index>"``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(docs))
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = counters.get(source, 0)
        counters[source] = idx + 1
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{source}#{idx}"
    return chunks


def build_vector_store(chunks: Sequence[Document], embeddings: Any) -> Any:
    """Build a FAISS vector store from chunks.

    Args:
        chunks: Chunked documents to index.
        embeddings: A LangChain ``Embeddings`` instance.

    Returns:
        A FAISS vector store ready for similarity search.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks.")
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(list(chunks), embeddings)


def retrieve(
    vector_store: Any,
    query: str,
    k: int = 4,
) -> List[RetrievedChunk]:
    """Run a similarity search and return enriched chunk objects.

    Args:
        vector_store: A vector store exposing ``similarity_search_with_score``.
        query: Natural-language query.
        k: Number of chunks to retrieve.

    Returns:
        A list of :class:`RetrievedChunk` ordered by descending similarity.
        FAISS returns L2 distance; we convert it to a similarity-style score
        ``1 / (1 + distance)`` so higher = more relevant.
    """
    try:
        pairs: List[Tuple[Document, float]] = (
            vector_store.similarity_search_with_score(query, k=k)
        )
    except Exception:
        logger.exception("similarity_search_with_score failed; falling back")
        docs = vector_store.similarity_search(query, k=k)
        pairs = [(d, 0.0) for d in docs]

    out: List[RetrievedChunk] = []
    for doc, distance in pairs:
        score = 1.0 / (1.0 + float(distance)) if distance is not None else None
        out.append(
            RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                source=doc.metadata.get("source", "unknown"),
                content=doc.page_content,
                score=score,
            )
        )
    return out


def format_context_block(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a context block for the LLM prompt.

    Each chunk is preceded by its ``[chunk_id]`` so the model can cite it
    inline.

    Args:
        chunks: Chunks to format.

    Returns:
        A single string ready to inject into the system/user prompt. Empty if
        no chunks were retrieved.
    """
    if not chunks:
        return ""
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}]\n{c.content.strip()}")
    return "\n\n---\n\n".join(lines)


def fingerprint(files: Iterable[Any], chunk_size: int, chunk_overlap: int) -> str:
    """Compute a deterministic cache key for an index built from ``files``.

    Combines each file's basename + byte content with the splitter
    parameters. Accepts either Streamlit ``UploadedFile`` objects (with
    ``.name`` / ``.getbuffer()``) or :class:`pathlib.Path` instances —
    both produce the same digest for byte-identical content, so a
    persisted session can collide-hit the in-memory and on-disk FAISS
    caches built from the original upload.

    Args:
        files: Iterable of ``UploadedFile``-like objects or ``Path``s.
        chunk_size: Chunk size used by the splitter.
        chunk_overlap: Chunk overlap used by the splitter.

    Returns:
        Hex digest suitable as a cache key.
    """
    h = hashlib.sha256()
    h.update(f"cs={chunk_size};co={chunk_overlap};".encode())
    for f in files:
        if isinstance(f, Path):
            name = Path(f).name
            payload = f.read_bytes()
        else:
            name = Path(getattr(f, "name", "")).name
            payload = bytes(f.getbuffer())
        h.update(name.encode())
        h.update(payload)
    return h.hexdigest()
