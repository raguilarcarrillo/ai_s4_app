"""Opt-in session persistence for Smart Teacher.

Two independent disk caches live under ``SMART_TEACHER_CACHE_DIR``
(``./.cache/`` by default):

* ``faiss/<content_fingerprint>/`` — Shared FAISS index, content-keyed.
  Any session whose uploaded files hash to the same fingerprint reuses
  this entry; no per-user information is stored here.
* ``sessions/<sha256(token)>/`` — Per-session blob containing the raw
  uploaded file bytes and a small ``manifest.json``. The on-disk
  directory name is the *hash* of the user's token, so a filesystem
  listing alone doesn't reveal active tokens — only the user holds the
  raw token they need to restore.

Both layers are written only when the user explicitly opts in via the
sidebar toggle. Without the opt-in nothing in this module is invoked
and no files leave RAM.

Environment knobs:

* ``SMART_TEACHER_CACHE_DIR`` — base directory (default ``./.cache``).
* ``SMART_TEACHER_SESSION_TTL_DAYS`` — days of inactivity before a
  session is auto-deleted (default 7).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache")
DEFAULT_TTL_DAYS = 7
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")


class PersistenceError(RuntimeError):
    """Raised when a persistence operation can't proceed safely."""


def cache_root() -> Path:
    """Resolved base cache directory (env override > default)."""
    raw = os.environ.get("SMART_TEACHER_CACHE_DIR", "").strip()
    return Path(raw) if raw else DEFAULT_CACHE_DIR


def faiss_dir() -> Path:
    return cache_root() / "faiss"


def sessions_dir() -> Path:
    return cache_root() / "sessions"


def _ttl_days() -> int:
    raw = os.environ.get("SMART_TEACHER_SESSION_TTL_DAYS", "").strip()
    if not raw:
        return DEFAULT_TTL_DAYS
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid SMART_TEACHER_SESSION_TTL_DAYS=%r; using default %d",
            raw,
            DEFAULT_TTL_DAYS,
        )
        return DEFAULT_TTL_DAYS


def new_token() -> str:
    """Mint a fresh 32-char hex session token."""
    return uuid.uuid4().hex


def _validate_token(token: str) -> str:
    cleaned = (token or "").strip().lower()
    if not _TOKEN_RE.match(cleaned):
        raise PersistenceError(
            "Token must be 32 hexadecimal characters (a fresh one is shown "
            "in the sidebar when persistence is enabled)."
        )
    return cleaned


def _hash_token(token: str) -> str:
    return hashlib.sha256(_validate_token(token).encode("utf-8")).hexdigest()


def _safe_name(name: str) -> str:
    safe = Path(name).name
    if not safe or safe in {".", ".."}:
        raise PersistenceError(f"Refusing unsafe filename: {name!r}")
    return safe


def _session_dir(token: str) -> Path:
    return sessions_dir() / _hash_token(token)


@dataclass
class SessionManifest:
    """Metadata for a persisted session.

    Raw file bytes are stored alongside (in ``files/``); this struct only
    captures what the app needs to reconstruct the indexing pipeline.
    """

    filenames: list[str]
    fingerprint: str
    chunk_size: int
    chunk_overlap: int
    embed_backend: str
    embed_model: str
    created_at: str
    last_accessed_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionManifest":
        return cls(
            filenames=list(data.get("filenames", [])),
            fingerprint=str(data.get("fingerprint", "")),
            chunk_size=int(data.get("chunk_size", 0)),
            chunk_overlap=int(data.get("chunk_overlap", 0)),
            embed_backend=str(data.get("embed_backend", "")),
            embed_model=str(data.get("embed_model", "")),
            created_at=str(data.get("created_at", "")),
            last_accessed_at=str(data.get("last_accessed_at", "")),
        )


def save_session(
    token: str,
    files: Iterable[tuple[str, bytes]],
    *,
    fingerprint: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_backend: str,
    embed_model: str,
) -> None:
    """Persist files + manifest for ``token``.

    Args:
        token: The user-facing session token. The on-disk directory is
            ``sha256(token)`` so the raw value never lands on disk.
        files: Iterable of ``(filename, bytes)`` pairs. Filenames are
            sanitized to basenames.
        fingerprint: Content fingerprint of the files + chunk params.
        chunk_size: Splitter chunk size at the time of save.
        chunk_overlap: Splitter chunk overlap at the time of save.
        embed_backend: Embeddings backend id used to build the index.
        embed_model: Embeddings model id used to build the index.

    Raises:
        PersistenceError: If a filename is unsafe or IO fails.
    """
    target = _session_dir(token)
    files_subdir = target / "files"
    files_subdir.mkdir(parents=True, exist_ok=True)
    # Wipe stale file bytes so a re-save reflects the current set.
    for existing in files_subdir.iterdir():
        if existing.is_file():
            existing.unlink()
    written: list[str] = []
    for raw_name, payload in files:
        safe = _safe_name(raw_name)
        (files_subdir / safe).write_bytes(payload)
        written.append(safe)
    now = datetime.now(tz=timezone.utc).isoformat()
    manifest = SessionManifest(
        filenames=written,
        fingerprint=fingerprint,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_backend=embed_backend,
        embed_model=embed_model,
        created_at=now,
        last_accessed_at=now,
    )
    (target / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )


def load_session(token: str) -> tuple[SessionManifest, list[Path]]:
    """Load a persisted session.

    Updates ``last_accessed_at`` so the TTL is "days since last use",
    not "days since creation".

    Args:
        token: The user-facing token.

    Returns:
        Tuple of (manifest, ordered list of absolute file paths).

    Raises:
        PersistenceError: If the token is invalid or no session exists.
    """
    target = _session_dir(token)
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        raise PersistenceError(
            "No persisted session found for this token. The token may be "
            "wrong, or the session may have expired and been cleaned up."
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PersistenceError(f"Could not read session manifest: {e}") from e
    manifest = SessionManifest.from_dict(data)
    files_subdir = target / "files"
    paths: list[Path] = []
    for name in manifest.filenames:
        try:
            safe = _safe_name(name)
        except PersistenceError:
            logger.warning("Skipping unsafe filename in manifest: %r", name)
            continue
        candidate = (files_subdir / safe).resolve()
        if not candidate.exists():
            logger.warning("Manifest references missing file: %s", safe)
            continue
        paths.append(candidate)
    # Touch last_accessed_at so active sessions don't get reaped.
    manifest.last_accessed_at = datetime.now(tz=timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )
    return manifest, paths


def delete_session(token: str) -> bool:
    """Remove a persisted session. Returns True if something was deleted."""
    target = _session_dir(token)
    if not target.exists():
        return False
    shutil.rmtree(target, ignore_errors=True)
    return True


def save_faiss(fingerprint: str, vector_store: Any) -> None:
    """Persist a FAISS index keyed by content fingerprint."""
    target = faiss_dir() / fingerprint
    target.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target))


def load_faiss(fingerprint: str, embeddings: Any) -> Optional[Any]:
    """Load a FAISS index for ``fingerprint``, or ``None`` if not cached."""
    target = faiss_dir() / fingerprint
    if not (target / "index.faiss").exists():
        return None
    from langchain_community.vectorstores import FAISS

    # We only ever load files we wrote ourselves under our own cache dir,
    # so allow_dangerous_deserialization is safe here.
    return FAISS.load_local(
        str(target),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def cleanup_expired() -> int:
    """Delete sessions older than the TTL. Returns the count removed.

    Best-effort: malformed / unreadable session directories are also
    removed so the cache self-heals.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=_ttl_days())
    removed = 0
    root = sessions_dir()
    if not root.exists():
        return 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.exists():
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            last = datetime.fromisoformat(data.get("last_accessed_at", ""))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if last < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except (ValueError, OSError, TypeError):
            logger.warning("Removing unreadable session dir %s", entry)
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed
