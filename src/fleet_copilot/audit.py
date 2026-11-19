"""Append-only, hash-chained audit log.

Every security-relevant event (input guard verdicts, plans, tool calls,
policy decisions, approval requests/decisions, executions, final answers) is
written as one JSON line. Each entry stores the SHA-256 of the previous entry,
so editing, reordering or deleting a line breaks the chain and ``verify()``
reports the first broken sequence number.

A hash chain alone cannot detect lines cut off the *end*. So after every append
the latest ``(seq, hash)`` is written to a head file next to the log, and
``verify()`` checks that the log still reaches that head. Someone with write
access to both files can still rewrite history; production would anchor the
head outside the host (WORM bucket, signed checkpoints or a transparency log).
That limitation is documented in the README.

Opening a log that fails verification raises ``AuditIntegrityError`` (fail
closed: no new events are appended on top of a tampered history) unless
``strict=False``, which the ``audit verify`` command uses to report the damage.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:   # POSIX advisory lock so the CLI and the UI can append to the same log safely
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

GENESIS = "0" * 64


class AuditIntegrityError(RuntimeError):
    pass


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class AuditLog:
    def __init__(self, path: str | Path | None = None, strict: bool = True):
        """``path=None`` keeps the log in memory (used by tests and evals)."""
        self.path = Path(path) if path else None
        self.head_path = self.path.with_name(self.path.name + ".head") if self.path else None
        self._lock = threading.Lock()
        self._entries: list[dict] = []
        self._size = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._reload()
            if strict:
                ok, bad = self.verify()
                if not ok:
                    raise AuditIntegrityError(f"audit log {self.path} failed verification at seq {bad}; "
                                              "refusing to append. Run `fleet-copilot audit verify`.")

    def _reload(self) -> None:
        if self.path and self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                self._entries = [json.loads(l) for l in fh if l.strip()]
            self._size = self.path.stat().st_size

    @contextlib.contextmanager
    def _file_lock(self):
        if not self.path or fcntl is None:
            yield
            return
        with open(self.path.with_name(self.path.name + ".lock"), "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _sync(self) -> None:
        """Another process (CLI and UI side by side) may have appended: re-read before extending the chain."""
        if self.path and self.path.exists() and self.path.stat().st_size != self._size:
            self._reload()
            ok, bad = self.verify()
            if not ok:
                raise AuditIntegrityError(f"audit log {self.path} failed verification at seq {bad}; refusing to append.")

    @property
    def entries(self) -> list[dict]:
        with self._lock:
            self._sync()
        return list(self._entries)

    def record(self, event: str, *, tenant_id: str, actor: str, thread_id: str | None = None, **data: Any) -> dict:
        with self._lock, self._file_lock():
            self._sync()
            prev = self._entries[-1]["hash"] if self._entries else GENESIS
            body = {
                "seq": len(self._entries) + 1,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "event": event,
                "tenant_id": tenant_id,
                "actor": actor,
                "thread_id": thread_id,
                "data": data,
                "prev_hash": prev,
            }
            body["hash"] = hashlib.sha256(_canonical(body).encode()).hexdigest()
            self._entries.append(body)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(_canonical(body) + "\n")
                self._size = self.path.stat().st_size
                tmp = self.head_path.with_name(self.head_path.name + ".tmp")
                tmp.write_text(json.dumps({"seq": body["seq"], "hash": body["hash"]}))
                tmp.replace(self.head_path)   # atomic
            return body

    def verify(self) -> tuple[bool, int | None]:
        """Return (ok, first_bad_seq)."""
        prev = GENESIS
        for e in self._entries:
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("prev_hash") != prev:
                return False, e.get("seq")
            if hashlib.sha256(_canonical(body).encode()).hexdigest() != e.get("hash"):
                return False, e.get("seq")
            prev = e["hash"]
        if self.head_path is not None and (self._entries or self.head_path.exists()):
            try:
                head = json.loads(self.head_path.read_text())
            except (OSError, ValueError):
                return False, (self._entries[-1]["seq"] if self._entries else 0)   # head missing/corrupt
            last = self._entries[-1] if self._entries else {"seq": 0, "hash": GENESIS}
            if head.get("seq") != last["seq"] or head.get("hash") != last["hash"]:
                return False, min(int(head.get("seq") or 0), last["seq"]) + 1      # truncated (or head rewritten)
        return True, None

    def events(self, event: str | None = None, thread_id: str | None = None) -> list[dict]:
        return [e for e in self._entries
                if (event is None or e["event"] == event) and (thread_id is None or e["thread_id"] == thread_id)]
