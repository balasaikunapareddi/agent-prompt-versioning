"""File-backed version store for system prompts.

Layout on disk (under the store directory)::

    versions/
        v1.txt    the prompt text, exactly as authored
        v1.json   metadata: version_id, parent, author_note,
                  created_at, status
        v2.txt / v2.json
        ...
    changelog.jsonl   one JSON object per decision (promote/rollback)
    runs/             one JSON file per lifecycle-graph run

Statuses: ``draft`` (created, not yet decided), ``active`` (the one
live version), ``superseded`` (was active, replaced by a promotion),
``rolled-back`` (candidate rejected after evals; never went live).

The store is deliberately boring: plain files, no database. A human
can read, diff, and back up the whole history with standard tools.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STATUS_DRAFT = "draft"
STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_ROLLED_BACK = "rolled-back"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class VersionRecord:
    version_id: str
    parent: Optional[str]
    author_note: str
    created_at: str
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


class VersionStore:
    """Create, read, diff, and transition prompt versions on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.runs_dir = self.root / "runs"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.changelog_path = self.root / "changelog.jsonl"

    # --- paths -------------------------------------------------------

    def _text_path(self, version_id: str) -> Path:
        return self.versions_dir / f"{version_id}.txt"

    def _meta_path(self, version_id: str) -> Path:
        return self.versions_dir / f"{version_id}.json"

    # --- CRUD --------------------------------------------------------

    def _next_id(self) -> str:
        existing = self.list_versions()
        n = 1
        ids = {v.version_id for v in existing}
        while f"v{n}" in ids:
            n += 1
        return f"v{n}"

    def create_version(
        self,
        prompt_text: str,
        author_note: str = "",
        parent: Optional[str] = None,
    ) -> VersionRecord:
        """Write a new draft version. Never overwrites an existing one."""
        if parent is not None and not self._meta_path(parent).exists():
            raise ValueError(f"parent version {parent!r} does not exist")
        record = VersionRecord(
            version_id=self._next_id(),
            parent=parent,
            author_note=author_note,
            created_at=_now(),
            status=STATUS_DRAFT,
        )
        self._text_path(record.version_id).write_text(
            prompt_text, encoding="utf-8"
        )
        self._save_meta(record)
        return record

    def _save_meta(self, record: VersionRecord) -> None:
        self._meta_path(record.version_id).write_text(
            json.dumps(record.to_dict(), indent=2), encoding="utf-8"
        )

    def get_record(self, version_id: str) -> VersionRecord:
        path = self._meta_path(version_id)
        if not path.exists():
            raise KeyError(f"unknown version {version_id!r}")
        return VersionRecord(**json.loads(path.read_text(encoding="utf-8")))

    def get_text(self, version_id: str) -> str:
        path = self._text_path(version_id)
        if not path.exists():
            raise KeyError(f"unknown version {version_id!r}")
        return path.read_text(encoding="utf-8")

    def list_versions(self) -> list[VersionRecord]:
        records = []
        for meta in sorted(self.versions_dir.glob("v*.json")):
            records.append(
                VersionRecord(**json.loads(meta.read_text(encoding="utf-8")))
            )
        records.sort(key=lambda r: int(r.version_id[1:]))
        return records

    def get_active(self) -> Optional[VersionRecord]:
        for record in self.list_versions():
            if record.status == STATUS_ACTIVE:
                return record
        return None

    def set_status(self, version_id: str, status: str) -> VersionRecord:
        record = self.get_record(version_id)
        record.status = status
        self._save_meta(record)
        return record

    # --- diff --------------------------------------------------------

    def diff_versions(self, from_id: str, to_id: str) -> str:
        """Unified diff between two versions' prompt text."""
        from_lines = self.get_text(from_id).splitlines(keepends=True)
        to_lines = self.get_text(to_id).splitlines(keepends=True)
        diff = difflib.unified_diff(
            from_lines,
            to_lines,
            fromfile=f"{from_id}.txt",
            tofile=f"{to_id}.txt",
        )
        return "".join(diff)

    # --- changelog & runs --------------------------------------------

    def append_changelog(self, entry: dict) -> dict:
        entry = {"timestamp": _now(), **entry}
        with self.changelog_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    def read_changelog(self) -> list[dict]:
        if not self.changelog_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.changelog_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    def save_run(self, run_id: str, state: dict) -> Path:
        path = self.runs_dir / f"{run_id}.json"
        path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return path
