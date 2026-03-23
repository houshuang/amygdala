"""YAML-backed data store with file locking and atomic writes.

Provides a thin, typed layer over a directory tree of YAML files. Each entity
type maps to a subdirectory; each entity is a single YAML file named by its ID.
"""

from __future__ import annotations

import fcntl
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml


# ---------------------------------------------------------------------------
# YAML formatting
# ---------------------------------------------------------------------------

class _YAMLDumper(yaml.SafeDumper):
    pass

def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

_YAMLDumper.add_representer(str, _str_representer)


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------

@contextmanager
def _file_lock(path: Path, lock_dir: Path | None = None):
    """Advisory exclusive lock for a file."""
    ldir = lock_dir or path.parent
    ldir.mkdir(parents=True, exist_ok=True)
    lock_path = ldir / f".{path.stem}.lock"
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        raise RuntimeError(f"File {path} is locked by another process")
    try:
        yield
    finally:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# YAML Store
# ---------------------------------------------------------------------------

class YAMLStore:
    """Directory-based YAML entity store.

    Args:
        base_dir: Root directory containing entity subdirectories.
        schema: Maps entity_type names to subdirectory names.
                 e.g. {"person": "persons", "work": "plays"}
    """

    def __init__(self, base_dir: str | Path, schema: dict[str, str]) -> None:
        self.base = Path(base_dir)
        self.schema = dict(schema)

    def _dir(self, entity_type: str) -> Path:
        subdir = self.schema.get(entity_type)
        if subdir is None:
            raise ValueError(f"Unknown entity type: {entity_type}")
        return self.base / subdir

    def _path(self, entity_type: str, entity_id: str) -> Path:
        return self._dir(entity_type) / f"{entity_id}.yaml"

    def load(self, entity_type: str, entity_id: str) -> dict[str, Any] | None:
        """Load a single entity. Returns None if not found."""
        path = self._path(entity_type, entity_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def save(self, entity_type: str, entity_id: str, data: dict[str, Any]) -> None:
        """Atomic write with advisory file lock."""
        path = self._path(entity_type, entity_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(path):
            tmp = path.with_suffix(".yaml.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                yaml.dump(data, fh, Dumper=_YAMLDumper, allow_unicode=True,
                          default_flow_style=False, sort_keys=False)
            tmp.replace(path)

    def delete(self, entity_type: str, entity_id: str) -> bool:
        """Delete an entity file. Returns True if it existed."""
        path = self._path(entity_type, entity_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def iter_type(self, entity_type: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Iterate over all entities of a type, yielding (id, data)."""
        d = self._dir(entity_type)
        if not d.exists():
            return
        for path in sorted(d.glob("*.yaml")):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                yield path.stem, data
            except Exception:
                continue

    def all_ids(self, entity_type: str) -> set[str]:
        """Return all entity IDs of a given type."""
        d = self._dir(entity_type)
        if not d.exists():
            return set()
        return {p.stem for p in d.glob("*.yaml")}

    def backup(self, entity_type: str, entity_id: str, backup_dir: str | Path | None = None) -> Path:
        """Create a timestamped backup of an entity file."""
        path = self._path(entity_type, entity_id)
        if not path.exists():
            raise FileNotFoundError(f"{entity_type}/{entity_id} does not exist")
        bdir = Path(backup_dir) if backup_dir else self.base / ".backups"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = bdir / timestamp / self.schema[entity_type] / f"{entity_id}.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        return dest
