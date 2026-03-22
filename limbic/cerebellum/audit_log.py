"""JSONL audit logging with extraction and analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    item_id: str
    action: str
    details: dict = field(default_factory=dict)
    cost: float = 0.0
    tier: str = ""

    def to_dict(self) -> dict:
        return {
            "ts": self.timestamp,
            "item_id": self.item_id,
            "action": self.action,
            "details": self.details,
            "cost": self.cost,
            "tier": self.tier,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> AuditEntry:
        return cls(
            timestamp=raw.get("ts", ""),
            item_id=raw.get("item_id", ""),
            action=raw.get("action", ""),
            details=raw.get("details", {}),
            cost=raw.get("cost", 0.0),
            tier=raw.get("tier", ""),
        )


@dataclass
class LogSummary:
    """Aggregate statistics from audit log entries."""
    total_cost: float = 0.0
    items_processed: int = 0
    error_count: int = 0
    by_tier: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_action: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Append-only JSONL logger with daily rotation by filename.

    Log files are named ``{prefix}_{YYYYMMDD}.jsonl`` inside log_dir.
    """

    def __init__(self, log_dir: Path, prefix: str = "audit"):
        self.log_dir = Path(log_dir)
        self.prefix = prefix

    def _today_path(self) -> Path:
        return self.log_dir / f"{self.prefix}_{date.today().strftime('%Y%m%d')}.jsonl"

    def log(self, entry: dict) -> None:
        """Append a raw dict as a single JSONL line."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self._today_path()
        if "ts" not in entry:
            entry["ts"] = datetime.now().isoformat()
        with open(path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_entry(self, entry: AuditEntry) -> None:
        """Append a typed AuditEntry."""
        self.log(entry.to_dict())


# ---------------------------------------------------------------------------
# Log reading
# ---------------------------------------------------------------------------

def read_logs(
    log_dir: Path,
    prefix: Optional[str] = None,
    since: Optional[str] = None,
) -> Iterator[AuditEntry]:
    """Read JSONL log files and yield AuditEntry objects.

    Args:
        log_dir: Directory containing .jsonl files.
        prefix: Only read files matching ``{prefix}_*.jsonl``. None = all .jsonl.
        since: ISO timestamp string; skip entries older than this.
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return

    pattern = f"{prefix}_*.jsonl" if prefix else "*.jsonl"
    for path in sorted(log_dir.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = raw.get("ts", "")
                if since and ts < since:
                    continue
                yield AuditEntry.from_dict(raw)


# ---------------------------------------------------------------------------
# Extraction and analysis
# ---------------------------------------------------------------------------

def extract_operations(
    entries: list[AuditEntry],
    op_types: Optional[list[str]] = None,
    dedup_key_fn: Optional[Callable[[dict], Any]] = None,
) -> dict[str, list[dict]]:
    """Group operations from audit entries by type, with optional dedup.

    Each entry's ``details`` dict is expected to contain an ``operations``
    list (or any nested list of dicts with a ``type`` field).  If not present,
    the entry itself is treated as a single operation.

    Args:
        entries: AuditEntry objects to scan.
        op_types: Only include these operation types. None = all.
        dedup_key_fn: Given an operation dict, returns a hashable key.
            When two operations share a key, the latest (by timestamp) wins.

    Returns:
        Dict mapping operation type to list of operation dicts.
    """
    raw_ops: list[tuple[str, dict]] = []  # (timestamp, op_dict)

    for entry in entries:
        ops = entry.details.get("operations", [])
        if not ops:
            # Treat the entry itself as an operation
            op = {
                "type": entry.action,
                "item_id": entry.item_id,
                **entry.details,
                "_ts": entry.timestamp,
            }
            ops = [op]
        else:
            # Attach timestamp for dedup
            for op in ops:
                op.setdefault("_ts", entry.timestamp)

        for op in ops:
            op_type = op.get("type", "unknown")
            if op_types and op_type not in op_types:
                continue
            raw_ops.append((entry.timestamp, op))

    # Dedup
    if dedup_key_fn:
        seen: dict[Any, dict] = {}
        for ts, op in raw_ops:
            key = dedup_key_fn(op)
            existing = seen.get(key)
            if existing is None or op.get("_ts", "") >= existing.get("_ts", ""):
                seen[key] = op
        deduped = list(seen.values())
    else:
        deduped = [op for _, op in raw_ops]

    # Group by type
    grouped: dict[str, list[dict]] = {}
    for op in deduped:
        op_type = op.get("type", "unknown")
        grouped.setdefault(op_type, []).append(op)

    return grouped


def summarize_logs(entries: list[AuditEntry]) -> LogSummary:
    """Compute aggregate statistics from audit entries."""
    summary = LogSummary()
    seen_items: set[str] = set()

    for entry in entries:
        summary.total_cost += entry.cost
        seen_items.add(entry.item_id)

        if entry.action == "error" or entry.details.get("status") == "error":
            summary.error_count += 1

        summary.by_action[entry.action] = summary.by_action.get(entry.action, 0) + 1

        tier = entry.tier or "unknown"
        if tier not in summary.by_tier:
            summary.by_tier[tier] = {"count": 0, "cost": 0.0}
        summary.by_tier[tier]["count"] += 1
        summary.by_tier[tier]["cost"] += entry.cost

    summary.items_processed = len(seen_items)
    return summary
