"""Resumable batch processor with state persistence and budget tracking."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ItemResult:
    """Result of processing a single item."""
    id: str
    status: str  # done, error, needs_review, skipped
    cost: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class BatchResult:
    """Aggregate result of a batch processing run."""
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    total_cost: float = 0.0


@dataclass
class BatchState:
    """Persistent state for a batch processing run."""
    items: dict[str, dict] = field(default_factory=dict)  # id -> {status, ts, cost, ...}
    total_cost: float = 0.0
    batches_run: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# StateStore (SQLite-backed)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    cost REAL DEFAULT 0.0,
    ts TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS run_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class StateStore:
    """Persistent SQLite-backed state with WAL mode for concurrent access.

    Replaces the previous JSON+flock implementation. Same public API.
    If a .json sibling exists when opening a .db, it is auto-migrated once.
    """

    def __init__(self, state_file: Path):
        self._path = Path(state_file)
        json_path = None
        # Accept .json paths for backward compatibility — use .db instead
        if self._path.suffix == ".json":
            json_path = self._path
            self._path = self._path.with_suffix(".db")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), timeout=30)
        self._conn.row_factory = sqlite3.Row
        if str(self._path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        # Initialize run_state defaults
        self._conn.execute(
            "INSERT OR IGNORE INTO run_state (key, value) VALUES ('total_cost', '0.0')")
        self._conn.execute(
            "INSERT OR IGNORE INTO run_state (key, value) VALUES ('batches_run', '0')")
        self._conn.execute(
            "INSERT OR IGNORE INTO run_state (key, value) VALUES ('started_at', ?)",
            (datetime.now().isoformat(),))
        self._conn.commit()
        # Auto-migrate from JSON if a sibling .json file exists and DB is empty
        if json_path is None:
            json_path = self._path.with_suffix(".json")
        if json_path.exists():
            self._migrate_from_json(json_path)

    def _migrate_from_json(self, json_path: Path) -> None:
        """One-time migration: import legacy JSON state into SQLite.

        Handles both 'items' and 'productions' as the items key (kulturperler
        used 'productions' in early versions). After migration the JSON file
        is renamed to .json.migrated so this only runs once.
        """
        existing_count = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        if existing_count > 0:
            # DB already has data — don't overwrite. Just rename the JSON.
            json_path.rename(json_path.with_suffix(".json.migrated"))
            log.info("StateStore: DB already populated, renamed %s", json_path.name)
            return

        raw = json.loads(json_path.read_text())
        items = raw.get("items") or raw.get("productions") or {}

        if not items:
            json_path.rename(json_path.with_suffix(".json.migrated"))
            return

        # Migrate run_state
        for key in ("total_cost", "batches_run", "started_at"):
            if key in raw:
                self._conn.execute(
                    "INSERT OR REPLACE INTO run_state (key, value) VALUES (?, ?)",
                    (key, str(raw[key])))

        # Migrate items
        for item_id, info in items.items():
            status = info.get("status", "unknown")
            cost = info.get("cost", 0.0)
            ts = info.get("ts", "")
            meta = {k: v for k, v in info.items() if k not in ("status", "cost", "ts")}
            self._conn.execute(
                "INSERT OR IGNORE INTO items (item_id, status, cost, ts, metadata) VALUES (?,?,?,?,?)",
                (str(item_id), status, float(cost), ts, json.dumps(meta)))

        self._conn.commit()
        json_path.rename(json_path.with_suffix(".json.migrated"))
        log.info("StateStore: migrated %d items from %s", len(items), json_path.name)

    @property
    def path(self) -> Path:
        return self._path

    def _get_run_val(self, key: str, default: str = "0") -> str:
        row = self._conn.execute(
            "SELECT value FROM run_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def load(self) -> BatchState:
        """Load full state into a BatchState object."""
        state = BatchState()
        state.total_cost = float(self._get_run_val("total_cost", "0.0"))
        state.batches_run = int(self._get_run_val("batches_run", "0"))
        state.started_at = self._get_run_val("started_at", state.started_at)
        for row in self._conn.execute("SELECT * FROM items").fetchall():
            meta = json.loads(row["metadata"])
            state.items[row["item_id"]] = {
                "status": row["status"],
                "cost": row["cost"],
                "ts": row["ts"],
                **meta,
            }
        return state

    def save(self, state: BatchState) -> None:
        """Write a full BatchState back to SQLite."""
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (key, value) VALUES ('total_cost', ?)",
            (str(state.total_cost),))
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (key, value) VALUES ('batches_run', ?)",
            (str(state.batches_run),))
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (key, value) VALUES ('started_at', ?)",
            (state.started_at,))
        # Sync items
        for item_id, info in state.items.items():
            status = info.get("status", "unknown")
            cost = info.get("cost", 0.0)
            ts = info.get("ts", datetime.now().isoformat())
            meta = {k: v for k, v in info.items() if k not in ("status", "cost", "ts")}
            self._conn.execute(
                "INSERT OR REPLACE INTO items (item_id, status, cost, ts, metadata) VALUES (?,?,?,?,?)",
                (item_id, status, cost, ts, json.dumps(meta)))
        self._conn.commit()

    def update_item(self, item_id: str, status: str, **kwargs: Any) -> None:
        """Atomic update for a single item. Concurrent-safe via SQLite WAL.

        If cost= is passed, it is also added to total_cost.
        """
        item_id = str(item_id)
        cost = kwargs.pop("cost", 0.0)
        ts = datetime.now().isoformat()
        meta = json.dumps(kwargs)

        # Merge with existing metadata
        existing = self._conn.execute(
            "SELECT metadata FROM items WHERE item_id = ?", (item_id,)
        ).fetchone()
        if existing:
            merged = json.loads(existing["metadata"])
            merged.update(kwargs)
            meta = json.dumps(merged)

        self._conn.execute(
            "INSERT OR REPLACE INTO items (item_id, status, cost, ts, metadata) VALUES (?,?,?,?,?)",
            (item_id, status, cost, ts, meta))
        if cost:
            self._conn.execute(
                "UPDATE run_state SET value = CAST(CAST(value AS REAL) + ? AS TEXT) WHERE key = 'total_cost'",
                (cost,))
        self._conn.commit()

    def get_pending(self, all_ids: list[str], done_statuses: Optional[set[str]] = None) -> list[str]:
        """Return IDs from all_ids that have not been processed."""
        if done_statuses is None:
            done_statuses = {"done", "verified", "applied", "skipped"}
        ph = ",".join("?" * len(done_statuses))
        done_ids = {
            row[0] for row in self._conn.execute(
                f"SELECT item_id FROM items WHERE status IN ({ph})",
                list(done_statuses),
            ).fetchall()
        }
        return [id_ for id_ in all_ids if str(id_) not in done_ids]

    def get_status_counts(self) -> dict[str, int]:
        """Summary of item statuses."""
        counts: dict[str, int] = {}
        for row in self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM items GROUP BY status"
        ).fetchall():
            counts[row["status"]] = row["cnt"]
        return counts


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """Run a user-provided function over items in batches with budget tracking.

    Args:
        state_store: Persistent state backend.
        max_cost: Stop processing when cumulative cost reaches this limit (None = unlimited).
        batch_size: Number of items per batch.
    """

    def __init__(
        self,
        state_store: StateStore,
        max_cost: Optional[float] = None,
        batch_size: int = 20,
    ):
        self.store = state_store
        self.max_cost = max_cost
        self.batch_size = batch_size

    def process(
        self,
        items: list[Any],
        process_fn: Callable[[list[Any]], list[ItemResult]],
        id_fn: Callable[[Any], str],
    ) -> BatchResult:
        """Process items in batches, skipping already-completed ones.

        Args:
            items: Full list of items to process.
            process_fn: Called once per batch, returns list of ItemResult.
            id_fn: Extracts a string ID from an item.

        Returns:
            Aggregate BatchResult.
        """
        all_ids = [id_fn(item) for item in items]
        pending_ids = set(self.store.get_pending(all_ids))
        pending_items = [item for item in items if id_fn(item) in pending_ids]

        result = BatchResult()
        result.skipped = len(items) - len(pending_items)
        state = self.store.load()
        cumulative_cost = state.total_cost

        # Split into batches
        batches = [
            pending_items[i : i + self.batch_size]
            for i in range(0, len(pending_items), self.batch_size)
        ]

        total_items = len(pending_items)
        processed_so_far = 0
        start_time = time.monotonic()

        for batch_idx, batch in enumerate(batches):
            # Budget check
            if self.max_cost is not None and cumulative_cost >= self.max_cost:
                log.info("Budget limit $%.2f reached (spent $%.2f). Stopping.", self.max_cost, cumulative_cost)
                break

            # Budget warning at 80%
            if self.max_cost is not None and cumulative_cost >= self.max_cost * 0.8:
                remaining = self.max_cost - cumulative_cost
                log.warning("Budget 80%% consumed. $%.2f remaining.", remaining)

            log.info(
                "[Batch %d/%d] %d items (%.1f%% complete, $%.2f spent)",
                batch_idx + 1,
                len(batches),
                len(batch),
                (processed_so_far / total_items * 100) if total_items else 0,
                cumulative_cost,
            )

            try:
                item_results = process_fn(batch)
            except Exception as e:
                log.error("Batch %d failed: %s", batch_idx + 1, e)
                for item in batch:
                    item_id = id_fn(item)
                    self.store.update_item(item_id, "error", error=str(e))
                    result.errors += 1
                continue

            batch_cost = 0.0
            budget_hit = False
            for ir in item_results:
                # Check budget before committing this item's cost
                if self.max_cost is not None and (cumulative_cost + batch_cost + ir.cost) > self.max_cost:
                    log.info("Budget limit $%.2f would be exceeded by item %s (cost $%.2f). Stopping.",
                             self.max_cost, ir.id, ir.cost)
                    budget_hit = True
                    break
                self.store.update_item(ir.id, ir.status, cost=ir.cost, **ir.metadata)
                batch_cost += ir.cost
                if ir.status == "error":
                    result.errors += 1
                else:
                    result.processed += 1

            cumulative_cost += batch_cost
            result.total_cost += batch_cost
            processed_so_far += len(batch)

            if budget_hit:
                break

            # Update aggregate state
            state = self.store.load()
            state.batches_run += 1
            state.total_cost = cumulative_cost
            self.store.save(state)

            # ETA
            elapsed = time.monotonic() - start_time
            if processed_so_far > 0:
                remaining_items = total_items - processed_so_far
                eta_seconds = elapsed / processed_so_far * remaining_items
                log.info(
                    "  Cost: $%.3f | ETA: %.0f min remaining",
                    batch_cost,
                    eta_seconds / 60,
                )

        return result
