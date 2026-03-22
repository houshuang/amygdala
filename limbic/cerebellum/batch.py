"""Resumable batch processor with state persistence and budget tracking."""

from __future__ import annotations

import fcntl
import json
import logging
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
# StateStore
# ---------------------------------------------------------------------------

class StateStore:
    """Persistent JSON state with atomic writes and file locking."""

    def __init__(self, state_file: Path):
        self._path = Path(state_file)
        self._lock_path = self._path.with_suffix(".lock")

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> BatchState:
        """Load state from disk, or return fresh state if missing."""
        if self._path.exists():
            with open(self._path) as f:
                raw = json.load(f)
            state = BatchState()
            state.items = raw.get("items", {})
            state.total_cost = raw.get("total_cost", 0.0)
            state.batches_run = raw.get("batches_run", 0)
            state.started_at = raw.get("started_at", state.started_at)
            return state
        return BatchState()

    def save(self, state: BatchState) -> None:
        """Atomic write: write to tmp file then rename."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "items": state.items,
                    "total_cost": state.total_cost,
                    "batches_run": state.batches_run,
                    "started_at": state.started_at,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        tmp.rename(self._path)

    def update_item(self, item_id: str, status: str, **kwargs: Any) -> None:
        """Thread-safe update for a single item."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.touch(exist_ok=True)
        with open(self._lock_path) as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            state = self.load()
            state.items[str(item_id)] = {
                "status": status,
                "ts": datetime.now().isoformat(),
                **kwargs,
            }
            self.save(state)

    def get_pending(self, all_ids: list[str], done_statuses: Optional[set[str]] = None) -> list[str]:
        """Return IDs from all_ids that have not been processed."""
        if done_statuses is None:
            done_statuses = {"done", "verified", "applied", "skipped"}
        state = self.load()
        return [
            id_ for id_ in all_ids
            if state.items.get(str(id_), {}).get("status") not in done_statuses
        ]

    def get_status_counts(self) -> dict[str, int]:
        """Summary of item statuses."""
        state = self.load()
        counts: dict[str, int] = {}
        for info in state.items.values():
            s = info.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
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
            for ir in item_results:
                self.store.update_item(ir.id, ir.status, cost=ir.cost, **ir.metadata)
                batch_cost += ir.cost
                if ir.status == "error":
                    result.errors += 1
                else:
                    result.processed += 1

            cumulative_cost += batch_cost
            result.total_cost += batch_cost
            processed_so_far += len(batch)

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
