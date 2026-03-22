"""Multi-tier verification orchestrator with auto-escalation and adaptive timeouts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .batch import BatchProcessor, ItemResult, StateStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Outcome of verifying a single item."""
    item_id: str
    status: str  # verified, flagged, error
    confidence: float = 0.0
    findings: list[str] = field(default_factory=list)
    cost: float = 0.0
    tier: str = ""
    metadata: dict = field(default_factory=dict)

    def to_item_result(self) -> ItemResult:
        """Convert to a BatchProcessor-compatible ItemResult."""
        effective_status = {
            "verified": "done",
            "flagged": "needs_review",
            "error": "error",
        }.get(self.status, self.status)
        return ItemResult(
            id=self.item_id,
            status=effective_status,
            cost=self.cost,
            metadata={
                "confidence": self.confidence,
                "findings": self.findings,
                "tier": self.tier,
                **self.metadata,
            },
        )


@dataclass
class VerificationTier:
    """A single verification tier (e.g., fast triage or deep verify).

    Args:
        name: Unique tier name.
        process_fn: Called with a list of items, returns list of VerificationResult.
        cost_estimate: Estimated cost per item (for budget planning).
        description: Human-readable description of this tier.
    """
    name: str
    process_fn: Callable[[list[Any]], list[VerificationResult]]
    cost_estimate: float = 0.0
    description: str = ""


@dataclass
class OrchestratorStatus:
    """Progress across all verification tiers."""
    tier_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    total_cost: float = 0.0
    remaining_items: int = 0

    def summary(self) -> str:
        """Human-readable status summary."""
        lines = []
        for tier_name, counts in self.tier_counts.items():
            parts = [f"{k}={v}" for k, v in sorted(counts.items())]
            lines.append(f"  {tier_name}: {', '.join(parts)}")
        lines.append(f"  Total cost: ${self.total_cost:.2f}")
        lines.append(f"  Remaining: {self.remaining_items}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TieredOrchestrator
# ---------------------------------------------------------------------------

class TieredOrchestrator:
    """Run items through multiple verification tiers with optional auto-escalation.

    Args:
        tiers: Ordered list of verification tiers (tier 0 = fastest/cheapest).
        state_store: Persistent state backend.
    """

    def __init__(self, tiers: list[VerificationTier], state_store: StateStore):
        self.tiers = {t.name: t for t in tiers}
        self._tier_order = [t.name for t in tiers]
        self.store = state_store

    def run(
        self,
        items: list[Any],
        id_fn: Callable[[Any], str],
        tier_name: Optional[str] = None,
        max_cost: Optional[float] = None,
        batch_size: int = 20,
        escalate: bool = False,
        escalation_filter: Optional[Callable[[dict], bool]] = None,
    ) -> dict[str, list[VerificationResult]]:
        """Run verification on items through one or more tiers.

        Args:
            items: Items to verify.
            id_fn: Extracts string ID from an item.
            tier_name: Run only this tier. If None and escalate=True, runs all tiers.
            max_cost: Total budget across all tiers.
            batch_size: Items per batch.
            escalate: If True, items flagged in tier N are passed to tier N+1.
            escalation_filter: Predicate on item state dict; if True, item escalates.
                Defaults to checking status == "needs_review".

        Returns:
            Dict mapping tier name to list of VerificationResult.
        """
        if escalation_filter is None:
            escalation_filter = lambda info: info.get("status") == "needs_review"

        tier_names = [tier_name] if tier_name else self._tier_order
        all_results: dict[str, list[VerificationResult]] = {}
        current_items = list(items)
        cumulative_cost = self.store.load().total_cost

        for tname in tier_names:
            if tname not in self.tiers:
                log.error("Unknown tier: %s", tname)
                continue

            tier = self.tiers[tname]
            log.info("=== Tier: %s (%s) — %d items ===", tname, tier.description, len(current_items))

            if not current_items:
                log.info("No items to process in tier %s.", tname)
                break

            remaining_budget = None
            if max_cost is not None:
                remaining_budget = max(0.0, max_cost - cumulative_cost)
                if remaining_budget <= 0:
                    log.info("Budget exhausted before tier %s.", tname)
                    break

            # Wrap the tier's process_fn to produce ItemResults for BatchProcessor
            def _make_process_fn(t: VerificationTier):
                def _process(batch: list[Any]) -> list[ItemResult]:
                    vresults = t.process_fn(batch)
                    # Tag each result with the tier name
                    for vr in vresults:
                        vr.tier = t.name
                    tier_results = all_results.setdefault(t.name, [])
                    tier_results.extend(vresults)
                    return [vr.to_item_result() for vr in vresults]
                return _process

            processor = BatchProcessor(
                state_store=self.store,
                max_cost=remaining_budget,
                batch_size=batch_size,
            )
            batch_result = processor.process(current_items, _make_process_fn(tier), id_fn)
            cumulative_cost += batch_result.total_cost

            # Auto-escalation: collect flagged items for the next tier
            if escalate and tname != tier_names[-1]:
                state = self.store.load()
                flagged_ids = set()
                for item_id, info in state.items.items():
                    if info.get("tier") == tname and escalation_filter(info):
                        flagged_ids.add(item_id)

                current_items = [item for item in items if id_fn(item) in flagged_ids]
                if current_items:
                    log.info("%d items escalated from %s to next tier.", len(current_items), tname)
                    # Reset their status so the next tier picks them up
                    for item in current_items:
                        self.store.update_item(id_fn(item), "pending", escalated_from=tname)
            else:
                current_items = []

        return all_results

    def status(self, all_ids: Optional[list[str]] = None) -> OrchestratorStatus:
        """Progress summary across all tiers."""
        state = self.store.load()
        tier_counts: dict[str, dict[str, int]] = {}

        for item_id, info in state.items.items():
            tier = info.get("tier", "unknown")
            status = info.get("status", "unknown")
            if tier not in tier_counts:
                tier_counts[tier] = {}
            tier_counts[tier][status] = tier_counts[tier].get(status, 0) + 1

        remaining = 0
        if all_ids is not None:
            pending = self.store.get_pending(all_ids)
            remaining = len(pending)

        return OrchestratorStatus(
            tier_counts=tier_counts,
            total_cost=state.total_cost,
            remaining_items=remaining,
        )


# ---------------------------------------------------------------------------
# Adaptive timeouts
# ---------------------------------------------------------------------------

def timeout_for(
    item: Any,
    base_timeout: int = 30,
    scale_fn: Optional[Callable[[Any], float]] = None,
    max_timeout: int = 1800,
) -> int:
    """Compute a timeout scaled by item complexity.

    Args:
        item: The item to compute timeout for.
        base_timeout: Minimum timeout in seconds.
        scale_fn: Returns a multiplier (>= 1.0) for the item. Defaults to 1.0.
        max_timeout: Upper bound on timeout.

    Returns:
        Timeout in seconds.
    """
    multiplier = scale_fn(item) if scale_fn else 1.0
    scaled = max(base_timeout, int(base_timeout * multiplier))
    return min(scaled, max_timeout)
