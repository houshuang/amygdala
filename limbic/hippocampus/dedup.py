"""Entity deduplication with composable veto-gate filtering.

Candidate pairs (produced externally, e.g. by fuzzy name matching or embedding
distance) pass through a chain of veto gates. Any gate can reject a pair. This
design keeps false-positive control explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class CandidatePair:
    """A candidate duplicate pair with their data and similarity score."""
    id_a: str
    id_b: str
    fields_a: dict[str, Any]
    fields_b: dict[str, Any]
    score: float = 0.0


@dataclass
class VetoGate:
    """A single filter that can reject a candidate pair.

    check_fn receives (fields_a, fields_b) and returns (accepted, reason).
    If accepted is False, the pair is rejected with the given reason.
    """
    name: str
    check_fn: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str]]


class ExclusionList:
    """Known false-positive pairs that should never be merged."""

    def __init__(self, pairs: set[frozenset[str]] | None = None) -> None:
        self._pairs: set[frozenset[str]] = pairs or set()

    def contains(self, id_a: str, id_b: str) -> bool:
        """Check if a pair is in the exclusion list."""
        return frozenset((str(id_a), str(id_b))) in self._pairs

    def add(self, id_a: str, id_b: str) -> None:
        """Add a pair to the exclusion list."""
        self._pairs.add(frozenset((str(id_a), str(id_b))))

    def __len__(self) -> int:
        return len(self._pairs)


@dataclass
class FilterResult:
    """Result of filtering a single candidate pair."""
    pair: CandidatePair
    accepted: bool
    reason: str


class VetoMatcher:
    """Runs candidate pairs through a chain of veto gates."""

    def __init__(
        self,
        gates: list[VetoGate],
        exclusions: ExclusionList | None = None,
    ) -> None:
        self.gates = list(gates)
        self.exclusions = exclusions or ExclusionList()

    def check_pair(self, pair: CandidatePair) -> FilterResult:
        """Run a single pair through all gates."""
        if self.exclusions.contains(pair.id_a, pair.id_b):
            return FilterResult(pair=pair, accepted=False, reason="explicitly excluded")
        for gate in self.gates:
            accepted, reason = gate.check_fn(pair.fields_a, pair.fields_b)
            if not accepted:
                return FilterResult(pair=pair, accepted=False, reason=f"{gate.name}: {reason}")
        return FilterResult(pair=pair, accepted=True, reason="passed all gates")

    def filter(self, candidates: list[CandidatePair]) -> list[FilterResult]:
        """Run all candidates through the gate chain."""
        return [self.check_pair(pair) for pair in candidates]


# ---------------------------------------------------------------------------
# Built-in gate constructors
# ---------------------------------------------------------------------------

def exact_field(field_name: str) -> VetoGate:
    """Both records must have the same value for field_name (if both present)."""
    def check(a: dict, b: dict) -> tuple[bool, str]:
        va, vb = a.get(field_name), b.get(field_name)
        if va is not None and vb is not None and va != vb:
            return False, f"{field_name} differs: {va!r} vs {vb!r}"
        return True, ""
    return VetoGate(name=f"exact_{field_name}", check_fn=check)


def initial_match(field_name: str) -> VetoGate:
    """First character of field values must match (case-insensitive)."""
    def check(a: dict, b: dict) -> tuple[bool, str]:
        va, vb = a.get(field_name, ""), b.get(field_name, "")
        sa, sb = str(va).strip(), str(vb).strip()
        if sa and sb and sa[0].lower() != sb[0].lower():
            return False, f"initial mismatch on {field_name}: '{sa[0]}' vs '{sb[0]}'"
        return True, ""
    return VetoGate(name=f"initial_{field_name}", check_fn=check)


def no_conflict(field_name: str) -> VetoGate:
    """If both records have a value for field_name, they must agree."""
    def check(a: dict, b: dict) -> tuple[bool, str]:
        va, vb = a.get(field_name), b.get(field_name)
        if va is not None and vb is not None and va != vb:
            return False, f"conflicting {field_name}: {va!r} vs {vb!r}"
        return True, ""
    return VetoGate(name=f"no_conflict_{field_name}", check_fn=check)


def gender_check(
    name_field: str,
    male_names: set[str],
    female_names: set[str],
) -> VetoGate:
    """Reject pairs where names suggest different genders."""
    def _first_name(fields: dict) -> str:
        name = str(fields.get(name_field, "")).strip()
        parts = name.split()
        return parts[0].lower() if parts else ""

    def check(a: dict, b: dict) -> tuple[bool, str]:
        fn_a, fn_b = _first_name(a), _first_name(b)
        a_male = fn_a in male_names
        a_female = fn_a in female_names
        b_male = fn_b in male_names
        b_female = fn_b in female_names
        if (a_male and b_female) or (a_female and b_male):
            return False, f"gender mismatch: '{fn_a}' vs '{fn_b}'"
        return True, ""
    return VetoGate(name="gender_check", check_fn=check)


def reference_ratio(min_ratio: float = 5.0, max_minor: int = 2) -> VetoGate:
    """Require one entity to dominate in references.

    Accepts if max_refs/min_refs >= min_ratio, or if the minor entity has
    at most max_minor references and the major has at least 3.
    The reference counts are read from a 'ref_count' field in each record.
    """
    def check(a: dict, b: dict) -> tuple[bool, str]:
        ra = a.get("ref_count", 0)
        rb = b.get("ref_count", 0)
        if not isinstance(ra, (int, float)) or not isinstance(rb, (int, float)):
            return True, ""
        high, low = max(ra, rb), min(ra, rb)
        if high >= 3 and low <= max_minor:
            return True, ""
        if low > 0 and high / low >= min_ratio:
            return True, ""
        return False, f"refs too balanced: {ra} vs {rb}"
    return VetoGate(name="reference_ratio", check_fn=check)
