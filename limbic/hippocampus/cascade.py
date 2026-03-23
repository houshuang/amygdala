"""Generic cascade handler for entity merges and deletes.

When entities reference each other (e.g., performances reference works, episodes
reference performances), merging or deleting an entity requires updating all
referencing entities. This module provides a storage-agnostic cascade engine
driven by a declarative reference graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# Reference schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceSpec:
    """Declares that source_type.field references target_type entities.

    If is_array is True, the field is a list of IDs (or a list of dicts each
    containing a sub-field with the ID). When sub_field is set, the reference
    lives inside array-of-dict entries (e.g., credits[].person_id).
    """
    source_type: str
    field: str
    target_type: str
    is_array: bool = False
    sub_field: str | None = None


@dataclass
class Reference:
    """A concrete reference found during a scan."""
    source_type: str
    source_id: str
    field: str
    target_id: str


class ReferenceGraph:
    """Schema of how entity types cross-reference each other."""

    def __init__(self, specs: list[ReferenceSpec]) -> None:
        self.specs = list(specs)
        self._by_target: dict[str, list[ReferenceSpec]] = {}
        for spec in self.specs:
            self._by_target.setdefault(spec.target_type, []).append(spec)

    def specs_targeting(self, target_type: str) -> list[ReferenceSpec]:
        """Return all specs where target_type is the referenced type."""
        return self._by_target.get(target_type, [])


# ---------------------------------------------------------------------------
# Data access protocol
# ---------------------------------------------------------------------------

# data_loader(entity_type) -> Iterator[(entity_id: str, data: dict)]
DataLoader = Callable[[str], Iterator[tuple[str, dict[str, Any]]]]

# data_writer(entity_type, entity_id, data) -> None
DataWriter = Callable[[str, str, dict[str, Any]], None]

# data_deleter(entity_type, entity_id) -> None
DataDeleter = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_references(
    graph: ReferenceGraph,
    entity_type: str,
    entity_id: str,
    data_loader: DataLoader,
) -> list[Reference]:
    """Find all entities that reference the given entity."""
    refs: list[Reference] = []
    for spec in graph.specs_targeting(entity_type):
        for src_id, src_data in data_loader(spec.source_type):
            if _field_references(src_data, spec, entity_id):
                refs.append(Reference(
                    source_type=spec.source_type,
                    source_id=src_id,
                    field=spec.field,
                    target_id=entity_id,
                ))
    return refs


def apply_merge(
    graph: ReferenceGraph,
    source_id: str,
    target_id: str,
    entity_type: str,
    data_loader: DataLoader,
    data_writer: DataWriter,
    data_deleter: DataDeleter,
) -> list[str]:
    """Relink all references from source to target, then delete source.

    Returns a list of human-readable modification descriptions.
    """
    modifications: list[str] = []
    for spec in graph.specs_targeting(entity_type):
        for src_entity_id, src_data in data_loader(spec.source_type):
            changed = _relink_field(src_data, spec, source_id, target_id)
            if changed:
                data_writer(spec.source_type, src_entity_id, src_data)
                modifications.append(
                    f"Relinked {spec.source_type}/{src_entity_id}.{spec.field}: "
                    f"{source_id} -> {target_id}"
                )
    data_deleter(entity_type, source_id)
    modifications.append(f"Deleted {entity_type}/{source_id}")
    return modifications


def apply_delete(
    graph: ReferenceGraph,
    entity_id: str,
    entity_type: str,
    data_loader: DataLoader,
    data_deleter: DataDeleter,
    *,
    force: bool = False,
) -> list[str]:
    """Delete an entity, checking for dangling references first.

    If force is False and references exist, raises ValueError listing them.
    Returns a list of modification descriptions.
    """
    refs = find_references(graph, entity_type, entity_id, data_loader)
    if refs and not force:
        ref_desc = ", ".join(f"{r.source_type}/{r.source_id}" for r in refs[:10])
        raise ValueError(
            f"Cannot delete {entity_type}/{entity_id}: "
            f"referenced by {len(refs)} entities ({ref_desc})"
        )
    data_deleter(entity_type, entity_id)
    return [f"Deleted {entity_type}/{entity_id}"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_id(value: Any) -> str:
    """Coerce an ID value to string for comparison."""
    return str(value)


def _field_references(data: dict[str, Any], spec: ReferenceSpec, entity_id: str) -> bool:
    """Check whether data[spec.field] references entity_id."""
    value = data.get(spec.field)
    if value is None:
        return False
    eid = _coerce_id(entity_id)

    if spec.is_array:
        if not isinstance(value, list):
            return False
        if spec.sub_field:
            return any(
                isinstance(item, dict) and _coerce_id(item.get(spec.sub_field)) == eid
                for item in value
            )
        return any(_coerce_id(v) == eid for v in value)

    return _coerce_id(value) == eid


def _relink_field(
    data: dict[str, Any],
    spec: ReferenceSpec,
    source_id: str,
    target_id: str,
) -> bool:
    """Rewrite references in data from source_id to target_id. Returns True if changed."""
    value = data.get(spec.field)
    if value is None:
        return False

    sid = _coerce_id(source_id)
    changed = False

    if spec.is_array:
        if not isinstance(value, list):
            return False
        tid = _coerce_id(target_id)
        if spec.sub_field:
            # If target already present, remove source entries (target keeps its metadata).
            # If target not present, rewrite source entries to point at target.
            has_target = any(
                isinstance(item, dict) and _coerce_id(item.get(spec.sub_field)) == tid
                for item in value
            )
            if has_target:
                new_list = [
                    item for item in value
                    if not (isinstance(item, dict) and _coerce_id(item.get(spec.sub_field)) == sid)
                ]
                if len(new_list) != len(value):
                    data[spec.field] = new_list
                    changed = True
            else:
                for item in value:
                    if isinstance(item, dict) and _coerce_id(item.get(spec.sub_field)) == sid:
                        item[spec.sub_field] = _coerce_to_type(item[spec.sub_field], target_id)
                        changed = True
        else:
            has_target = any(_coerce_id(v) == tid for v in value)
            if has_target:
                # Target already present — just remove source entries
                new_list = [v for v in value if _coerce_id(v) != sid]
                if len(new_list) != len(value):
                    data[spec.field] = new_list
                    changed = True
            else:
                new_list = []
                for v in value:
                    if _coerce_id(v) == sid:
                        new_list.append(_coerce_to_type(v, target_id))
                        changed = True
                    else:
                        new_list.append(v)
                if changed:
                    data[spec.field] = new_list
    else:
        if _coerce_id(value) == sid:
            data[spec.field] = _coerce_to_type(value, target_id)
            changed = True

    return changed


def _coerce_to_type(original: Any, new_id: str) -> Any:
    """Coerce new_id to match the type of the original value."""
    if isinstance(original, int):
        try:
            return int(new_id)
        except ValueError:
            return new_id
    return new_id
