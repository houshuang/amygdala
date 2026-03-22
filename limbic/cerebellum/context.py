"""Context builder for LLM verification prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """Incrementally builds structured context for an LLM verification prompt.

    Usage::

        ctx = ContextBuilder()
        ctx.add_entity("work", "42", {"title": "Peer Gynt", "year": 1867})
        ctx.add_related("performances", [{"id": 1, "venue": "DNS"}])
        ctx.add_metadata("category", "teater")
        prompt_text = ctx.build(format="markdown")
    """

    def __init__(self) -> None:
        self._entities: list[dict[str, Any]] = []
        self._related: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}

    def add_entity(self, entity_type: str, entity_id: str, data: dict) -> "ContextBuilder":
        """Add a primary entity to the context."""
        self._entities.append({
            "type": entity_type,
            "id": entity_id,
            "data": data,
        })
        return self

    def add_related(self, label: str, entities: list[dict]) -> "ContextBuilder":
        """Add a group of related entities under a label."""
        self._related.append({
            "label": label,
            "entities": entities,
        })
        return self

    def add_metadata(self, key: str, value: Any) -> "ContextBuilder":
        """Add an extra metadata key-value pair."""
        self._metadata[key] = value
        return self

    def build(self, format: str = "markdown") -> str | dict:
        """Render the collected context.

        Args:
            format: "markdown" returns a string, "json" returns a dict.
        """
        if format == "json":
            return self._build_json()
        return self._build_markdown()

    # -- private -----------------------------------------------------------

    def _build_json(self) -> dict:
        result: dict[str, Any] = {}
        if self._entities:
            result["entities"] = [
                {"type": e["type"], "id": e["id"], **e["data"]}
                for e in self._entities
            ]
        if self._related:
            result["related"] = {
                r["label"]: r["entities"] for r in self._related
            }
        if self._metadata:
            result["metadata"] = dict(self._metadata)
        return result

    def _build_markdown(self) -> str:
        lines: list[str] = []

        for ent in self._entities:
            lines.append(f"## {ent['type']}/{ent['id']}")
            for k, v in ent["data"].items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        for rel in self._related:
            label = rel["label"]
            items = rel["entities"]
            lines.append(f"### {label} ({len(items)})")
            for item in items:
                parts = [f"{k}={v}" for k, v in item.items()]
                lines.append(f"  - {', '.join(parts)}")
            lines.append("")

        if self._metadata:
            lines.append("### Metadata")
            for k, v in self._metadata.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch context helper
# ---------------------------------------------------------------------------

def build_batch_context(
    items: list[Any],
    context_fn: Callable[[Any], ContextBuilder],
    format: str = "markdown",
) -> str:
    """Build combined context for a batch of items.

    Args:
        items: The items to build context for.
        context_fn: Called per item, should return a populated ContextBuilder.
        format: "markdown" or "json" (json batches are joined as a JSON array string).

    Returns:
        Combined context as a single string.
    """
    if format == "json":
        parts = [context_fn(item).build(format="json") for item in items]
        return json.dumps(parts, ensure_ascii=False, indent=2)

    sections: list[str] = []
    for i, item in enumerate(items):
        ctx = context_fn(item)
        section = ctx.build(format="markdown")
        if i > 0:
            sections.append("---")
        sections.append(section)
    return "\n".join(sections)
