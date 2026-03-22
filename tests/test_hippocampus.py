"""Tests for limbic.hippocampus — proposals, cascade, dedup, validate, store."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from limbic.hippocampus.proposals import Proposal, Change, ProposalStore, parse_field_value
from limbic.hippocampus.cascade import (
    ReferenceSpec, ReferenceGraph, find_references, apply_merge, apply_delete,
)
from limbic.hippocampus.dedup import (
    CandidatePair, VetoGate, VetoMatcher, ExclusionList,
    exact_field, initial_match, no_conflict, gender_check, reference_ratio,
)
from limbic.hippocampus.validate import (
    ValidationResult, Rule, Validator,
    required_field, valid_values, reference_exists, no_orphans, conditional_required,
)
from limbic.hippocampus.store import YAMLStore


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def proposal_store(tmp_dir):
    return ProposalStore(tmp_dir / "proposals")


@pytest.fixture
def yaml_store(tmp_dir):
    schema = {"person": "persons", "work": "works", "performance": "performances"}
    store = YAMLStore(tmp_dir / "data", schema)
    # Seed some data
    store.save("person", "1", {"id": 1, "name": "Alice"})
    store.save("person", "2", {"id": 2, "name": "Bob"})
    store.save("work", "10", {"id": 10, "title": "Hamlet", "playwright_id": 1})
    store.save("work", "11", {"id": 11, "title": "Macbeth", "playwright_id": 2})
    store.save("performance", "100", {"id": 100, "work_id": 10, "year": 2020})
    store.save("performance", "101", {"id": 101, "work_id": 11, "year": 2021})
    return store


# ===========================================================================
# ProposalStore tests
# ===========================================================================

class TestProposalStore:
    def test_create_modify(self, proposal_store):
        p = proposal_store.create_modify(
            "person/42",
            {"name": "Alice Updated", "birth_year": "1980"},
            title="Fix name",
            reasoning="Typo correction",
            current_state={"name": "Alce", "birth_year": None},
        )
        assert p.status == "pending"
        assert p.id.startswith("prop_")
        assert len(p.changes) == 1
        assert p.changes[0].action == "modify"
        assert p.changes[0].entity_type == "person"
        assert p.changes[0].entity_id == "42"
        assert p.changes[0].proposed_state["name"] == "Alice Updated"
        assert p.changes[0].proposed_state["birth_year"] == 1980

    def test_create_merge(self, proposal_store):
        p = proposal_store.create_merge(
            "person/2", "person/1",
            title="Merge duplicate",
            reasoning="Same person",
        )
        assert p.changes[0].action == "merge"
        assert p.changes[0].merge_target == "1"

    def test_merge_different_types_raises(self, proposal_store):
        with pytest.raises(ValueError, match="Cannot merge different types"):
            proposal_store.create_merge("person/1", "work/10", title="Bad", reasoning="test")

    def test_create_delete(self, proposal_store):
        p = proposal_store.create_delete(
            "work/99", title="Remove orphan", reasoning="No references",
        )
        assert p.changes[0].action == "delete"

    def test_lifecycle(self, proposal_store):
        p = proposal_store.create_modify(
            "work/1", {"title": "New Title"}, title="Rename", reasoning="Better name",
        )
        # List pending
        pending = proposal_store.list_pending()
        assert len(pending) == 1
        assert pending[0].id == p.id

        # Approve
        approved = proposal_store.approve(p.id)
        assert approved.status == "approved"
        assert len(proposal_store.list_pending()) == 0
        assert len(proposal_store.list_approved()) == 1

        # Load by ID
        loaded = proposal_store.load(p.id)
        assert loaded is not None
        assert loaded.status == "approved"

        # Mark applied
        applied = proposal_store.mark_applied(p.id)
        assert applied.status == "applied"
        assert len(proposal_store.list_approved()) == 0

    def test_reject(self, proposal_store):
        p = proposal_store.create_delete("work/5", title="Delete", reasoning="test")
        rejected = proposal_store.reject(p.id)
        assert rejected.status == "rejected"
        assert len(proposal_store.list_pending()) == 0

    def test_approve_nonexistent_raises(self, proposal_store):
        with pytest.raises(FileNotFoundError):
            proposal_store.approve("prop_nonexistent")

    def test_roundtrip_serialization(self, proposal_store):
        p = proposal_store.create_modify(
            "person/7",
            {"tags": "[1, 2, 3]", "active": "true", "score": "42", "notes": "null"},
            title="Parse test",
            reasoning="test value parsing",
            current_state={"name": "Test"},
        )
        loaded = proposal_store.load(p.id)
        assert loaded is not None
        changes = loaded.changes[0]
        assert changes.proposed_state["tags"] == [1, 2, 3]
        assert changes.proposed_state["active"] is True
        assert changes.proposed_state["score"] == 42
        assert changes.proposed_state["notes"] is None


class TestParseFieldValue:
    def test_int(self):
        assert parse_field_value("42") == 42
        assert parse_field_value("-5") == -5

    def test_bool(self):
        assert parse_field_value("true") is True
        assert parse_field_value("False") is False

    def test_null(self):
        assert parse_field_value("null") is None
        assert parse_field_value("None") is None

    def test_list(self):
        assert parse_field_value("[1, 2, 3]") == [1, 2, 3]

    def test_dict(self):
        assert parse_field_value("{'a': 1}") == {"a": 1}

    def test_string_fallback(self):
        assert parse_field_value("hello world") == "hello world"


# ===========================================================================
# Cascade tests
# ===========================================================================

class TestCascade:
    def _make_graph(self):
        return ReferenceGraph([
            ReferenceSpec("performance", "work_id", "work"),
            ReferenceSpec("work", "playwright_id", "person"),
            ReferenceSpec("performance", "credits", "person", is_array=True, sub_field="person_id"),
        ])

    def _make_data(self):
        return {
            "person": {
                "1": {"id": 1, "name": "Alice"},
                "2": {"id": 2, "name": "Bob"},
            },
            "work": {
                "10": {"id": 10, "title": "Hamlet", "playwright_id": 1},
                "11": {"id": 11, "title": "Macbeth", "playwright_id": 2},
            },
            "performance": {
                "100": {"id": 100, "work_id": 10, "credits": [{"person_id": 1, "role": "actor"}]},
                "101": {"id": 101, "work_id": 11, "credits": [{"person_id": 2, "role": "director"}]},
            },
        }

    def _loader(self, data):
        def load(entity_type):
            return iter(data.get(entity_type, {}).items())
        return load

    def _writer(self, data):
        def write(entity_type, entity_id, entity_data):
            data[entity_type][entity_id] = entity_data
        return write

    def _deleter(self, data):
        def delete(entity_type, entity_id):
            data[entity_type].pop(entity_id, None)
        return delete

    def test_find_references(self):
        graph = self._make_graph()
        data = self._make_data()
        refs = find_references(graph, "work", "10", self._loader(data))
        assert len(refs) == 1
        assert refs[0].source_type == "performance"
        assert refs[0].source_id == "100"

    def test_find_references_person(self):
        graph = self._make_graph()
        data = self._make_data()
        refs = find_references(graph, "person", "1", self._loader(data))
        # Should find work/10 (playwright_id) and performance/100 (credits)
        assert len(refs) == 2
        ref_sources = {(r.source_type, r.source_id) for r in refs}
        assert ("work", "10") in ref_sources
        assert ("performance", "100") in ref_sources

    def test_apply_merge_work(self):
        graph = self._make_graph()
        data = self._make_data()
        mods = apply_merge(graph, "10", "11", "work",
                           self._loader(data), self._writer(data), self._deleter(data))
        # Performance 100 should now reference work 11
        assert data["performance"]["100"]["work_id"] == 11
        # Work 10 should be deleted
        assert "10" not in data["work"]
        assert any("Relinked" in m for m in mods)
        assert any("Deleted" in m for m in mods)

    def test_apply_merge_person(self):
        graph = self._make_graph()
        data = self._make_data()
        mods = apply_merge(graph, "1", "2", "person",
                           self._loader(data), self._writer(data), self._deleter(data))
        # Work 10 playwright_id should now be 2
        assert data["work"]["10"]["playwright_id"] == 2
        # Performance 100 credits should reference person 2
        assert data["performance"]["100"]["credits"][0]["person_id"] == 2
        assert "1" not in data["person"]

    def test_apply_delete_with_references_raises(self):
        graph = self._make_graph()
        data = self._make_data()
        with pytest.raises(ValueError, match="referenced by"):
            apply_delete(graph, "10", "work", self._loader(data), self._deleter(data))

    def test_apply_delete_force(self):
        graph = self._make_graph()
        data = self._make_data()
        mods = apply_delete(graph, "10", "work", self._loader(data), self._deleter(data), force=True)
        assert "10" not in data["work"]
        assert any("Deleted" in m for m in mods)

    def test_apply_delete_no_references(self):
        graph = self._make_graph()
        data = self._make_data()
        # Add an unreferenced work
        data["work"]["99"] = {"id": 99, "title": "Orphan"}
        mods = apply_delete(graph, "99", "work", self._loader(data), self._deleter(data))
        assert "99" not in data["work"]


# ===========================================================================
# Dedup tests
# ===========================================================================

class TestVetoGates:
    def test_exact_field_accepts_matching(self):
        gate = exact_field("birth_year")
        ok, reason = gate.check_fn({"birth_year": 1950}, {"birth_year": 1950})
        assert ok is True

    def test_exact_field_rejects_mismatch(self):
        gate = exact_field("birth_year")
        ok, reason = gate.check_fn({"birth_year": 1950}, {"birth_year": 1960})
        assert ok is False
        assert "differs" in reason

    def test_exact_field_accepts_missing(self):
        gate = exact_field("birth_year")
        ok, _ = gate.check_fn({"birth_year": 1950}, {})
        assert ok is True

    def test_initial_match_accepts(self):
        gate = initial_match("name")
        ok, _ = gate.check_fn({"name": "Alice"}, {"name": "Alicia"})
        assert ok is True

    def test_initial_match_rejects(self):
        gate = initial_match("name")
        ok, _ = gate.check_fn({"name": "Alice"}, {"name": "Bob"})
        assert ok is False

    def test_no_conflict_accepts_one_missing(self):
        gate = no_conflict("wikidata_id")
        ok, _ = gate.check_fn({"wikidata_id": "Q123"}, {})
        assert ok is True

    def test_no_conflict_rejects_different(self):
        gate = no_conflict("wikidata_id")
        ok, _ = gate.check_fn({"wikidata_id": "Q123"}, {"wikidata_id": "Q456"})
        assert ok is False

    def test_gender_check(self):
        gate = gender_check("name", male_names={"hans", "per"}, female_names={"kari", "anne"})
        ok, _ = gate.check_fn({"name": "Hans Berg"}, {"name": "Kari Berg"})
        assert ok is False
        ok, _ = gate.check_fn({"name": "Hans Berg"}, {"name": "Per Berg"})
        assert ok is True

    def test_reference_ratio_accepts_skewed(self):
        gate = reference_ratio(min_ratio=5.0, max_minor=2)
        ok, _ = gate.check_fn({"ref_count": 20}, {"ref_count": 1})
        assert ok is True

    def test_reference_ratio_rejects_balanced(self):
        gate = reference_ratio(min_ratio=5.0, max_minor=2)
        ok, _ = gate.check_fn({"ref_count": 5}, {"ref_count": 4})
        assert ok is False


class TestVetoMatcher:
    def test_full_pipeline(self):
        gates = [
            exact_field("birth_year"),
            initial_match("name"),
        ]
        exclusions = ExclusionList({frozenset(("1", "3"))})
        matcher = VetoMatcher(gates, exclusions)

        candidates = [
            CandidatePair("1", "2", {"name": "Alice", "birth_year": 1950}, {"name": "Alicia", "birth_year": 1950}, 0.9),
            CandidatePair("1", "3", {"name": "Alice"}, {"name": "Alice"}, 1.0),  # excluded
            CandidatePair("4", "5", {"name": "Bob", "birth_year": 1960}, {"name": "Bob", "birth_year": 1970}, 0.8),
        ]
        results = matcher.filter(candidates)
        assert len(results) == 3
        assert results[0].accepted is True  # matching
        assert results[1].accepted is False  # excluded
        assert "excluded" in results[1].reason
        assert results[2].accepted is False  # birth year conflict

    def test_exclusion_list(self):
        el = ExclusionList()
        el.add("1", "2")
        assert el.contains("1", "2") is True
        assert el.contains("2", "1") is True  # frozenset is order-independent
        assert el.contains("1", "3") is False
        assert len(el) == 1


# ===========================================================================
# Validation tests
# ===========================================================================

class TestValidator:
    def _entities(self):
        return {
            "person": {
                "1": {"id": 1, "name": "Alice"},
                "2": {"id": 2, "name": ""},  # empty name
                "3": {"id": 3},  # missing name
            },
            "work": {
                "10": {"id": 10, "title": "Hamlet", "category": "teater", "author_id": "1"},
                "11": {"id": 11, "title": "Unknown", "category": "invalid_cat", "author_id": "99"},
            },
        }

    def test_required_field(self):
        rules = [required_field("person", "name")]
        v = Validator(rules)
        result = v.validate(self._entities())
        # Person 2 has empty name, person 3 has missing name
        assert len(result.errors) == 2
        assert any("2" in e for e in result.errors)
        assert any("3" in e for e in result.errors)

    def test_valid_values(self):
        rules = [valid_values("work", "category", {"teater", "opera", "konsert"})]
        v = Validator(rules)
        result = v.validate(self._entities())
        assert len(result.errors) == 1
        assert "invalid_cat" in result.errors[0]

    def test_reference_exists(self):
        rules = [reference_exists("work", "author_id", "person")]
        v = Validator(rules)
        result = v.validate(self._entities())
        # Work 11 references person 99 which does not exist
        assert len(result.errors) == 1
        assert "99" in result.errors[0]

    def test_no_orphans(self):
        rules = [no_orphans("person", [("work", "author_id")])]
        v = Validator(rules)
        result = v.validate(self._entities())
        # Persons 2 and 3 are not referenced by any work
        assert len(result.warnings) == 2

    def test_conditional_required(self):
        rules = [conditional_required(
            "work",
            lambda d: d.get("category") == "teater",
            "author_id",
            condition_label="category is teater",
        )]
        v = Validator(rules)
        entities = {
            "work": {
                "10": {"id": 10, "category": "teater", "author_id": "1"},
                "11": {"id": 11, "category": "teater"},  # missing author
                "12": {"id": 12, "category": "opera"},  # OK, not teater
            },
        }
        result = v.validate(entities)
        assert len(result.errors) == 1
        assert "11" in result.errors[0]

    def test_validation_result_merge(self):
        r1 = ValidationResult(errors=["e1"], warnings=["w1"])
        r2 = ValidationResult(errors=["e2"], warnings=["w2"])
        r1.merge(r2)
        assert len(r1.errors) == 2
        assert len(r1.warnings) == 2
        assert not r1.ok

    def test_validation_result_ok(self):
        r = ValidationResult()
        assert r.ok is True
        r.warnings.append("just a warning")
        assert r.ok is True  # warnings don't affect ok
        r.errors.append("an error")
        assert r.ok is False

    def test_warning_severity(self):
        rules = [required_field("person", "email", severity="warning")]
        v = Validator(rules)
        entities = {"person": {"1": {"id": 1, "name": "Alice"}}}
        result = v.validate(entities)
        assert len(result.errors) == 0
        assert len(result.warnings) == 1


# ===========================================================================
# YAMLStore tests
# ===========================================================================

class TestYAMLStore:
    def test_save_and_load(self, yaml_store):
        data = yaml_store.load("person", "1")
        assert data is not None
        assert data["name"] == "Alice"

    def test_load_nonexistent(self, yaml_store):
        assert yaml_store.load("person", "999") is None

    def test_delete(self, yaml_store):
        assert yaml_store.delete("person", "1") is True
        assert yaml_store.load("person", "1") is None
        assert yaml_store.delete("person", "1") is False  # already gone

    def test_iter_type(self, yaml_store):
        items = list(yaml_store.iter_type("person"))
        assert len(items) == 2
        ids = {eid for eid, _ in items}
        assert ids == {"1", "2"}

    def test_all_ids(self, yaml_store):
        ids = yaml_store.all_ids("person")
        assert ids == {"1", "2"}

    def test_all_ids_empty_type(self, yaml_store):
        ids = yaml_store.all_ids("work")
        assert "10" in ids
        assert "11" in ids

    def test_unknown_type_raises(self, yaml_store):
        with pytest.raises(ValueError, match="Unknown entity type"):
            yaml_store.load("invalid", "1")

    def test_backup(self, yaml_store, tmp_dir):
        backup_path = yaml_store.backup("person", "1", backup_dir=tmp_dir / "backups")
        assert backup_path.exists()
        with open(backup_path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "Alice"

    def test_backup_nonexistent_raises(self, yaml_store):
        with pytest.raises(FileNotFoundError):
            yaml_store.backup("person", "999")

    def test_save_creates_directories(self, tmp_dir):
        store = YAMLStore(tmp_dir / "fresh", {"item": "items"})
        store.save("item", "1", {"id": 1, "value": "test"})
        loaded = store.load("item", "1")
        assert loaded["value"] == "test"

    def test_unicode_handling(self, yaml_store):
        yaml_store.save("person", "3", {"id": 3, "name": "Bjorn Stoveland"})
        data = yaml_store.load("person", "3")
        assert data["name"] == "Bjorn Stoveland"

    def test_multiline_strings(self, yaml_store):
        yaml_store.save("work", "20", {"id": 20, "title": "Test", "description": "Line 1\nLine 2\nLine 3"})
        data = yaml_store.load("work", "20")
        assert data["description"] == "Line 1\nLine 2\nLine 3"


# ===========================================================================
# Integration: Cascade + Store
# ===========================================================================

class TestCascadeWithStore:
    def test_merge_via_store(self, yaml_store):
        graph = ReferenceGraph([
            ReferenceSpec("performance", "work_id", "work"),
            ReferenceSpec("work", "playwright_id", "person"),
        ])

        def loader(entity_type):
            return yaml_store.iter_type(entity_type)

        def writer(entity_type, entity_id, data):
            yaml_store.save(entity_type, entity_id, data)

        def deleter(entity_type, entity_id):
            yaml_store.delete(entity_type, entity_id)

        # Merge work 10 into work 11
        mods = apply_merge(graph, "10", "11", "work", loader, writer, deleter)

        # Work 10 should be gone
        assert yaml_store.load("work", "10") is None

        # Performance 100 should now reference work 11
        perf = yaml_store.load("performance", "100")
        assert perf["work_id"] == 11

        assert len(mods) >= 2
