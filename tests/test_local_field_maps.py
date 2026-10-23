"""Synthetic explicit-order map tests; expected JSON is independent of production."""

from dataclasses import FrozenInstanceError, fields
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest

from src.data import local_field_maps as module
from src.data.local_field_maps import (
    LocalFieldMap,
    load_local_field_map,
    prepare_local_field_map,
    save_local_field_map,
)


ASSIGNMENTS = (("vB", "b"), ("vA", "x"), ("vA", "y"))


class IntSubclass(int):
    pass


class StrSubclass(str):
    pass


class TupleSubclass(tuple):
    pass


def reference_semantic(party=41, epoch=3, assignments=ASSIGNMENTS):
    return {
        "format": "thesisfedrec.local_field_map",
        "schema_version": 1,
        "entity_kind": "item",
        "map_algorithm": "explicit-order.v1",
        "party_id": party,
        "schema_epoch": epoch,
        "rows": [
            {"local_id": index, "view_name": pair[0], "field_name": pair[1]}
            for index, pair in enumerate(assignments)
        ],
    }


def reference_bytes(payload):
    return json.dumps(
        payload, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")


def reference_artifact(party=41, epoch=3, assignments=ASSIGNMENTS):
    semantic = reference_semantic(party, epoch, assignments)
    return {**semantic, "map_id": hashlib.sha256(reference_bytes(semantic)).hexdigest()}


def make_map(**overrides):
    arguments = {"party_id": 41, "schema_epoch": 3, "field_assignments": ASSIGNMENTS}
    return prepare_local_field_map(**(arguments | overrides))


def load_expected(path, **overrides):
    arguments = {
        "expected_party_id": 41, "expected_epoch": 3,
        "expected_map_id": reference_artifact()["map_id"],
    }
    return load_local_field_map(path, **(arguments | overrides))


@pytest.mark.parametrize("assignments", [
    (), ASSIGNMENTS, (("vA", "same"), ("vB", "same")),
    (("é", "a b"), ("e\u0301", "名字")), (("v", "a\u200db"),),
])
@pytest.mark.parametrize("identity", [(0, 0), (41, 3), (2**63 - 1, 2**63 - 1)])
def test_independent_semantic_hash_and_exact_saved_bytes(tmp_path, assignments, identity):
    party, epoch = identity
    local_map = prepare_local_field_map(
        party_id=party, schema_epoch=epoch, field_assignments=assignments,
    )
    expected = reference_artifact(party, epoch, assignments)
    assert local_map.map_id == expected["map_id"]
    assert local_map.field_assignments == assignments
    path = tmp_path / "map.json"
    save_local_field_map(local_map, str(path))
    assert path.read_bytes() == reference_bytes(expected) + b"\n"
    assert not path.read_bytes().endswith(b"\n\n")
    assert list(tmp_path.iterdir()) == [path]
    loaded = load_local_field_map(
        path, expected_party_id=party, expected_epoch=epoch,
        expected_map_id=expected["map_id"],
    )
    assert loaded == local_map


def test_explicit_order_party_epoch_and_names_are_identity_bearing():
    base = make_map()
    variants = [
        make_map(party_id=42), make_map(schema_epoch=4),
        make_map(field_assignments=tuple(reversed(ASSIGNMENTS))),
        make_map(field_assignments=(("vB", "renamed"),) + ASSIGNMENTS[1:]),
        make_map(field_assignments=ASSIGNMENTS[:-1]),
        make_map(field_assignments=ASSIGNMENTS + (("vA", "new"),)),
    ]
    assert len({base.map_id, *(variant.map_id for variant in variants)}) == 7
    assert base.field_assignments == ASSIGNMENTS


def test_record_is_frozen_slotted_keyword_only_and_owns_tuple_structure():
    original = tuple(tuple([view, field]) for view, field in ASSIGNMENTS)
    local_map = make_map(field_assignments=original)
    assert {field.name for field in fields(local_map)} == {
        "party_id", "schema_epoch", "field_assignments", "map_id",
    }
    assert local_map.field_assignments is not original
    assert all(stored is not old for stored, old in zip(local_map.field_assignments, original))
    assert not hasattr(local_map, "__dict__")
    with pytest.raises(FrozenInstanceError):
        local_map.party_id = 12
    with pytest.raises(TypeError):
        local_map.field_assignments[0] = ("other", "field")
    with pytest.raises(TypeError):
        LocalFieldMap(41, 3, ASSIGNMENTS, local_map.map_id)


@pytest.mark.parametrize("argument", ["party_id", "schema_epoch"])
@pytest.mark.parametrize("value", [True, False, np.int64(1), IntSubclass(1), 1.0, "1", None])
def test_count_types_are_exact_builtin_int(argument, value):
    with pytest.raises(TypeError):
        make_map(**{argument: value})


@pytest.mark.parametrize("argument", ["party_id", "schema_epoch"])
@pytest.mark.parametrize("value", [-1, 2**63])
def test_count_ranges(argument, value):
    with pytest.raises(ValueError):
        make_map(**{argument: value})


@pytest.mark.parametrize("value", [
    [], None, TupleSubclass(ASSIGNMENTS), (("v", "x"), ["v", "y"]),
    (TupleSubclass(("v", "x")),), (("v", 1),), ((1, "x"),),
    ((StrSubclass("v"), "x"),), (("v", StrSubclass("x")),),
])
def test_assignment_container_and_name_types(value):
    with pytest.raises(TypeError):
        make_map(field_assignments=value)


@pytest.mark.parametrize("value", [((),), (("v",),), (("v", "x", "y"),)])
def test_assignment_pair_shape(value):
    with pytest.raises(ValueError):
        make_map(field_assignments=value)


@pytest.mark.parametrize("bad", ["", " ", " leading", "trailing ", "\ta", "a\n", "a\x00b", "a\x1fb", "a\x7fb", "a\x85b", "a\x9fb"])
@pytest.mark.parametrize("position", [0, 1])
def test_all_control_and_surrounding_whitespace_names_rejected(bad, position):
    pair = ["view", "field"]
    pair[position] = bad
    with pytest.raises(ValueError):
        make_map(field_assignments=(tuple(pair),))


def test_duplicate_pairs_rejected_without_rejecting_equal_names_in_distinct_views():
    with pytest.raises(ValueError):
        make_map(field_assignments=(("v", "x"), ("v", "x")))
    assert make_map(field_assignments=(("v", "x"), ("w", "x"))).field_assignments == (
        ("v", "x"), ("w", "x"),
    )


def test_direct_record_construction_revalidates_content():
    base = make_map()
    assert LocalFieldMap(
        party_id=41, schema_epoch=3, field_assignments=ASSIGNMENTS, map_id=base.map_id,
    ) == base
    with pytest.raises(ValueError, match="semantic content"):
        LocalFieldMap(
            party_id=42, schema_epoch=3, field_assignments=ASSIGNMENTS, map_id=base.map_id,
        )


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
def test_save_never_overwrites_any_existing_target(tmp_path, kind):
    target = tmp_path / "map.json"
    if kind == "file":
        target.write_bytes(b"existing\n")
    elif kind == "directory":
        target.mkdir()
    else:
        destination = tmp_path / "original.json"
        if kind == "symlink":
            destination.write_bytes(b"original\n")
        target.symlink_to(destination)
    before_names = sorted(path.name for path in tmp_path.iterdir())
    before_link = os.readlink(target) if target.is_symlink() else None
    with pytest.raises(FileExistsError):
        save_local_field_map(make_map(), target)
    assert sorted(path.name for path in tmp_path.iterdir()) == before_names
    if kind == "file":
        assert target.read_bytes() == b"existing\n"
    elif kind == "directory":
        assert target.is_dir() and list(target.iterdir()) == []
    else:
        assert os.readlink(target) == before_link
        if kind == "symlink":
            assert target.read_bytes() == b"original\n"


def test_atomic_publication_cannot_overwrite_a_racing_creator(tmp_path, monkeypatch):
    target = tmp_path / "map.json"
    original_link = module.os.link

    def raced_link(source, destination):
        Path(destination).write_bytes(b"racer\n")
        return original_link(source, destination)

    monkeypatch.setattr(module.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        save_local_field_map(make_map(), target)
    assert target.read_bytes() == b"racer\n"
    assert list(tmp_path.iterdir()) == [target]


def test_publication_occurs_only_after_complete_write_and_fsync(tmp_path, monkeypatch):
    target = tmp_path / "map.json"
    expected = reference_bytes(reference_artifact()) + b"\n"
    original_fsync, original_link = module.os.fsync, module.os.link
    stages = []

    def checked_fsync(descriptor):
        assert os.fstat(descriptor).st_size == len(expected)
        stages.append("fsync")
        return original_fsync(descriptor)

    def checked_link(source, destination):
        assert stages == ["fsync"]
        assert Path(source).parent == target.parent
        assert Path(source).read_bytes() == expected
        stages.append("link")
        return original_link(source, destination)

    monkeypatch.setattr(module.os, "fsync", checked_fsync)
    monkeypatch.setattr(module.os, "link", checked_link)
    save_local_field_map(make_map(), target)
    assert stages == ["fsync", "link"]
    assert target.read_bytes() == expected


@pytest.mark.parametrize("failure_stage", ["fsync", "link"])
def test_io_failure_propagates_and_removes_only_own_temporary(tmp_path, monkeypatch, failure_stage):
    keep = tmp_path / ".unrelated.tmp"
    keep.write_bytes(b"keep")
    sentinel = PermissionError("injected filesystem error")

    def fail(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(module.os, failure_stage, fail)
    with pytest.raises(PermissionError) as caught:
        save_local_field_map(make_map(), tmp_path / "map.json")
    assert caught.value is sentinel
    assert list(tmp_path.iterdir()) == [keep]
    assert keep.read_bytes() == b"keep"


def test_caller_must_create_parent_and_io_errors_are_not_swallowed(tmp_path):
    with pytest.raises(FileNotFoundError):
        save_local_field_map(make_map(), tmp_path / "absent" / "map.json")
    with pytest.raises(FileNotFoundError):
        load_expected(tmp_path / "absent.json")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("corruption", ["map_id", "party_id", "field_assignments"])
def test_save_rechecks_corrupted_record_before_filesystem_mutation(tmp_path, corruption):
    local_map = make_map()
    # Deliberate internals bypass is not prevented; public consumers still validate.
    object.__setattr__(local_map, corruption, {
        "map_id": "0" * 64, "party_id": 99,
        "field_assignments": (("other", "field"),),
    }[corruption])
    with pytest.raises(ValueError):
        save_local_field_map(local_map, tmp_path / "map.json")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [None, {}, object()])
def test_save_requires_map_type(tmp_path, bad):
    with pytest.raises(TypeError):
        save_local_field_map(bad, tmp_path / "map.json")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [None, 1, b"map.json"])
def test_path_types(tmp_path, bad):
    with pytest.raises(TypeError):
        save_local_field_map(make_map(), bad)
    with pytest.raises(TypeError):
        load_expected(bad)


def test_load_accepts_noncanonical_physical_whitespace_and_key_order(tmp_path):
    artifact = reference_artifact()
    path = tmp_path / "map.json"
    artifact = dict(reversed(list(artifact.items())))
    artifact["rows"] = [dict(reversed(list(row.items()))) for row in artifact["rows"]]
    physical = ("  \n" + json.dumps(artifact, indent=3) + "\n\n ").encode()
    path.write_bytes(physical)
    assert load_expected(path) == make_map()
    assert path.read_bytes() == physical


@pytest.mark.parametrize("text", [
    "{", "", "null null", '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
    '{"a":1,"a":2}',
])
def test_load_rejects_malformed_json_duplicate_keys_and_nonfinite_constants(tmp_path, text):
    path = tmp_path / "map.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_expected(path)
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("location", ["top", "row"])
def test_duplicate_keys_rejected_even_when_values_match(tmp_path, location):
    text = reference_bytes(reference_artifact()).decode()
    if location == "top":
        text = text.replace('"party_id":41', '"party_id":41,"party_id":41')
    else:
        text = text.replace('"local_id":0', '"local_id":0,"local_id":0')
    path = tmp_path / "map.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_expected(path)


@pytest.mark.parametrize("payload", [None, [], 1, True, "map"])
def test_json_top_level_wrong_type(tmp_path, payload):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TypeError):
        load_expected(path)


@pytest.mark.parametrize("key", list(reference_artifact()))
def test_every_top_level_key_is_required(tmp_path, key):
    artifact = reference_artifact()
    del artifact[key]
    path = tmp_path / "map.json"
    path.write_bytes(reference_bytes(artifact))
    with pytest.raises(ValueError, match="keys"):
        load_expected(path)


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p.update(extra=1), ValueError),
    (lambda p: p.update(format="other"), ValueError),
    (lambda p: p.update(format=1), TypeError),
    (lambda p: p.update(entity_kind="user"), ValueError),
    (lambda p: p.update(entity_kind=False), TypeError),
    (lambda p: p.update(map_algorithm="digest-order.v1"), ValueError),
    (lambda p: p.update(map_algorithm=None), TypeError),
    (lambda p: p.update(schema_version=2), ValueError),
    (lambda p: p.update(schema_version=True), TypeError),
    (lambda p: p.update(schema_version=1.0), TypeError),
    (lambda p: p.update(party_id=True), TypeError),
    (lambda p: p.update(party_id=-1), ValueError),
    (lambda p: p.update(schema_epoch=2**63), ValueError),
    (lambda p: p.update(schema_epoch="3"), TypeError),
    (lambda p: p.update(rows={}), TypeError),
    (lambda p: p.update(rows=[None]), TypeError),
    (lambda p: p["rows"][0].update(extra=0), ValueError),
    (lambda p: p["rows"][0].pop("field_name"), ValueError),
    (lambda p: p["rows"][0].update(local_id=True), TypeError),
    (lambda p: p["rows"][0].update(local_id=1), ValueError),
    (lambda p: p["rows"][0].update(local_id=-1), ValueError),
    (lambda p: p["rows"][1].update(local_id=0), ValueError),
    (lambda p: p["rows"].reverse(), ValueError),
    (lambda p: p["rows"][0].update(view_name=""), ValueError),
    (lambda p: p["rows"][0].update(field_name="x\x85y"), ValueError),
    (lambda p: p["rows"][0].update(field_name=1), TypeError),
    (lambda p: p["rows"][2].update(field_name="x"), ValueError),
    (lambda p: p.update(map_id=""), ValueError),
    (lambda p: p.update(map_id="A" * 64), ValueError),
    (lambda p: p.update(map_id="0" * 63), ValueError),
    (lambda p: p.update(map_id=5), TypeError),
    (lambda p: p.update(map_id="0" * 64), ValueError),
    (lambda p: p["rows"][0].update(field_name="renamed"), ValueError),
])
def test_strict_json_schema_content_and_identity(tmp_path, mutation, error):
    artifact = reference_artifact()
    mutation(artifact)
    physical = reference_bytes(artifact)
    path = tmp_path / "map.json"
    path.write_bytes(physical)
    with pytest.raises(error):
        load_expected(path)
    assert path.read_bytes() == physical


@pytest.mark.parametrize("argument,value,error", [
    ("expected_party_id", 42, ValueError),
    ("expected_epoch", 4, ValueError),
    ("expected_map_id", "0" * 64, ValueError),
    ("expected_party_id", True, TypeError),
    ("expected_epoch", np.int64(3), TypeError),
    ("expected_epoch", -1, ValueError),
    ("expected_map_id", StrSubclass("0" * 64), TypeError),
    ("expected_map_id", None, TypeError),
    ("expected_map_id", "A" * 64, ValueError),
])
def test_expected_identity_is_strict_and_not_inferred(tmp_path, argument, value, error):
    path = tmp_path / "map.json"
    physical = reference_bytes(reference_artifact())
    path.write_bytes(physical)
    with pytest.raises(error):
        load_expected(path, **{argument: value})
    assert path.read_bytes() == physical


def test_no_implicit_rename_or_epoch_migration(tmp_path):
    changed = reference_artifact(assignments=(("vB", "renamed"),) + ASSIGNMENTS[1:])
    path = tmp_path / "renamed.json"
    path.write_bytes(reference_bytes(changed))
    with pytest.raises(ValueError, match="expected_map_id"):
        load_expected(path)
    assert load_expected(path, expected_map_id=changed["map_id"]).map_id == changed["map_id"]
    epoch = reference_artifact(epoch=4)
    path.write_bytes(reference_bytes(epoch))
    with pytest.raises(ValueError, match="expected_epoch"):
        load_expected(path, expected_map_id=epoch["map_id"])


def test_map_operations_leave_python_numpy_rng_and_inputs_unchanged(tmp_path):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    inputs = tuple(tuple(pair) for pair in ASSIGNMENTS)
    local_map = make_map(field_assignments=inputs)
    path = tmp_path / "map.json"
    save_local_field_map(local_map, path)
    assert load_expected(path) == local_map
    assert inputs == ASSIGNMENTS
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]
