"""Explicit party-local field identities and their small versioned JSON artifact.

The map is trusted local bookkeeping, not an authorization or semantic-alignment
proof. It contains no feature payload, model parameters, random state, or metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata


_MAX_INT = 2**63 - 1
_FORMAT = "thesisfedrec.local_field_map"
_ALGORITHM = "explicit-order.v1"
_SEMANTIC_KEYS = {
    "format", "schema_version", "entity_kind", "map_algorithm", "party_id",
    "schema_epoch", "rows",
}


def _validate_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in int")
    if not 0 <= value <= _MAX_INT:
        raise ValueError(f"{name} must be in [0, 2**63-1]")


def _validate_text(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty without surrounding whitespace")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError(f"{name} must not contain Unicode control characters")


def _validate_hash(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase 64-character SHA-256 hex")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validate_assignments(assignments: object) -> None:
    if type(assignments) is not tuple:
        raise TypeError("field_assignments must be an exact tuple")
    for pair in assignments:
        if type(pair) is not tuple:
            raise TypeError("each field assignment must be an exact tuple")
        if len(pair) != 2:
            raise ValueError("each field assignment must contain exactly two names")
        for name in pair:
            if type(name) is not str:
                raise TypeError("field assignment names must be exact built-in str")
    for view_name, field_name in assignments:
        _validate_text(view_name, "view_name")
        _validate_text(field_name, "field_name")
    if len(set(assignments)) != len(assignments):
        raise ValueError("field assignment pairs must be unique")


def _semantic(
    party_id: int, schema_epoch: int, assignments: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "format": _FORMAT,
        "schema_version": 1,
        "entity_kind": "item",
        "map_algorithm": _ALGORITHM,
        "party_id": party_id,
        "schema_epoch": schema_epoch,
        "rows": [
            {"local_id": index, "view_name": view, "field_name": field}
            for index, (view, field) in enumerate(assignments)
        ],
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalFieldMap:
    """Immutable explicit local-ID order; construct with the validated factory."""

    party_id: int
    schema_epoch: int
    field_assignments: tuple[tuple[str, str], ...]
    map_id: str

    def __post_init__(self) -> None:
        _validate_local_map(self)
        object.__setattr__(
            self, "field_assignments",
            tuple((view, field) for view, field in self.field_assignments),
        )


def _validate_local_map(local_map: object) -> None:
    if type(local_map) is not LocalFieldMap:
        raise TypeError("local_map must be an exact LocalFieldMap")
    _validate_int(local_map.party_id, "party_id")
    _validate_int(local_map.schema_epoch, "schema_epoch")
    _validate_assignments(local_map.field_assignments)
    _validate_hash(local_map.map_id, "map_id")
    digest = hashlib.sha256(_canonical_bytes(_semantic(
        local_map.party_id, local_map.schema_epoch, local_map.field_assignments,
    ))).hexdigest()
    if digest != local_map.map_id:
        raise ValueError("local_map map_id does not match its semantic content")


def prepare_local_field_map(
    *, party_id: int, schema_epoch: int,
    field_assignments: tuple[tuple[str, str], ...],
) -> LocalFieldMap:
    """Validate caller-assigned order without sorting, coercion, or randomness."""
    _validate_int(party_id, "party_id")
    _validate_int(schema_epoch, "schema_epoch")
    _validate_assignments(field_assignments)
    digest = hashlib.sha256(_canonical_bytes(
        _semantic(party_id, schema_epoch, field_assignments),
    )).hexdigest()
    return LocalFieldMap(
        party_id=party_id, schema_epoch=schema_epoch,
        field_assignments=field_assignments, map_id=digest,
    )


def _path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be str or pathlib.Path")
    return Path(path)


def save_local_field_map(local_map: LocalFieldMap, path: str | Path) -> None:
    """Create a new file atomically without replacing any existing target.

    The parent directory must already exist. Publication uses a same-directory
    hard link after flushing/fsyncing the complete temporary file; unlike a
    pre-existence check followed by replace, this does not overwrite a racer.
    Filesystem exceptions propagate. Only this call's temporary name is removed.
    """
    _validate_local_map(local_map)
    payload = _semantic(
        local_map.party_id, local_map.schema_epoch, local_map.field_assignments,
    )
    payload["map_id"] = local_map.map_id
    encoded = _canonical_bytes(payload) + b"\n"
    target = _path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _check_keys(value: object, keys: set[str], name: str) -> None:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a JSON object")
    if set(value) != keys:
        raise ValueError(f"{name} has missing or unknown keys")


def _check_literal(value: object, expected: str, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a JSON string")
    if value != expected:
        raise ValueError(f"unsupported {name}: {value!r}")


def load_local_field_map(
    path: str | Path, *, expected_party_id: int,
    expected_epoch: int, expected_map_id: str,
) -> LocalFieldMap:
    """Read strict version-1 JSON and verify content plus all expected identities."""
    _validate_int(expected_party_id, "expected_party_id")
    _validate_int(expected_epoch, "expected_epoch")
    _validate_hash(expected_map_id, "expected_map_id")
    text = _path(path).read_text(encoding="utf-8")
    payload = json.loads(
        text, object_pairs_hook=_unique_json_object,
        parse_constant=_invalid_json_constant,
    )
    _check_keys(payload, _SEMANTIC_KEYS | {"map_id"}, "map artifact")
    _check_literal(payload["format"], _FORMAT, "format")
    _check_literal(payload["entity_kind"], "item", "entity_kind")
    _check_literal(payload["map_algorithm"], _ALGORITHM, "map_algorithm")
    _validate_int(payload["schema_version"], "schema_version")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    _validate_int(payload["party_id"], "party_id")
    _validate_int(payload["schema_epoch"], "schema_epoch")
    _validate_hash(payload["map_id"], "map_id")
    rows = payload["rows"]
    if type(rows) is not list:
        raise TypeError("rows must be a JSON list")
    for index, row in enumerate(rows):
        _check_keys(row, {"local_id", "view_name", "field_name"}, "map row")
        _validate_int(row["local_id"], "local_id")
        if row["local_id"] != index:
            raise ValueError("local IDs must be ordered and contiguous from zero")
    local_map = prepare_local_field_map(
        party_id=payload["party_id"], schema_epoch=payload["schema_epoch"],
        field_assignments=tuple(
            (row["view_name"], row["field_name"]) for row in rows
        ),
    )
    if payload["map_id"] != local_map.map_id:
        raise ValueError("declared map_id does not match semantic content")
    if local_map.party_id != expected_party_id:
        raise ValueError("local map party does not match expected_party_id")
    if local_map.schema_epoch != expected_epoch:
        raise ValueError("local map epoch does not match expected_epoch")
    if local_map.map_id != expected_map_id:
        raise ValueError("local map identity does not match expected_map_id")
    return local_map
