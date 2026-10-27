"""Owning, versioned handoff of already-produced FedGMF terminal values.

The identities are trusted local consistency bindings, not signatures or proof of
training history. No candidate, metric, optimizer, random state, or fit object is
retained. Callers must keep source state stable during capture.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.federation.runner import FedGMFHFLFitResult, FedGMFHFLRoundSummary
from src.models.fedgmf import FedGMFLocalState


_MAX_INT = 2**63 - 1
_CONTEXT_FORMAT = "thesisfedrec.fedgmf_warmup.context"
_SNAPSHOT_FORMAT = "thesisfedrec.fedgmf_warmup.snapshot"
_FILE_FORMAT = "thesisfedrec.fedgmf_warmup.file"
_SHARED_KEYS = {"item_embeddings", "prediction_weight", "prediction_bias"}
_CONTEXT_KEYS = {
    "format", "version", "data_binding", "partition_binding", "profile_binding",
    "code_revision", "num_items", "num_users", "embedding_dim", "dtype",
    "party_user_ids",
}
_SNAPSHOT_KEYS = {
    "format", "version", "context_id", "terminal_round_count",
    "round_summaries", "buffers",
}
_FIELDS = (
    "protocol_version", "context_id", "snapshot_id", "data_binding",
    "partition_binding", "profile_binding", "code_revision", "num_items",
    "num_users", "embedding_dim", "dtype", "party_user_ids",
    "terminal_round_count", "round_summaries", "buffers",
)
_NUMPY_DTYPES = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8"),
                 "int64": np.dtype("<i8")}
_TORCH_DTYPES = {"float32": torch.float32, "float64": torch.float64,
                 "int64": torch.int64}


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class FedGMFWarmupSnapshot:
    """Factory-only immutable values; save/materialize revalidate every field.

    Python hostile reflection is not a security boundary. All retained values
    are exact built-in immutable scalars, recursively immutable tuples or bytes.
    """

    protocol_version: int
    context_id: str
    snapshot_id: str
    data_binding: str
    partition_binding: str
    profile_binding: str
    code_revision: str
    num_items: int
    num_users: int
    embedding_dim: int
    dtype: str
    party_user_ids: tuple[tuple[int, tuple[int, ...]], ...]
    terminal_round_count: int
    round_summaries: tuple[tuple[int, tuple[int, ...], int], ...]
    buffers: tuple[tuple[str, str, tuple[int, ...], int, str, bytes], ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use capture_fedgmf_warmup or load_fedgmf_warmup")


def _integer(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in int")
    if not minimum <= value <= _MAX_INT:
        raise ValueError(f"{name} is outside its supported integer range")


def _hex(value: object, name: str, length: int = 64) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase {length}-character hex")


def _literal(value: object, expected: str, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if value != expected:
        raise ValueError(f"unsupported {name}")


def _version(value: object) -> None:
    _integer(value, "version")
    if value != 1:
        raise ValueError("unsupported version")


def _dtype(value: object, *, floating: bool = False) -> None:
    if type(value) is not str:
        raise TypeError("dtype must be an exact built-in str")
    supported = ("float32", "float64") if floating else _NUMPY_DTYPES
    if value not in supported:
        raise ValueError("unsupported dtype")


def _keys(value: object, expected: set[str], name: str) -> None:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be exact built-in strings")
    if set(value) != expected:
        raise ValueError(f"{name} has missing or extra keys")


def _ownership(value: object, num_users: int) -> None:
    if type(value) is not tuple:
        raise TypeError("party_user_ids must be an exact tuple")
    if not value:
        raise ValueError("party_user_ids must not be empty")
    total = 0
    for row in value:
        if type(row) is not tuple:
            raise TypeError("ownership rows must be exact tuples")
        if len(row) != 2:
            raise ValueError("ownership rows must have two members")
        party, users = row
        _integer(party, "party_id")
        if type(users) is not tuple:
            raise TypeError("owned user IDs must be an exact tuple")
        if not users:
            raise ValueError("each party must own users")
        total += len(users)
    # Compare actual declaration size before constructing any union or range.
    if total != num_users:
        raise ValueError("ownership size must equal num_users")
    seen: set[int] = set()
    previous_party = -1
    for party, users in value:
        if party <= previous_party:
            raise ValueError("party IDs must be strictly increasing")
        previous_party = party
        previous_user = -1
        for user in users:
            _integer(user, "user_id")
            if user >= num_users or user <= previous_user or user in seen:
                raise ValueError("ownership must uniquely cover the full user domain")
            previous_user = user
            seen.add(user)


def _rounds(value: object, count: int, ownership: tuple) -> None:
    if type(value) is not tuple:
        raise TypeError("round_summaries must be an exact tuple")
    _integer(count, "terminal_round_count", minimum=1)
    if len(value) != count:
        raise ValueError("terminal_round_count does not match round summaries")
    roster = {party for party, _ in ownership}
    for index, row in enumerate(value):
        if type(row) is not tuple:
            raise TypeError("round rows must be exact tuples")
        if len(row) != 3:
            raise ValueError("round rows must have three members")
        round_index, selected, positives = row
        _integer(round_index, "round_index")
        if round_index != index:
            raise ValueError("round indices must be contiguous from zero")
        if type(selected) is not tuple:
            raise TypeError("selected_party_ids must be an exact tuple")
        if not selected:
            raise ValueError("selected_party_ids must not be empty")
        previous = -1
        for party in selected:
            _integer(party, "selected party_id")
            if party <= previous or party not in roster:
                raise ValueError("selected parties must be sorted members of the roster")
            previous = party
        _integer(positives, "total_positive_interactions", minimum=1)


def _nbytes(shape: tuple, dtype: str) -> int:
    if type(shape) is not tuple:
        raise TypeError("shape must be an exact tuple")
    numel = 1
    for dimension in shape:
        _integer(dimension, "shape dimension", minimum=1)
        numel *= dimension
        if numel > _MAX_INT:
            raise ValueError("buffer element capacity exceeded")
    nbytes = numel * _NUMPY_DTYPES[dtype].itemsize
    if nbytes > _MAX_INT:
        raise ValueError("buffer byte capacity exceeded")
    return nbytes


def _specs(num_items: int, dimension: int, dtype: str, ownership: tuple) -> tuple:
    rows = [
        ("item_embeddings", dtype, (num_items, dimension)),
        ("score_weight", dtype, (dimension,)),
        ("score_bias", dtype, ()),
    ]
    for party, users in ownership:
        rows.extend(((f"party.{party}.user_ids", "int64", (len(users),)),
                     (f"party.{party}.user_embeddings", dtype,
                      (len(users), dimension))))
    return tuple((name, kind, shape, _nbytes(shape, kind))
                 for name, kind, shape in rows)


def _cjson(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _context(data_binding: str, partition_binding: str, profile_binding: str,
             code_revision: str, num_items: int, num_users: int,
             embedding_dim: int, dtype: str, party_user_ids: tuple) -> dict:
    return {
        "format": _CONTEXT_FORMAT, "version": 1,
        "data_binding": data_binding, "partition_binding": partition_binding,
        "profile_binding": profile_binding, "code_revision": code_revision,
        "num_items": num_items, "num_users": num_users,
        "embedding_dim": embedding_dim, "dtype": dtype,
        "party_user_ids": [{"party_id": party, "user_ids": list(users)}
                           for party, users in party_user_ids],
    }


def _snapshot_object(context_id: str, count: int, rounds: tuple,
                     buffers: tuple) -> dict:
    return {
        "format": _SNAPSHOT_FORMAT, "version": 1, "context_id": context_id,
        "terminal_round_count": count,
        "round_summaries": [
            {"round_index": index, "selected_party_ids": list(selected),
             "total_positive_interactions": positives}
            for index, selected, positives in rounds
        ],
        "buffers": [{"name": name, "dtype": dtype, "shape": list(shape),
                     "nbytes": nbytes, "sha256": digest}
                    for name, dtype, shape, nbytes, digest, _ in buffers],
    }


def _validate_context_values(data_binding: object, partition_binding: object,
                             profile_binding: object, code_revision: object,
                             num_items: object, num_users: object,
                             dimension: object, dtype: object,
                             ownership: object) -> tuple:
    for name, value in (("data_binding", data_binding),
                        ("partition_binding", partition_binding),
                        ("profile_binding", profile_binding)):
        _hex(value, name)
    _hex(code_revision, "code_revision", 40)
    _integer(num_items, "num_items", minimum=1)
    _integer(num_users, "num_users", minimum=1)
    _integer(dimension, "embedding_dim", minimum=1)
    _dtype(dtype, floating=True)
    _ownership(ownership, num_users)
    return _specs(num_items, dimension, dtype, ownership)


def _buffer_content(buffers: tuple, ownership: tuple) -> None:
    for name, dtype, _, _, digest, raw in buffers:
        if _sha(raw) != digest:
            raise ValueError(f"buffer hash mismatch: {name}")
        if dtype != "int64" and not np.isfinite(
            np.frombuffer(raw, dtype=_NUMPY_DTYPES[dtype])
        ).all():
            raise ValueError(f"buffer must be finite: {name}")
    for index, (_, users) in enumerate(ownership):
        raw = buffers[3 + 2 * index][5]
        decoded = np.frombuffer(raw, dtype=_NUMPY_DTYPES["int64"])
        if any(int(actual) != expected for actual, expected in zip(decoded, users)):
            raise ValueError("user ID buffer disagrees with ownership")


def _validate_snapshot(snapshot: object) -> tuple[dict, dict]:
    if type(snapshot) is not FedGMFWarmupSnapshot:
        raise TypeError("snapshot must be an exact FedGMFWarmupSnapshot")
    try:
        values = tuple(getattr(snapshot, field) for field in _FIELDS)
    except AttributeError as error:
        raise TypeError("snapshot has missing fields") from error
    (version, context_id, snapshot_id, data, partition, profile, revision,
     items, users, dimension, dtype, ownership, count, rounds, buffers) = values
    _version(version)
    _hex(context_id, "context_id")
    _hex(snapshot_id, "snapshot_id")
    specs = _validate_context_values(data, partition, profile, revision, items,
                                     users, dimension, dtype, ownership)
    _rounds(rounds, count, ownership)
    if type(buffers) is not tuple:
        raise TypeError("buffers must be an exact tuple")
    if len(buffers) != len(specs):
        raise ValueError("wrong buffer count")
    for row, spec in zip(buffers, specs):
        if type(row) is not tuple:
            raise TypeError("buffer rows must be exact tuples")
        if len(row) != 6:
            raise ValueError("buffer rows must have six members")
        name, kind, shape, nbytes, digest, raw = row
        _literal(name, spec[0], "buffer name")
        _dtype(kind)
        if kind != spec[1]:
            raise ValueError("buffer dtype disagrees with schema")
        if type(shape) is not tuple:
            raise TypeError("buffer shape must be an exact tuple")
        _nbytes(shape, kind)
        if shape != spec[2]:
            raise ValueError("buffer shape disagrees with schema")
        _integer(nbytes, "buffer nbytes")
        if nbytes != spec[3]:
            raise ValueError("buffer nbytes disagrees with shape")
        _hex(digest, "buffer sha256")
        if type(raw) is not bytes:
            raise TypeError("raw buffer must be exact bytes")
        if len(raw) != nbytes:
            raise ValueError("raw buffer length disagrees with nbytes")
    context = _context(data, partition, profile, revision, items, users,
                       dimension, dtype, ownership)
    if _sha(_cjson(context)) != context_id:
        raise ValueError("context_id does not match snapshot context")
    summary = _snapshot_object(context_id, count, rounds, buffers)
    if _sha(_cjson(summary)) != snapshot_id:
        raise ValueError("snapshot_id does not match snapshot structure")
    _buffer_content(buffers, ownership)
    return context, summary


def _make_snapshot(data: str, partition: str, profile: str, revision: str,
                   items: int, users: int, dimension: int, dtype: str,
                   ownership: tuple, rounds: tuple, buffers: tuple) -> FedGMFWarmupSnapshot:
    context_id = _sha(_cjson(_context(data, partition, profile, revision, items,
                                     users, dimension, dtype, ownership)))
    count = len(rounds)
    snapshot_id = _sha(_cjson(_snapshot_object(context_id, count, rounds, buffers)))
    snapshot = object.__new__(FedGMFWarmupSnapshot)
    values = (1, context_id, snapshot_id, data, partition, profile, revision,
              items, users, dimension, dtype, ownership, count, rounds, buffers)
    for field, value in zip(_FIELDS, values):
        object.__setattr__(snapshot, field, value)
    _validate_snapshot(snapshot)
    return snapshot


def _tensor_header(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.is_nested or value.layout != torch.strided:
        raise ValueError(f"{name} must be a non-nested strided tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")


def _tensor(value: object, name: str, dtype: str, shape: tuple) -> None:
    _tensor_header(value, name)
    if value.dtype != _TORCH_DTYPES[dtype]:
        raise TypeError(f"{name} has an unsupported or inconsistent dtype")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has the wrong original shape")


def _raw_tensor(value: torch.Tensor, dtype: str) -> bytes:
    # Resolve legal view bits without changing caller storage or autograd state.
    array = value.detach().resolve_conj().resolve_neg().contiguous().numpy()
    return array.astype(_NUMPY_DTYPES[dtype], copy=False).tobytes(order="C")


def capture_fedgmf_warmup(
    fit_result: FedGMFHFLFitResult, *, data_binding: str, partition_binding: str,
    profile_binding: str, code_revision: str, expected_num_items: int,
    expected_num_users: int,
    expected_party_user_ids: tuple[tuple[int, tuple[int, ...]], ...],
) -> FedGMFWarmupSnapshot:
    """Copy validated terminal values; candidates and metrics are never consumed."""
    if type(fit_result) is not FedGMFHFLFitResult:
        raise TypeError("fit_result must be an exact FedGMFHFLFitResult")
    for name, value in (("data_binding", data_binding),
                        ("partition_binding", partition_binding),
                        ("profile_binding", profile_binding)):
        _hex(value, name)
    _hex(code_revision, "code_revision", 40)
    _integer(expected_num_items, "expected_num_items", minimum=1)
    _integer(expected_num_users, "expected_num_users", minimum=1)
    _ownership(expected_party_user_ids, expected_num_users)
    shared = fit_result.final_shared_state
    _keys(shared, _SHARED_KEYS, "final_shared_state")
    local = fit_result.final_local_states
    if type(local) is not dict:
        raise TypeError("final_local_states must be an exact dict")
    for party in local:
        _integer(party, "local party_id")
    if set(local) != {party for party, _ in expected_party_user_ids}:
        raise ValueError("local states do not match the declared roster")
    E = shared["item_embeddings"]
    _tensor_header(E, "item_embeddings")
    if E.dtype not in (torch.float32, torch.float64):
        raise TypeError("item_embeddings must be float32 or float64")
    if E.ndim != 2 or E.shape[0] != expected_num_items:
        raise ValueError("item_embeddings must have original shape [I,d]")
    dimension = E.shape[1]
    _integer(dimension, "embedding_dim", minimum=1)
    dtype = "float32" if E.dtype == torch.float32 else "float64"
    specs = _specs(expected_num_items, dimension, dtype, expected_party_user_ids)
    _tensor(E, "item_embeddings", dtype, (expected_num_items, dimension))
    weight, bias = shared["prediction_weight"], shared["prediction_bias"]
    _tensor(weight, "prediction_weight", dtype, (1, dimension))
    _tensor(bias, "prediction_bias", dtype, (1,))
    floating = [E, weight, bias]
    local_tensors = []
    for party, users in expected_party_user_ids:
        state = local[party]
        if type(state) is not FedGMFLocalState:
            raise TypeError("local states must be exact FedGMFLocalState records")
        _tensor(state.user_ids, "local user_ids", "int64", (len(users),))
        _tensor(state.user_embeddings, "local user_embeddings", dtype,
                (len(users), dimension))
        local_tensors.extend((state.user_ids, state.user_embeddings))
        floating.append(state.user_embeddings)
    summaries = fit_result.round_summaries
    if type(summaries) is not tuple:
        raise TypeError("round_summaries must be an exact tuple")
    rows = []
    for summary in summaries:
        if type(summary) is not FedGMFHFLRoundSummary:
            raise TypeError("summaries must be exact FedGMFHFLRoundSummary records")
        rows.append((summary.round_index, summary.selected_party_ids,
                     summary.total_positive_interactions))
    rounds = tuple(rows)
    _rounds(rounds, len(rounds), expected_party_user_ids)
    # Every source shape and Python integer capacity has now been checked.
    for value in floating:
        if not torch.isfinite(value).all().item():
            raise ValueError("all floating source values must be finite")
    for index, (_, users) in enumerate(expected_party_user_ids):
        if tuple(local_tensors[2 * index].detach().tolist()) != users:
            raise ValueError("local user IDs disagree with declared row ownership")
    tensors = [E, weight[0], bias[0], *local_tensors]
    buffers = []
    for (name, kind, shape, nbytes), value in zip(specs, tensors):
        raw = _raw_tensor(value, kind)
        buffers.append((name, kind, shape, nbytes, _sha(raw), raw))
    return _make_snapshot(data_binding, partition_binding, profile_binding,
                          code_revision, expected_num_items, expected_num_users,
                          dimension, dtype, expected_party_user_ids, rounds,
                          tuple(buffers))


def materialize_fedgmf_warmup(
    snapshot: FedGMFWarmupSnapshot,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, FedGMFLocalState]]:
    """Return independent, owning, contiguous CPU values without changing RNG."""
    _validate_snapshot(snapshot)
    values = []
    with torch.no_grad():
        for _, dtype, shape, _, _, raw in snapshot.buffers:
            # Native-endian copy plus clone keeps the delivered tensor independent
            # of bytes, temporary arrays, other buffers, and previous deliveries.
            array = np.frombuffer(raw, dtype=_NUMPY_DTYPES[dtype]).reshape(shape)
            native = array.astype(np.dtype(dtype), copy=True)
            values.append(torch.from_numpy(native).clone())
    local = {}
    for index, (party, _) in enumerate(snapshot.party_user_ids):
        local[party] = FedGMFLocalState(user_ids=values[3 + 2 * index],
                                      user_embeddings=values[4 + 2 * index])
    return values[0], values[1], values[2], local


def _path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be str or pathlib.Path")
    return Path(path)


def save_fedgmf_warmup(snapshot: FedGMFWarmupSnapshot, path: str | Path) -> None:
    """Publish canonical JSON with a same-parent atomic no-overwrite hard link."""
    context, summary = _validate_snapshot(snapshot)
    payload = {
        "format": _FILE_FORMAT, "version": 1,
        "context_id": snapshot.context_id, "snapshot_id": snapshot.snapshot_id,
        "context": context, "snapshot": summary,
        "tensors": [{"name": name,
                     "data": base64.b64encode(raw).decode("ascii")}
                    for name, _, _, _, _, raw in snapshot.buffers],
    }
    encoded = _cjson(payload) + b"\n"
    target = _path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.",
                                            suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if stream.write(encoded) != len(encoded):
                raise OSError("incomplete warmup artifact write")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _read_file(target: Path, cap: int) -> bytes:
    size = target.stat().st_size
    if size < 3 or size > cap:
        raise ValueError("file length is impossible or exceeds max_file_bytes")
    chunks = []
    length = 0
    # Never pass cap+1 (which can exceed Py_ssize_t) as a read size.
    limit = min(size + 1, cap + 1)
    with target.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != size:
            raise ValueError("file length changed before reading")
        while length < limit:
            chunk = stream.read(min(65536, limit - length))
            if not chunk:
                break
            chunks.append(chunk)
            length += len(chunk)
            if length > cap or length > size:
                raise ValueError("file length changed or exceeds max_file_bytes")
        if os.fstat(stream.fileno()).st_size != size:
            raise ValueError("file length changed while reading")
    if length != size:
        raise ValueError("file length disagrees with stat")
    return b"".join(chunks)


def _json_ownership(value: object) -> tuple:
    if type(value) is not list:
        raise TypeError("party_user_ids must be a JSON list")
    rows = []
    for row in value:
        _keys(row, {"party_id", "user_ids"}, "ownership row")
        if type(row["user_ids"]) is not list:
            raise TypeError("user_ids must be a JSON list")
        rows.append((row["party_id"], tuple(row["user_ids"])))
    return tuple(rows)


def _json_rounds(value: object) -> tuple:
    if type(value) is not list:
        raise TypeError("round_summaries must be a JSON list")
    rows = []
    for row in value:
        _keys(row, {"round_index", "selected_party_ids",
                    "total_positive_interactions"}, "round row")
        if type(row["selected_party_ids"]) is not list:
            raise TypeError("selected_party_ids must be a JSON list")
        rows.append((row["round_index"], tuple(row["selected_party_ids"]),
                     row["total_positive_interactions"]))
    return tuple(rows)


def _load_headers(payload: object, expected_context: str, expected_snapshot: str,
                  physical_size: int) -> tuple:
    _keys(payload, {"format", "version", "context_id", "snapshot_id", "context",
                    "snapshot", "tensors"}, "warmup file")
    _literal(payload["format"], _FILE_FORMAT, "file format")
    _version(payload["version"])
    _hex(payload["context_id"], "context_id")
    _hex(payload["snapshot_id"], "snapshot_id")
    if payload["context_id"] != expected_context or payload["snapshot_id"] != expected_snapshot:
        raise ValueError("file identities do not match caller expectations")
    context = payload["context"]
    _keys(context, _CONTEXT_KEYS, "context")
    _literal(context["format"], _CONTEXT_FORMAT, "context format")
    _version(context["version"])
    ownership = _json_ownership(context["party_user_ids"])
    arguments = (context["data_binding"], context["partition_binding"],
                 context["profile_binding"], context["code_revision"],
                 context["num_items"], context["num_users"],
                 context["embedding_dim"], context["dtype"], ownership)
    specs = _validate_context_values(*arguments)
    if _sha(_cjson(context)) != expected_context:
        raise ValueError("context digest mismatch")
    summary = payload["snapshot"]
    _keys(summary, _SNAPSHOT_KEYS, "snapshot")
    _literal(summary["format"], _SNAPSHOT_FORMAT, "snapshot format")
    _version(summary["version"])
    _hex(summary["context_id"], "snapshot context_id")
    if summary["context_id"] != expected_context:
        raise ValueError("snapshot context mismatch")
    rounds = _json_rounds(summary["round_summaries"])
    _rounds(rounds, summary["terminal_round_count"], ownership)
    descriptors, tensors = summary["buffers"], payload["tensors"]
    if type(descriptors) is not list or type(tensors) is not list:
        raise TypeError("buffers and tensors must be JSON lists")
    if len(descriptors) != len(specs) or len(tensors) != len(specs):
        raise ValueError("wrong descriptor or tensor count")
    headers = []
    total_bytes = 0
    for descriptor, tensor, spec in zip(descriptors, tensors, specs):
        _keys(descriptor, {"name", "dtype", "shape", "nbytes", "sha256"}, "descriptor")
        _literal(descriptor["name"], spec[0], "descriptor name")
        _dtype(descriptor["dtype"])
        if descriptor["dtype"] != spec[1]:
            raise ValueError("descriptor dtype mismatch")
        if type(descriptor["shape"]) is not list:
            raise TypeError("descriptor shape must be a JSON list")
        shape = tuple(descriptor["shape"])
        nbytes = _nbytes(shape, descriptor["dtype"])
        if shape != spec[2]:
            raise ValueError("descriptor shape mismatch")
        _integer(descriptor["nbytes"], "descriptor nbytes")
        if descriptor["nbytes"] != nbytes or nbytes != spec[3]:
            raise ValueError("descriptor nbytes mismatch")
        total_bytes += nbytes
        if nbytes > physical_size or total_bytes > physical_size:
            raise ValueError("declared raw buffers exceed physical file budget")
        _hex(descriptor["sha256"], "descriptor sha256")
        _keys(tensor, {"name", "data"}, "tensor")
        _literal(tensor["name"], spec[0], "tensor name")
        data = tensor["data"]
        if type(data) is not str:
            raise TypeError("tensor data must be an exact string")
        if not data.isascii() or len(data) != 4 * ((nbytes + 2) // 3):
            raise ValueError("base64 encoded length or ASCII requirement failed")
        headers.append((*spec, descriptor["sha256"], data))
    if _sha(_cjson(summary)) != expected_snapshot:
        raise ValueError("snapshot digest mismatch")
    return arguments, rounds, tuple(headers)


def load_fedgmf_warmup(
    path: str | Path, *, expected_context_id: str, expected_snapshot_id: str,
    max_file_bytes: int,
) -> FedGMFWarmupSnapshot:
    """Read one bounded canonical artifact and verify all expected identities."""
    target = _path(path)
    _hex(expected_context_id, "expected_context_id")
    _hex(expected_snapshot_id, "expected_snapshot_id")
    _integer(max_file_bytes, "max_file_bytes", minimum=1)
    raw_file = _read_file(target, max_file_bytes)
    payload = json.loads(raw_file.decode("utf-8"), object_pairs_hook=_unique_object,
                         parse_constant=_invalid_constant)
    if raw_file != _cjson(payload) + b"\n":
        raise ValueError("artifact must be canonical JSON followed by one LF")
    arguments, rounds, headers = _load_headers(payload, expected_context_id,
                                               expected_snapshot_id, len(raw_file))
    # All metadata, capacities and encoded lengths precede the first decode.
    buffers = []
    for name, dtype, shape, nbytes, digest, data in headers:
        raw = base64.b64decode(data, validate=True)
        if (len(raw) != nbytes or base64.b64encode(raw).decode("ascii") != data
                or _sha(raw) != digest):
            raise ValueError("noncanonical base64, byte length, or buffer hash")
        buffers.append((name, dtype, shape, nbytes, digest, raw))
    return _make_snapshot(*arguments, rounds, tuple(buffers))
