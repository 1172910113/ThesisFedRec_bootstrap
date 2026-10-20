"""Deterministic public bases for Topic 1.2 v12 §18.6 (U4-r1).

Saved float32 coordinates and their checksum are authoritative. This module
constructs/validates numerical records, with no distribution or persistence.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import platform
import sys
import unicodedata

import numpy as np
import torch


_FORMAT = "thesisfedrec.public_basis"
_POLICIES = ("public_E_pivot_mgs_v1", "public_random_mgs_v1")
_RNG = "numpy.random.Generator(PCG64).standard_normal(float64)"
_ENV_KEYS = ("python", "numpy", "torch", "machine", "byteorder")


@dataclass(frozen=True, slots=True)
class PublicBasisRecord:
    format: str
    format_version: int
    policy: str
    policy_version: int
    n: int
    d: int
    p: int
    basis_seed: int
    construction_dtype: str
    delivery_dtype: str
    rng_implementation: str
    environment: tuple[tuple[str, str], ...]
    rank_threshold: float | None
    selected_pivots: tuple[int, ...]
    random_attempts: int
    random_acceptances: int
    coordinate_fallback: tuple[int, ...]
    public_embedding_id: str
    catalogue_id: str
    embedding_dtype: str
    embedding_fingerprint: str
    basis_sha256: str
    numerical_payload_bytes: int
    binding_id: str


@dataclass(frozen=True, slots=True)
class PublicBasis:
    """Owned B plus immutable scalar/tuple record; B itself remains mutable."""

    basis: torch.Tensor
    record: PublicBasisRecord


def _canonical(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _binding(record: PublicBasisRecord) -> str:
    payload = asdict(record)
    del payload["binding_id"]
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _text(value: str, name: str) -> None:
    if (
        type(value) is not str or not value or value.strip() != value
        or any(unicodedata.category(char) == "Cc" for char in value)
    ):
        raise ValueError(f"{name} must be a nonblank identity without outer whitespace/controls")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8 text") from error


def _digest(value: str, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _inputs(
    item_embeddings: torch.Tensor, p: int, basis_seed: int, policy: str,
    public_embedding_id: str, catalogue_id: str,
) -> np.ndarray:
    if (
        not isinstance(item_embeddings, torch.Tensor)
        or item_embeddings.layout != torch.strided
        or item_embeddings.device.type != "cpu"
        or item_embeddings.dtype not in (torch.float32, torch.float64)
        or item_embeddings.requires_grad
        or item_embeddings.ndim != 2
        or min(item_embeddings.shape) < 1
        or not torch.isfinite(item_embeddings).all().item()
    ):
        raise ValueError("E must be frozen finite dense strided CPU float32/float64 [n,d], n,d>0")
    if type(p) is not int or not 1 <= p <= item_embeddings.shape[0]:
        raise ValueError("p must be an integer in [1,n], excluding bool")
    if type(basis_seed) is not int or basis_seed < 0:
        raise ValueError("basis_seed must be a nonnegative integer, excluding bool")
    if type(policy) is not str or policy not in _POLICIES:
        raise ValueError("unsupported public basis policy")
    _text(public_embedding_id, "public_embedding_id")
    _text(catalogue_id, "catalogue_id")
    return item_embeddings.resolve_conj().resolve_neg().numpy()


def _array_hash(array: np.ndarray, dtype: str) -> str:
    # Feed a buffer, avoiding a second full-size Python bytes copy.
    canonical = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def _embedding_fingerprint(array: np.ndarray) -> str:
    dtype = "float32" if array.dtype == np.float32 else "float64"
    header = {
        "format": "thesisfedrec.public_embedding", "version": 1,
        "shape": list(array.shape), "dtype": dtype,
        "sha256": _array_hash(array, "<f4" if dtype == "float32" else "<f8"),
    }
    return hashlib.sha256(_canonical(header)).hexdigest()


def _norm(value: np.ndarray) -> float:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm):
        raise ValueError("nonfinite norm in public basis arithmetic")
    return norm


def _residual(candidate: np.ndarray, q: np.ndarray, count: int) -> np.ndarray:
    residual = np.array(candidate, dtype=np.float64, copy=True)
    for _ in range(2):
        for column in range(count):
            coefficient = float(np.dot(q[:, column], residual))
            if not math.isfinite(coefficient):
                raise ValueError("nonfinite MGS projection")
            residual -= coefficient * q[:, column]
    if not np.isfinite(residual).all():
        raise ValueError("nonfinite MGS residual")
    return residual


def _append(q: np.ndarray, count: int, residual: np.ndarray, norm: float) -> None:
    residual /= norm
    # argmax returns the first exact tie in public item-row order.
    pivot = int(np.argmax(np.abs(residual)))
    if residual[pivot] < 0:
        residual *= -1
    if not np.isfinite(residual).all():
        raise ValueError("nonfinite normalized basis vector")
    q[:, count] = residual


def _pivot_columns(e: np.ndarray, q: np.ndarray, tau: float) -> tuple[int, ...]:
    pivots: list[int] = []
    while len(pivots) < q.shape[1]:
        norms = {
            j: _norm(_residual(e[:, j], q, len(pivots)))
            for j in range(e.shape[1]) if j not in pivots
        }
        maximum = max(norms.values(), default=0.0)
        if maximum <= tau:
            break
        selected = min(j for j, norm in norms.items() if norm > tau and maximum - norm <= tau)
        residual = _residual(e[:, selected], q, len(pivots))
        _append(q, len(pivots), residual, _norm(residual))
        pivots.append(selected)
    return tuple(pivots)


def _complete(q: np.ndarray, count: int, seed: int) -> tuple[int, int, tuple[int, ...]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    attempts, accepted = 0, 0
    for _ in range(2 * q.shape[1]):
        if count == q.shape[1]:
            break
        candidate = rng.standard_normal(q.shape[0], dtype=np.float64)
        original_norm = _norm(candidate)
        residual = _residual(candidate, q, count)
        norm = _norm(residual)
        attempts += 1
        if norm <= 1e-12 * max(1.0, original_norm):
            continue
        _append(q, count, residual, norm)
        count += 1
        accepted += 1
    coordinates: list[int] = []
    for row in range(q.shape[0]):
        if count == q.shape[1]:
            break
        candidate = np.zeros(q.shape[0], dtype=np.float64)
        candidate[row] = 1
        residual = _residual(candidate, q, count)
        norm = _norm(residual)
        if norm <= 1e-12:  # original standard-coordinate norm is exactly one
            continue
        _append(q, count, residual, norm)
        coordinates.append(row)
        count += 1
    if count != q.shape[1]:
        raise ValueError("coordinate completion could not produce all p columns")
    return attempts, accepted, tuple(coordinates)


def _delivery(basis: torch.Tensor, n: int, p: int) -> None:
    if (
        not isinstance(basis, torch.Tensor) or basis.layout != torch.strided
        or basis.device.type != "cpu" or basis.dtype != torch.float32
        or basis.requires_grad or basis.shape != (n, p)
        or not basis.is_contiguous() or not torch.isfinite(basis).all().item()
    ):
        raise ValueError("B must be frozen finite contiguous CPU float32 [n,p]")
    # Check the actual delivered float32 coordinates, accumulating in float64.
    values = basis.numpy().astype(np.float64)
    gram = values.T @ values
    gram[np.diag_indices(p)] -= 1
    if _norm(gram) > 1e-5 * max(1, p):
        raise ValueError("delivered B fails Frobenius orthogonality tolerance")


def build_public_basis(
    item_embeddings: torch.Tensor, *, p: int, basis_seed: int, policy: str,
    public_embedding_id: str, catalogue_id: str,
) -> PublicBasis:
    """Construct the frozen two-pass pivot/completion policy in float64.

    E is read only and never retained. All parameters are explicit; no global
    RNG state is touched. See README for byte formats and ownership/lifecycle.
    """
    array = _inputs(item_embeddings, p, basis_seed, policy, public_embedding_id, catalogue_id)
    n, d = array.shape
    fingerprint = _embedding_fingerprint(array)
    q = np.empty((n, p), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            tau = None
            pivots: tuple[int, ...] = ()
            if policy == _POLICIES[0]:
                e = array.astype(np.float64, copy=False)
                tau = 1e-12 * max(1.0, _norm(e))
                pivots = _pivot_columns(e, q, tau)
            attempts, acceptances, coordinates = _complete(q, len(pivots), basis_seed)
            basis = torch.from_numpy(q).to(torch.float32)
            _delivery(basis, n, p)
    except FloatingPointError as error:
        raise ValueError("nonfinite public basis construction arithmetic") from error
    environment = (
        ("python", platform.python_version()), ("numpy", np.__version__),
        ("torch", str(torch.__version__)), ("machine", platform.machine() or "unknown"),
        ("byteorder", sys.byteorder),
    )
    record = PublicBasisRecord(
        format=_FORMAT, format_version=1, policy=policy, policy_version=1,
        n=n, d=d, p=p, basis_seed=basis_seed,
        construction_dtype="float64", delivery_dtype="float32",
        rng_implementation=_RNG, environment=environment, rank_threshold=tau,
        selected_pivots=pivots, random_attempts=attempts,
        random_acceptances=acceptances, coordinate_fallback=coordinates,
        public_embedding_id=public_embedding_id, catalogue_id=catalogue_id,
        embedding_dtype=str(array.dtype), embedding_fingerprint=fingerprint,
        basis_sha256=_array_hash(basis.numpy(), "<f4"),
        numerical_payload_bytes=4 * n * p, binding_id="",
    )
    return PublicBasis(basis, replace(record, binding_id=_binding(record)))


def _indices(value: tuple[int, ...], limit: int, name: str) -> None:
    if (
        type(value) is not tuple
        or any(type(index) is not int or not 0 <= index < limit for index in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"malformed {name}")


def _record_schema(record: PublicBasisRecord) -> None:
    if type(record) is not PublicBasisRecord:
        raise ValueError("record must be PublicBasisRecord")
    for name in ("format", "policy", "construction_dtype", "delivery_dtype", "rng_implementation", "embedding_dtype"):
        _text(getattr(record, name), name)
    if (
        record.format != _FORMAT or record.policy not in _POLICIES
        or record.construction_dtype != "float64" or record.delivery_dtype != "float32"
        or record.rng_implementation != _RNG
        or record.embedding_dtype not in ("float32", "float64")
    ):
        raise ValueError("unsupported record format, policy or numerical implementation")
    for name in ("format_version", "policy_version", "n", "d", "p", "basis_seed", "random_attempts", "random_acceptances", "numerical_payload_bytes"):
        if type(getattr(record, name)) is not int:
            raise ValueError(f"record {name} must be an integer, excluding bool")
    if (
        record.format_version != 1 or record.policy_version != 1
        or record.n < 1 or record.d < 1 or not 1 <= record.p <= record.n
        or record.basis_seed < 0
        or not 0 <= record.random_acceptances <= record.random_attempts <= 2 * record.p
        or record.numerical_payload_bytes != 4 * record.n * record.p
    ):
        raise ValueError("malformed record dimensions, versions or completion counts")
    _indices(record.selected_pivots, record.d, "selected_pivots")
    _indices(record.coordinate_fallback, record.n, "coordinate_fallback")
    if (
        tuple(sorted(record.coordinate_fallback)) != record.coordinate_fallback
        or len(record.selected_pivots) + record.random_acceptances + len(record.coordinate_fallback) != record.p
        or (record.coordinate_fallback and record.random_attempts != 2 * record.p)
        or (len(record.selected_pivots) == record.p and record.random_attempts != 0)
    ):
        raise ValueError("inconsistent completion record")
    if record.policy == _POLICIES[0]:
        if type(record.rank_threshold) is not float or not math.isfinite(record.rank_threshold) or record.rank_threshold < 1e-12:
            raise ValueError("malformed E rank threshold")
    elif record.rank_threshold is not None or record.selected_pivots:
        raise ValueError("random policy cannot contain E pivots/rank threshold")
    if type(record.environment) is not tuple or len(record.environment) != len(_ENV_KEYS):
        raise ValueError("malformed numerical environment")
    for pair, key in zip(record.environment, _ENV_KEYS):
        if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str or pair[0] != key:
            raise ValueError("malformed numerical environment entry")
        _text(pair[1], key)
    _text(record.public_embedding_id, "public_embedding_id")
    _text(record.catalogue_id, "catalogue_id")
    for name in ("embedding_fingerprint", "basis_sha256", "binding_id"):
        _digest(getattr(record, name), name)


def validate_public_basis(
    result: PublicBasis, *, item_embeddings: torch.Tensor, p: int,
    basis_seed: int, policy: str, public_embedding_id: str, catalogue_id: str,
    expected_binding_id: str,
) -> None:
    """Validate supplied bytes/configuration against explicit expected identity.

    Never regenerate/reorthogonalize B. Caller IDs assert provenance; expected
    binding must come from trusted caller bookkeeping, not the received record.
    """
    array = _inputs(item_embeddings, p, basis_seed, policy, public_embedding_id, catalogue_id)
    _digest(expected_binding_id, "expected_binding_id")
    if type(result) is not PublicBasis:
        raise ValueError("result must be PublicBasis")
    record = result.record
    _record_schema(record)
    if (
        (record.n, record.d) != array.shape or record.p != p
        or record.basis_seed != basis_seed or record.policy != policy
        or record.public_embedding_id != public_embedding_id
        or record.catalogue_id != catalogue_id
        or record.embedding_dtype != str(array.dtype)
        or record.embedding_fingerprint != _embedding_fingerprint(array)
    ):
        raise ValueError("public basis does not match expected E/catalogue/configuration")
    if record.binding_id != expected_binding_id or record.binding_id != _binding(record):
        raise ValueError("public basis binding mismatch")
    _delivery(result.basis, record.n, record.p)
    if _array_hash(result.basis.numpy(), "<f4") != record.basis_sha256:
        raise ValueError("public basis bytes have changed")
