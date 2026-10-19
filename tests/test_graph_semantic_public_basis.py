"""Independent scalar reference and synthetic evidence for T12-BASIS-K0."""

from dataclasses import asdict, replace
import gc
import hashlib
import inspect
import json
import math
import random
import resource
import tracemalloc
import weakref

import numpy as np
import pytest
import torch

from src.methods.graph_semantic import (
    PublicBasis, PublicBasisRecord, build_public_basis, validate_public_basis,
    make_operator_sketch, prepare_path_factors, prepare_source_kernel,
)
from src.methods.graph_semantic import public_basis


PIVOT = "public_E_pivot_mgs_v1"
RANDOM = "public_random_mgs_v1"
POLICIES = (PIVOT, RANDOM)


def _kwargs(p=2, seed=37, policy=PIVOT):
    return dict(p=p, basis_seed=seed, policy=policy, public_embedding_id="public-E/v1", catalogue_id="catalogue/v1")


def _build(e, **changes):
    kwargs = _kwargs()
    kwargs.update(changes)
    return build_public_basis(e, **kwargs)


def _validate(result, e, **changes):
    record = result.record
    kwargs = dict(
        p=record.p, basis_seed=record.basis_seed, policy=record.policy,
        public_embedding_id=record.public_embedding_id, catalogue_id=record.catalogue_id,
        expected_binding_id=record.binding_id,
    )
    kwargs.update(changes)
    validate_public_basis(result, item_embeddings=e, **kwargs)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _checksum(basis):
    # Test-owned byte encoder deliberately uses explicit struct packing.
    import struct
    return hashlib.sha256(b"".join(struct.pack("<f", float(x)) for x in basis.reshape(-1))).hexdigest()


def _rebind(result, basis=None, **changes):
    if basis is None:
        basis = result.basis
    record = replace(result.record, basis_sha256=_checksum(basis), **changes)
    payload = asdict(record)
    payload.pop("binding_id")
    record = replace(record, binding_id=hashlib.sha256(_canonical(payload)).hexdigest())
    return PublicBasis(basis, record)


def _reference(e, p, seed, policy, candidates=None):
    """Scalar/list two-pass MGS, with a separate actual-PCG64 candidate stream.

    No production helpers, BLAS projections, matrix QR, or matrix SVD are used.
    """
    matrix = e.detach().numpy().tolist()
    n, d = e.shape
    q, pivots, coordinates = [], [], []
    norm = lambda v: math.sqrt(math.fsum(float(x) * float(x) for x in v))
    def residual(vector):
        r = list(map(float, vector))
        for _ in (0, 1):
            for column in q:
                coefficient = math.fsum(x * y for x, y in zip(column, r))
                r = [x - coefficient * y for x, y in zip(r, column)]
        return r
    def append(vector):
        magnitude = norm(vector)
        column = [x / magnitude for x in vector]
        row = max(range(n), key=lambda i: abs(column[i]))
        if column[row] < 0:
            column = [-x for x in column]
        q.append(column)
    tau = None
    if policy == PIVOT:
        tau = 1e-12 * max(1.0, norm([x for row in matrix for x in row]))
        while len(q) < p:
            remaining = [j for j in range(d) if j not in pivots]
            residuals = {j: residual([matrix[i][j] for i in range(n)]) for j in remaining}
            lengths = {j: norm(r) for j, r in residuals.items()}
            maximum = max(lengths.values(), default=0.0)
            eligible = [j for j in remaining if lengths[j] > tau and abs(lengths[j] - maximum) <= tau]
            if maximum <= tau:
                break
            selected = min(eligible)
            append(residuals[selected])
            pivots.append(selected)
    generator = np.random.Generator(np.random.PCG64(seed))
    stream = iter(candidates) if candidates is not None else None
    attempts, accepted = 0, 0
    while len(q) < p and attempts < 2 * p:
        vector = next(stream) if stream is not None else generator.standard_normal(n, dtype=np.float64)
        attempts += 1
        r = residual(vector)
        if norm(r) <= 1e-12 * max(1.0, norm(vector)):
            continue
        append(r)
        accepted += 1
    for row in range(n):
        if len(q) == p:
            break
        vector = [float(i == row) for i in range(n)]
        r = residual(vector)
        if norm(r) > 1e-12:
            append(r)
            coordinates.append(row)
    return np.array(q, dtype=np.float64).T.astype(np.float32), {
        "selected_pivots": tuple(pivots), "rank_threshold": tau,
        "random_attempts": attempts, "random_acceptances": accepted,
        "coordinate_fallback": tuple(coordinates),
    }


FIXTURES = [
    ([[1, 0, 0], [0, 3, 0], [0, 0, 2]], 1),  # p below rank; largest original column
    ([[1, 0, 0], [0, 3, 0], [0, 0, 2]], 2),
    ([[1, 0], [0, 1], [0, 0]], 2),  # equal norms
    ([[1, 2, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0]], 3),
    ([[0, 0], [0, 0], [0, 0]], 3),  # all-zero E and p=n
    ([[1], [2], [0], [-1], [3]], 4),  # rank deficiency; p>d
    ([[-1], [1]], 1),  # largest-coordinate sign tie
    ([[1, 1, 1], [0, 1e-7, 0], [0, 0, 1e-7], [0, 0, 0]], 3),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("values,p", FIXTURES)
def test_independent_scalar_reference(values, p, policy, dtype):
    e = torch.tensor(values, dtype=dtype)
    result = _build(e, p=p, policy=policy)
    expected, details = _reference(e, p, 37, policy)
    error = float(np.max(np.abs(result.basis.numpy() - expected)))
    np.testing.assert_allclose(result.basis.numpy(), expected, atol=2e-7, rtol=2e-7)
    for key, value in details.items():
        if key == "rank_threshold" and value is not None:
            assert result.record.rank_threshold == pytest.approx(value, rel=1e-14)
        else:
            assert getattr(result.record, key) == value
    _validate(result, e)
    assert result.basis.dtype == torch.float32 and result.basis.is_contiguous()
    assert not result.basis.requires_grad and result.basis.device.type == "cpu"
    print("BASIS_REFERENCE=" + json.dumps({"dtype": str(dtype), "policy": policy, "n": e.shape[0], "d": e.shape[1], "p": p, "max_absolute_error": error, "pivots": result.record.selected_pivots}))


def test_hand_checkable_unequal_norms_truncate_at_p():
    e = torch.diag(torch.tensor([1., 3., 2.], dtype=torch.float64))
    one = _build(e, p=1)
    two = _build(e, p=2)
    assert one.record.selected_pivots == (1,)
    assert two.record.selected_pivots == (1, 2)
    assert torch.equal(one.basis, torch.tensor([[0.], [1.], [0.]]))
    assert torch.equal(two.basis, torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))
    assert two.record.random_attempts == 0


@pytest.mark.parametrize("delta,expected", [(0.0, 0), (0.5e-12, 0), (4e-12, 1)])
def test_exact_and_near_equal_pivot_norm_tie_rule(delta, expected):
    e = torch.diag(torch.tensor([1., 1. + delta], dtype=torch.float64))
    result = _build(e, p=1)
    assert result.record.selected_pivots == (expected,)
    np.testing.assert_array_equal(result.basis.numpy(), _reference(e, 1, 37, PIVOT)[0])


def test_near_tie_candidate_must_exceed_rank_threshold():
    e = torch.diag(torch.tensor([0.5e-12, 1.1e-12], dtype=torch.float64))
    result = _build(e, p=1)
    assert result.record.rank_threshold == 1e-12
    assert result.record.selected_pivots == (1,)


@pytest.mark.parametrize("vector,expected", [([-1., 1.], [1 / math.sqrt(2), -1 / math.sqrt(2)]), ([0., -2., 2.], [0, 1 / math.sqrt(2), -1 / math.sqrt(2)])])
def test_sign_exact_ties_use_first_public_item_row(vector, expected):
    result = _build(torch.tensor(vector, dtype=torch.float64)[:, None], p=1)
    np.testing.assert_allclose(result.basis.numpy()[:, 0], expected, atol=1e-7)


def _controlled_rng(monkeypatch, vectors):
    calls = []
    stream = iter(vectors)
    class Generator:
        def standard_normal(self, size, dtype):
            calls.append((size, dtype))
            return np.array(next(stream), dtype=np.float64)
    def constructor(bit_generator):
        assert type(bit_generator) is np.random.PCG64
        return Generator()
    monkeypatch.setattr(public_basis.np.random, "Generator", constructor)
    return calls


def test_completion_total_attempt_cap_and_coordinate_fallback(monkeypatch):
    e = torch.tensor([[2., 1.], [0, 0], [0, 0], [0, 0]], dtype=torch.float64)
    vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
    expected, details = _reference(e, 3, 37, PIVOT, vectors)
    calls = _controlled_rng(monkeypatch, vectors)
    result = _build(e, p=3)
    assert calls == [(4, np.float64)] * 6
    assert result.record.random_attempts == details["random_attempts"] == 6
    assert result.record.random_acceptances == 1
    assert result.record.coordinate_fallback == (2,)
    np.testing.assert_array_equal(result.basis.numpy(), expected)
    np.testing.assert_array_equal(expected, np.eye(4, 3, dtype=np.float32))
    _validate(result, e)


def test_random_completion_cap_counts_rejections_and_stops_at_p(monkeypatch):
    e = torch.zeros((4, 1), dtype=torch.float64)
    calls = _controlled_rng(monkeypatch, [[0, 0, 0, 0]] * 4)
    result = _build(e, policy=RANDOM)
    assert len(calls) == 4
    assert result.record.random_attempts == 4 and result.record.random_acceptances == 0
    assert result.record.coordinate_fallback == (0, 1)
    np.testing.assert_array_equal(result.basis.numpy(), np.eye(4, 2, dtype=np.float32))


def test_completion_draw_order_and_early_stop(monkeypatch):
    e = torch.zeros((4, 1), dtype=torch.float64)
    calls = _controlled_rng(monkeypatch, [[0, -1, 0, 0], [0, 0, -1, 0]])
    result = _build(e, policy=RANDOM)
    assert len(calls) == 2
    assert result.record.random_attempts == result.record.random_acceptances == 2
    assert result.record.coordinate_fallback == ()
    np.testing.assert_array_equal(result.basis.numpy(), [[0, 0], [1, 0], [0, 1], [0, 0]])


def test_completion_rejection_scales_by_original_candidate_norm(monkeypatch):
    e = torch.tensor([[1.], [0], [0]], dtype=torch.float64)
    calls = _controlled_rng(monkeypatch, [[1e6, 1e-7, 0], [1e6, 3e-6, 0]])
    result = _build(e)
    assert len(calls) == 2 and result.record.random_acceptances == 1
    assert result.record.coordinate_fallback == ()
    np.testing.assert_array_equal(result.basis.numpy(), np.eye(3, 2, dtype=np.float32))


def test_actual_pcg64_candidate_order_and_signs():
    e = torch.zeros((5, 2), dtype=torch.float64)
    expected, details = _reference(e, 3, 19, RANDOM)
    result = _build(e, p=3, basis_seed=19, policy=RANDOM)
    np.testing.assert_array_equal(result.basis.numpy(), expected)
    assert result.record.random_attempts == details["random_attempts"] == 3
    assert result.record.rng_implementation == "numpy.random.Generator(PCG64).standard_normal(float64)"


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strided_inputs_and_exact_repeatability(policy, dtype):
    full = torch.arange(48, dtype=dtype).reshape(8, 6)
    e = full[::2, ::2]
    assert not e.is_contiguous()
    first = _build(e, policy=policy, p=3)
    second = _build(e.contiguous(), policy=policy, p=3)
    assert torch.equal(first.basis, second.basis)
    assert first.record == second.record
    _validate(first, e)
    # A real-valued negative-view bit still denotes a valid dense strided E.
    negative = torch._neg_view(e)
    assert _build(negative, policy=policy).record == _build(-e, policy=policy).record


@pytest.mark.parametrize("policy", POLICIES)
def test_global_rng_preservation_and_input_nonmutation(policy):
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    e = torch.tensor([[1., 2.], [3., 4.], [-1., 0.]], dtype=torch.float64)
    copy, pointer = e.clone(), e.data_ptr()
    result = _build(e, policy=policy)
    _validate(result, e)
    assert random.getstate() == py_state
    state_after = np.random.get_state()
    assert state_after[0] == np_state[0]
    np.testing.assert_array_equal(state_after[1], np_state[1])
    assert state_after[2:] == np_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.equal(e, copy) and not e.requires_grad and e.data_ptr() == pointer
    assert result.basis.data_ptr() != pointer and result.basis._base is None


def test_no_retained_E_storage_or_mutable_caller_aliases():
    e = torch.eye(3, dtype=torch.float32)
    reference, pointer = weakref.ref(e), e.data_ptr()
    result = _build(e)
    e.fill_(20)
    assert torch.equal(result.basis, torch.eye(3, 2))
    assert result.basis.data_ptr() != pointer
    del e
    gc.collect()
    assert reference() is None
    assert set(asdict(result.record)) == {field.name for field in __import__('dataclasses').fields(PublicBasisRecord)}


@pytest.mark.parametrize("e", [None, np.ones((3, 2)), torch.ones(3), torch.ones((2, 2, 2)), torch.ones((0, 2)), torch.ones((3, 0)), torch.ones((3, 2), dtype=torch.int64), torch.ones((3, 2), dtype=torch.float16), torch.ones((3, 2), requires_grad=True), torch.full((3, 2), float('nan')), torch.full((3, 2), float('inf')), torch.ones((3, 2)).to_sparse(), torch.empty((3, 2), device='meta')])
def test_reject_invalid_E(e):
    with pytest.raises(ValueError):
        _build(e)


@pytest.mark.parametrize("key,value", [
    ("p", True), ("p", 0), ("p", -1), ("p", 4), ("p", 1.0), ("p", np.int64(1)),
    ("basis_seed", False), ("basis_seed", -1), ("basis_seed", 1.0), ("basis_seed", np.int64(1)),
    ("policy", "qr"), ("policy", None), ("policy", []),
    ("public_embedding_id", ""), ("public_embedding_id", " "), ("public_embedding_id", " v1"),
    ("public_embedding_id", "v1\n"), ("public_embedding_id", "a\x00b"), ("public_embedding_id", "\ud800"),
    ("catalogue_id", None), ("catalogue_id", 123), ("catalogue_id", "v1 "),
])
def test_parameter_validation(key, value):
    with pytest.raises(ValueError):
        _build(torch.eye(3), **{key: value})


def test_large_nonnegative_seed_and_unicode_identity():
    e = torch.eye(3)
    result = _build(e, basis_seed=2**100 + 1, catalogue_id="公共目录/一", public_embedding_id="公共E/一")
    _validate(result, e)
    assert result.record.basis_seed == 2**100 + 1


def test_nonfinite_construction_arithmetic_is_rejected():
    e = torch.full((3, 2), 1e308, dtype=torch.float64)
    with pytest.raises(ValueError, match="nonfinite"):
        _build(e)
    # The random policy never uses E's norms/projections for direction selection.
    random_result = _build(e, policy=RANDOM)
    assert torch.isfinite(random_result.basis).all()


@pytest.mark.parametrize("candidate", [[1e308, 1e308, 1e308], [float('nan'), 0, 0]])
def test_nonfinite_completion_arithmetic_is_rejected(monkeypatch, candidate):
    _controlled_rng(monkeypatch, [candidate])
    with pytest.raises(ValueError, match="nonfinite"):
        _build(torch.zeros((3, 1)))


@pytest.mark.parametrize("change", ["E_content", "E_dtype", "E_identity", "catalogue", "p", "seed", "policy", "E_shape"])
def test_identity_changes_and_expected_mismatch_rejection(change):
    e = torch.tensor([[2., 0.], [0, 1], [0, 1], [0, 0]], dtype=torch.float64)
    result = _build(e)
    changed_e, kwargs = e.clone(), _kwargs()
    if change == "E_content":
        changed_e[3, 0] = 0.1
    elif change == "E_dtype":
        changed_e = changed_e.float()
    elif change == "E_identity":
        kwargs["public_embedding_id"] = "public-E/v2"
    elif change == "catalogue":
        kwargs["catalogue_id"] = "catalogue/v2"
    elif change == "p":
        kwargs["p"] = 3
    elif change == "seed":
        kwargs["basis_seed"] = 91
    elif change == "policy":
        kwargs["policy"] = RANDOM
    else:
        changed_e = torch.cat([changed_e, torch.zeros((4, 1), dtype=torch.float64)], dim=1)
    new_result = build_public_basis(changed_e, **kwargs)
    assert new_result.record.binding_id != result.record.binding_id
    with pytest.raises(ValueError):
        validate_public_basis(result, item_embeddings=changed_e, expected_binding_id=result.record.binding_id, **kwargs)
    with pytest.raises(ValueError, match="binding"):
        validate_public_basis(new_result, item_embeddings=changed_e, expected_binding_id=result.record.binding_id, **kwargs)


def test_random_policy_directions_independent_of_finite_E_values_and_d():
    a, b = torch.arange(8, dtype=torch.float64).reshape(4, 2), torch.full((4, 2), 999., dtype=torch.float64)
    first, second = _build(a, policy=RANDOM), _build(b, policy=RANDOM)
    different_d = _build(torch.zeros((4, 7), dtype=torch.float32), policy=RANDOM)
    assert torch.equal(first.basis, second.basis) and torch.equal(first.basis, different_d.basis)
    assert first.record.embedding_fingerprint != second.record.embedding_fingerprint
    assert first.record.binding_id != second.record.binding_id
    assert first.record.selected_pivots == () and first.record.rank_threshold is None


def test_independent_identity_encodings_and_checksum():
    import struct
    e = torch.tensor([[1., 2.], [0, -1], [3, 0]], dtype=torch.float64)
    result = _build(e)
    input_bytes = b"".join(struct.pack("<d", float(x)) for x in e.reshape(-1))
    expected_E = hashlib.sha256(_canonical({"format": "thesisfedrec.public_embedding", "version": 1, "shape": [3, 2], "dtype": "float64", "sha256": hashlib.sha256(input_bytes).hexdigest()})).hexdigest()
    assert result.record.embedding_fingerprint == expected_E
    assert result.record.basis_sha256 == _checksum(result.basis)
    payload = asdict(result.record)
    payload.pop("binding_id")
    expected_binding = hashlib.sha256(_canonical(payload)).hexdigest()
    assert result.record.binding_id == expected_binding
    assert result.record.numerical_payload_bytes == 4 * 3 * 2
    assert dict(result.record.environment)["numpy"] == np.__version__


@pytest.mark.parametrize("kind", ["mutated", "sign_flip", "column_swap", "signed_zero", "checksum_record", "expected_binding", "malformed_binding"])
def test_changed_bytes_or_binding_rejected(kind):
    e = torch.eye(3)
    result = _build(e)
    binding = result.record.binding_id
    if kind == "mutated":
        result.basis[0, 0] += 1e-6
    elif kind == "sign_flip":
        result.basis[:, 0] *= -1
    elif kind == "column_swap":
        result.basis[:] = result.basis[:, [1, 0]]
    elif kind == "signed_zero":
        result.basis[2, 0] = -0.0
    elif kind == "checksum_record":
        result = PublicBasis(result.basis, replace(result.record, basis_sha256="f" * 64))
    elif kind == "expected_binding":
        binding = "f" * 64
    else:
        binding = "bad"
    with pytest.raises(ValueError):
        _validate(result, e, expected_binding_id=binding)


@pytest.mark.parametrize("field,value", [
    ("format", "bad"), ("format", None), ("format_version", True), ("policy_version", 2),
    ("policy", "unknown"), ("n", False), ("d", 0), ("p", 0), ("basis_seed", -1),
    ("construction_dtype", "float32"), ("delivery_dtype", "float64"), ("embedding_dtype", "float16"),
    ("rng_implementation", "MT19937"), ("rank_threshold", float('nan')), ("rank_threshold", 0),
    ("selected_pivots", [0, 1]), ("selected_pivots", (0, 0)), ("selected_pivots", (99,)),
    ("random_attempts", 10), ("random_acceptances", True), ("coordinate_fallback", (2, 1)),
    ("environment", ()), ("environment", (("python", "3"),) * 5),
    ("numerical_payload_bytes", 1), ("public_embedding_id", ""), ("catalogue_id", "bad\n"),
    ("embedding_fingerprint", "x" * 64), ("basis_sha256", 0), ("binding_id", "A" * 64),
])
def test_malformed_record_rejected(field, value):
    e = torch.eye(3)
    result = _build(e)
    changed = PublicBasis(result.basis, replace(result.record, **{field: value}))
    with pytest.raises(ValueError):
        _validate(changed, e, expected_binding_id=result.record.binding_id)


@pytest.mark.parametrize("kind", ["shape", "dtype", "strided", "gradient", "nan", "record_dict", "result_dict"])
def test_malformed_delivery_rejected(kind):
    e = torch.eye(3)
    result = _build(e)
    basis = result.basis
    if kind == "shape":
        basis = basis[:2]
    elif kind == "dtype":
        basis = basis.double()
    elif kind == "strided":
        basis = torch.stack([basis, basis], dim=2)[:, :, 0]
        assert not basis.is_contiguous()
    elif kind == "gradient":
        basis = basis.clone().requires_grad_()
    elif kind == "nan":
        basis = basis.clone()
        basis[0, 0] = float('nan')
    if kind in ("result_dict", "record_dict"):
        invalid = dict(basis=basis, record=asdict(result.record)) if kind == "result_dict" else PublicBasis(basis, asdict(result.record))
        with pytest.raises(ValueError):
            validate_public_basis(invalid, item_embeddings=e, expected_binding_id=result.record.binding_id, **_kwargs())
    else:
        with pytest.raises(ValueError):
            _validate(PublicBasis(basis, result.record), e)


@pytest.mark.parametrize("multiple,valid", [(0.5, True), (2.0, False)])
def test_delivered_orthogonality_threshold_without_reorthogonalizing(multiple, valid):
    e = torch.eye(3)
    result = _build(e)
    basis = result.basis.clone()
    basis[:, 0] *= math.sqrt(1 + multiple * 1e-5 * result.record.p)
    changed = _rebind(result, basis)
    before = changed.basis.clone()
    if valid:
        _validate(changed, e)
    else:
        with pytest.raises(ValueError, match="orthogonality"):
            _validate(changed, e)
    assert torch.equal(changed.basis, before)


def test_validator_does_not_regenerate_and_saved_bytes_are_authoritative(monkeypatch):
    e = torch.eye(3)
    result = _build(e)
    def forbidden(*args, **kwargs):
        raise AssertionError("basis reconstruction during validation")
    monkeypatch.setattr(public_basis, "build_public_basis", forbidden)
    monkeypatch.setattr(public_basis, "_pivot_columns", forbidden)
    monkeypatch.setattr(public_basis, "_complete", forbidden)
    _validate(result, e)
    # An authoritative record from another numerical environment can be checked
    # by its trusted binding without requiring local bitwise reconstruction.
    environment = tuple((key, "other-version" if key == "numpy" else value) for key, value in result.record.environment)
    other_environment = _rebind(result, environment=environment)
    _validate(other_environment, e)
    with pytest.raises(ValueError, match="binding"):
        _validate(other_environment, e, expected_binding_id=result.record.binding_id)


def test_test_owned_payload_and_record_round_trip(tmp_path):
    e = torch.tensor([[1., 2.], [-1, 0], [0, 1]], dtype=torch.float64)
    result = _build(e)
    payload_file, record_file = tmp_path / "basis.f32le", tmp_path / "record.json"
    payload_file.write_bytes(result.basis.numpy().astype("<f4").tobytes(order="C"))
    record_file.write_bytes(_canonical(asdict(result.record)))
    data = json.loads(record_file.read_bytes())
    for name in ("selected_pivots", "coordinate_fallback"):
        data[name] = tuple(data[name])
    data["environment"] = tuple(tuple(pair) for pair in data["environment"])
    record = PublicBasisRecord(**data)
    loaded = torch.from_numpy(np.fromfile(payload_file, dtype="<f4").reshape(record.n, record.p).copy())
    restored = PublicBasis(loaded, record)
    _validate(restored, e, expected_binding_id=result.record.binding_id)
    assert torch.equal(restored.basis, result.basis) and restored.record == result.record
    assert payload_file.stat().st_size == 4 * record.n * record.p
    print("BASIS_ARTIFACT_EVIDENCE=" + json.dumps({"numerical_payload_bytes": payload_file.stat().st_size, "serialized_record_bytes": record_file.stat().st_size, "basis_sha256": _checksum(loaded), "binding_id": record.binding_id, "path": str(tmp_path), "scope": "test-owned files, not network traffic or persistence integration"}))


def test_unmodified_float32_basis_accepted_by_U4_with_independent_dense_sketch():
    e = torch.tensor([[2., 0.], [0, 1], [0, 1]], dtype=torch.float32)
    result = _build(e)
    paths = prepare_path_factors(torch.arange(3), torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 1]]), num_items=3)
    source = prepare_source_kernel(paths, e, torch.ones(2), torch.tensor(0.), result.basis)
    assert source.basis.data_ptr() == result.basis.data_ptr()
    sketch = make_operator_sketch(source, torch.zeros(2))
    t = 1 / (2 * math.sqrt(2))
    dense = np.array([[0, t, 0], [t, 0, t], [0, t, 0]])
    b = result.basis.numpy().astype(np.float64)
    expected = np.stack([0.5 * b.T @ dense @ b, 0.5 * b.T @ dense @ dense @ b])
    np.testing.assert_allclose(sketch.numpy(), expected, atol=1e-5, rtol=1e-5)
    print("BASIS_U4_COMPATIBILITY=" + json.dumps({"max_absolute_sketch_error": float(np.max(np.abs(sketch.numpy() - expected))), "basis_sha256": result.record.basis_sha256}))


def test_bounded_resource_observation(monkeypatch):
    n, d, p = 4096, 12, 8
    # Strided float32 E exercises canonical-read and float64-construction copies.
    storage = torch.arange(n * d * 2, dtype=torch.float32).reshape(n, d * 2)
    e = torch.sin(storage / 100)[:, ::2]
    shapes, maxima = set(), {"residual_bytes": 0, "Q_bytes": 0, "E64_bytes": 0}
    originals = {name: getattr(public_basis.np, name) for name in ("empty", "zeros", "array", "ascontiguousarray")}
    def instrument(name):
        def call(*args, **kwargs):
            result = originals[name](*args, **kwargs)
            if isinstance(result, np.ndarray):
                shapes.add(tuple(result.shape))
            return result
        return call
    for name in originals:
        monkeypatch.setattr(public_basis.np, name, instrument(name))
    original_residual, original_pivot = public_basis._residual, public_basis._pivot_columns
    def observed_residual(candidate, q, count):
        result = original_residual(candidate, q, count)
        maxima["residual_bytes"] = max(maxima["residual_bytes"], result.nbytes)
        maxima["Q_bytes"] = max(maxima["Q_bytes"], q.nbytes)
        return result
    def observed_pivot(array, q, tau):
        maxima["E64_bytes"] = array.nbytes
        assert not np.shares_memory(array, e.numpy())
        return original_pivot(array, q, tau)
    monkeypatch.setattr(public_basis, "_residual", observed_residual)
    monkeypatch.setattr(public_basis, "_pivot_columns", observed_pivot)
    tracemalloc.start()
    result = _build(e, p=p)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert (n, n) not in shapes
    assert result.basis.untyped_storage().nbytes() == 4 * n * p
    source = inspect.getsource(public_basis)
    for forbidden in ("linalg.qr", "linalg.svd", "np.cov(", "q @ q.T", "np.random.seed(", "torch.manual_seed("):
        assert forbidden not in source
    gram = result.basis.double().T @ result.basis.double()
    error = torch.linalg.matrix_norm(gram - torch.eye(p, dtype=torch.float64)).item()
    print("BASIS_RESOURCE_OBSERVATION=" + json.dumps({
        "n": n, "d": d, "p": p, "input_dtype": "float32", "input_strided": True,
        "input_view_logical_bytes": e.numel() * e.element_size(),
        "canonical_E_read_copy_bytes_for_this_strided_input": e.numel() * e.element_size(),
        **maxima, "projection_product_vector_bytes": 8 * n,
        "delivery_float64_validation_copy_bytes": 8 * n * p,
        "orthogonality_gram_bytes": 8 * p * p,
        "delivered_numerical_payload_bytes": 4 * n * p,
        "serialized_record_bytes": len(_canonical(asdict(result.record))),
        "observed_numpy_shapes": sorted(shapes, key=lambda shape: (len(shape), shape)),
        "tracemalloc_peak_bytes": peak,
        "process_peak_rss_kib_including_imports_and_other_tests": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "delivered_frobenius_orthogonality_error": error,
        "limits": "shape trace covers selected NumPy calls; source inspection covers matmul/astype sites; tracemalloc excludes untracked native/BLAS/PyTorch allocations; inventory components need not peak together; no per-party copy/coordinator/network claim; p=n output is allowed; not resource qualification",
    }, sort_keys=True))
