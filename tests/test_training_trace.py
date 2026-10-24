"""Synthetic PW7 trace checks with independent serialization and fixed inputs.

No real data, private production helper, method kernel or training is used.
Invalid consumed-domain cases mutate normally constructed baseline records to
exercise the explicit revalidation boundary; no bundle constructor is bypassed.
"""

from collections import UserDict
from dataclasses import FrozenInstanceError, fields
from hashlib import sha256
import json
import random

import numpy as np
import pytest
import torch

import src.data.training_trace as trace_module
from src.data.contracts import (
    DatasetBundle,
    InteractionSplits,
    InteractionTable,
    PartyData,
)
from src.data.negative_sampling import sample_negative_items
from src.data.training_trace import (
    TrainingTrace,
    build_training_trace,
    check_training_trace_context,
)
from src.federation.round_drain import (
    advance_round_drain_cursor,
    peek_round_drain_step,
    prepare_round_drain_plan,
    start_round_drain,
)
from src.federation.trace_batch import pack_party_group_batch


LIMIT = 2**63 - 1
EXCLUDED = ("train", "valid", "test")
USERS = (0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3)
ITEMS = (0, 1, 0, 2, 3, 1, 2, 3, 0, 1, 2, 4, 3)
TRAIN = (0, 1, 2, 5, 8)
VALID = (3, 6, 9, 11)
TEST = (4, 7, 10, 12)
TRACE_FIELDS = (
    "protocol_version", "trace_id", "context_id", "run_seed", "round_index",
    "local_epoch", "negatives_per_positive", "replace", "excluded_splits",
    "selected_party_ids", "positive_order_seeds", "negative_seed",
    "ordered_positive_ids", "negative_items",
)
# Recorded from literal independent seed envelopes before the focused run;
# NumPy 1.26.4 permutations below are version-scoped replay fixtures.
POSITIVE_SEEDS = (
    (7, 17522385641706874545),
    (41, 12101413937057227020),
    (90, 4955421891240060543),
)
NEGATIVE_SEED = 13107405540649085677


def _ids(values):
    return np.array(values, dtype=np.int64)


def _party(party_id, users, train, *, party_type=PartyData):
    return party_type(
        party_id=party_id,
        train_interaction_ids=_ids(train),
        valid_interaction_ids=_ids(()),
        test_interaction_ids=_ids(()),
        user_ids=_ids(users),
        item_ids=_ids(()),
        feature_view_names=(),
        graph_view_names=(),
        owns_labels=bool(train),
        metadata={},
    )


def _arguments(*, bundle_type=DatasetBundle):
    # User 3 supplies item 4 in a real support interaction, so the accepted
    # constructor sees complete processed namespaces. User 0's repeated item 0
    # at distinct interaction IDs is deliberate, not deduplicated by tracing.
    bundle = bundle_type(
        name="synthetic-training-trace", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=_ids(range(len(USERS))), user_ids=_ids(USERS),
            item_ids=_ids(ITEMS), timestamps=_ids(range(len(USERS))),
            values=np.ones(len(USERS), dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=_ids(TRAIN),
            valid_interaction_ids=_ids(VALID),
            test_interaction_ids=_ids(TEST),
        ),
        user_id_map={1: 0, "1": 1, "u2": 2, "u3": 3},
        item_id_map={f"i{i}": i for i in range(5)}, metadata={},
    )
    return dict(
        bundle=bundle,
        parties=(
            _party(41, (0, 1), (0, 1, 2, 5)),
            _party(13, (3,), ()),
            _party(90, (), ()),
            _party(7, (2,), (8,)),
        ),
        selected_party_ids=(41, 90, 7), round_index=4, local_epoch=2,
        run_seed=2026, negatives_per_positive=1,
        excluded_splits=EXCLUDED, replace=False,
    )


def _find_party(arguments, party_id):
    return next(p for p in arguments["parties"] if p.party_id == party_id)


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def _seed_oracle(stream, *, party_id=None, run_seed=2026,
                 round_index=4, local_epoch=2):
    payload = {
        "format": "thesisfedrec.training_trace.seed", "version": 1,
        "run_seed": run_seed, "round_index": round_index,
        "local_epoch": local_epoch, "stream": stream,
    }
    if party_id is not None:
        payload["party_id"] = party_id
    return int.from_bytes(sha256(_canonical(payload)).digest()[:8], "big")


def _fixed_context():
    # Hand-written semantic object for the fixed fixture, not generated from
    # production validation, source serialization or the result being tested.
    return {
        "format": "thesisfedrec.training_trace.context", "version": 1,
        "user_map": [
            {"processed_id": 0, "raw_type": "int", "raw_id": 1},
            {"processed_id": 1, "raw_type": "str", "raw_id": "1"},
            {"processed_id": 2, "raw_type": "str", "raw_id": "u2"},
            {"processed_id": 3, "raw_type": "str", "raw_id": "u3"},
        ],
        "item_map": [
            {"processed_id": i, "raw_type": "str", "raw_id": f"i{i}"}
            for i in range(5)
        ],
        "interaction_ids": list(range(13)), "user_ids": list(USERS),
        "item_ids": list(ITEMS),
        "splits": {"train": list(TRAIN), "valid": list(VALID), "test": list(TEST)},
        "parties": [
            {"party_id": 7, "user_ids": [2], "train_interaction_ids": [8],
             "owns_labels": True},
            {"party_id": 41, "user_ids": [0, 1],
             "train_interaction_ids": [0, 1, 2, 5], "owns_labels": True},
            {"party_id": 90, "user_ids": [], "train_interaction_ids": [],
             "owns_labels": False},
        ],
    }


def _trace_semantic(trace):
    # Independent literal wire-format mapping, deliberately no production
    # hash/helper. Also used to test structural checks behind a consistent hash.
    return {
        "format": "thesisfedrec.training_trace", "version": 1,
        "context_id": trace.context_id, "run_seed": trace.run_seed,
        "round_index": trace.round_index, "local_epoch": trace.local_epoch,
        "negatives_per_positive": trace.negatives_per_positive,
        "replace": trace.replace, "excluded_splits": list(trace.excluded_splits),
        "selected_party_ids": list(trace.selected_party_ids),
        "positive_order_seeds": [
            {"party_id": p, "seed": s} for p, s in trace.positive_order_seeds
        ],
        "negative_seed": trace.negative_seed,
        "ordered_positive_ids": [
            {"party_id": p, "interaction_ids": list(ids)}
            for p, ids in trace.ordered_positive_ids
        ],
        "negative_items": [
            {"interaction_id": i, "item_ids": list(ids)}
            for i, ids in trace.negative_items
        ],
    }


def _check(trace, arguments):
    return check_training_trace_context(
        trace, arguments["bundle"], arguments["parties"]
    )


def _forbid(*args, **kwargs):
    raise AssertionError("unexpected sampler or RNG call")


def _corrupt(trace, *, rehash=False, **changes):
    # The checker must reject malformed records, even though supported public
    # mutation is frozen. This helper deliberately bypasses only TrainingTrace,
    # never DatasetBundle or PartyData construction.
    result = object.__new__(TrainingTrace)
    for field in fields(trace):
        object.__setattr__(result, field.name, getattr(trace, field.name))
    for name, value in changes.items():
        object.__setattr__(result, name, value)
    if rehash:
        digest = sha256(_canonical(_trace_semantic(result))).hexdigest()
        object.__setattr__(result, "trace_id", digest)
    return result


def test_literal_context_seed_envelopes_and_realized_trace_identity():
    arguments = _arguments()
    trace = build_training_trace(**arguments)
    assert type(trace) is TrainingTrace
    assert tuple(f.name for f in fields(trace)) == TRACE_FIELDS
    assert trace.protocol_version == 1
    assert trace.context_id == sha256(_canonical(_fixed_context())).hexdigest()
    assert trace.selected_party_ids == (7, 41, 90)
    positive_seeds = tuple(
        (p, _seed_oracle("positive-order", party_id=p)) for p in (7, 41, 90)
    )
    assert trace.positive_order_seeds == positive_seeds == POSITIVE_SEEDS
    assert trace.negative_seed == _seed_oracle("negative-items") == NEGATIVE_SEED
    assert trace.ordered_positive_ids == ((7, (8,)), (41, (0, 5, 1, 2)), (90, ()))
    assert tuple(i for i, _ in trace.negative_items) == TRAIN
    assert trace.trace_id == sha256(_canonical(_trace_semantic(trace))).hexdigest()
    assert _check(trace, arguments) is None


@pytest.mark.parametrize("q,replace", [(0, False), (1, False), (1, True), (4, True)])
def test_exact_existing_sampler_comparison_at_derived_negative_seed(q, replace):
    arguments = _arguments()
    arguments.update(negatives_per_positive=q, replace=replace)
    expected = sample_negative_items(
        arguments["bundle"], TRAIN, negatives_per_positive=q,
        strategy="uniform", seed=_seed_oracle("negative-items"),
        excluded_splits=EXCLUDED, replace=replace,
    )
    trace = build_training_trace(**arguments)
    assert trace.negative_items == tuple(expected.items())
    assert all(row == tuple(sorted(row)) for _, row in trace.negative_items)
    for interaction_id, row in trace.negative_items:
        user = USERS[interaction_id]
        known = {item for owner, item in zip(USERS, ITEMS) if owner == user}
        assert not known.intersection(row)
    assert tuple(i for _, order in trace.ordered_positive_ids for i in order)


@pytest.mark.parametrize("selection,q,expected_calls", [
    ((41, 90, 7), 1, 1), ((41, 90, 7), 0, 0),
    ((), 3, 0), ((90,), 3, 0),
])
def test_sampler_once_or_zero_and_private_rng_streams(
    monkeypatch, selection, q, expected_calls
):
    arguments = _arguments()
    arguments.update(selected_party_ids=selection, negatives_per_positive=q)
    sampler_calls, rng_calls = [], []
    original_rng = np.random.default_rng

    def recording_sampler(bundle, positive_ids, **options):
        sampler_calls.append((bundle, tuple(positive_ids), options))
        return sample_negative_items(bundle, positive_ids, **options)

    def recording_rng(seed):
        rng_calls.append(seed)
        return original_rng(seed)

    monkeypatch.setattr(trace_module, "sample_negative_items", recording_sampler)
    monkeypatch.setattr(np.random, "default_rng", recording_rng)
    trace = build_training_trace(**arguments)
    assert len(sampler_calls) == expected_calls
    if expected_calls:
        assert sampler_calls == [(arguments["bundle"], TRAIN, {
            "negatives_per_positive": 1, "strategy": "uniform",
            "seed": _seed_oracle("negative-items"),
            "excluded_splits": EXCLUDED, "replace": False,
        })]
    expected_rng = [
        _seed_oracle("positive-order", party_id=p) for p in sorted(selection)
    ]
    if expected_calls:
        expected_rng.append(_seed_oracle("negative-items"))
    assert rng_calls == expected_rng
    if not q:
        assert trace.negative_items == tuple((i, ()) for i in TRAIN)


def test_insertion_order_nonselected_changes_and_added_empty_party():
    arguments = _arguments()
    expected = build_training_trace(**arguments)
    arguments["parties"] = tuple(reversed(arguments["parties"]))
    arguments["selected_party_ids"] = [90, 7, 41]
    bundle = arguments["bundle"]
    bundle.user_id_map = dict(reversed(tuple(bundle.user_id_map.items())))
    bundle.item_id_map = dict(reversed(tuple(bundle.item_id_map.items())))
    _find_party(arguments, 13).party_id = 1001
    assert build_training_trace(**arguments) == expected
    arguments["selected_party_ids"] = (7, 41)
    no_empty = build_training_trace(**arguments)
    assert no_empty.negative_seed == expected.negative_seed
    assert no_empty.negative_items == expected.negative_items
    assert no_empty.ordered_positive_ids == expected.ordered_positive_ids[:-1]
    assert no_empty.context_id != expected.context_id
    assert no_empty.trace_id != expected.trace_id


@pytest.mark.parametrize("field", ["run_seed", "round_index", "local_epoch"])
def test_seed_inputs_change_identity_without_changing_context(field):
    arguments = _arguments()
    first = build_training_trace(**arguments)
    arguments[field] += 1
    second = build_training_trace(**arguments)
    assert first.context_id == second.context_id
    assert first.trace_id != second.trace_id
    assert first.positive_order_seeds != second.positive_order_seeds
    assert first.negative_seed != second.negative_seed


def test_owned_builtin_only_record_public_immutability_and_input_isolation():
    arguments = _arguments()
    trace = build_training_trace(**arguments)

    def builtin_tree(value):
        if type(value) is tuple:
            for child in value:
                builtin_tree(child)
        else:
            assert type(value) in (str, int, bool)

    for field in fields(trace):
        builtin_tree(getattr(trace, field.name))
    assert not hasattr(trace, "__dict__")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        trace.run_seed = 0
    with pytest.raises(TypeError):
        TrainingTrace()
    with pytest.raises(TypeError):
        TrainingTrace(**{f.name: getattr(trace, f.name) for f in fields(trace)})
    before = _trace_semantic(trace)
    arguments["bundle"].interactions.item_ids[:] = 0
    arguments["bundle"].user_id_map.clear()
    _find_party(arguments, 41).train_interaction_ids[:] = 0
    arguments["parties"] = ()
    assert _trace_semantic(trace) == before


def test_checker_has_no_rng_or_sampling_and_preserves_all_global_rng(monkeypatch):
    arguments = _arguments()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    trace = build_training_trace(**arguments)
    monkeypatch.setattr(trace_module, "sample_negative_items", _forbid)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    assert _check(trace, arguments) is None
    assert _check(trace, arguments) is None
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


class _Unread:
    def __getattribute__(self, name):
        raise AssertionError(f"unconsumed field was inspected: {name}")

    def __iter__(self):
        raise AssertionError("unconsumed field was iterated")

    def __len__(self):
        raise AssertionError("unconsumed field length was inspected")


def test_only_consumed_payload_is_read_and_unselected_payload_is_opaque():
    arguments = _arguments()
    expected = build_training_trace(**arguments)
    bundle = arguments["bundle"]
    for name in ("name", "schema_version", "metadata", "feature_views",
                 "graph_views", "continual_views", "parties", "partition_name",
                 "partition_kind"):
        setattr(bundle, name, _Unread())
    bundle.interactions.timestamps = _Unread()
    bundle.interactions.values = _Unread()
    for party in arguments["parties"]:
        for name in ("item_ids", "valid_interaction_ids", "test_interaction_ids",
                     "feature_view_names", "graph_view_names", "metadata"):
            setattr(party, name, _Unread())
    unselected = _find_party(arguments, 13)
    unselected.train_interaction_ids = _Unread()
    unselected.user_ids = _Unread()
    unselected.owns_labels = _Unread()
    assert build_training_trace(**arguments) == expected
    assert _check(expected, arguments) is None


CONSUMED_PATHS = (
    "interactions.interaction_ids", "interactions.user_ids",
    "interactions.item_ids", "splits.train_interaction_ids",
    "splits.valid_interaction_ids", "splits.test_interaction_ids",
    "party.train_interaction_ids", "party.user_ids",
)


def _location(arguments, path):
    owner, field = path.split(".")
    record = (_find_party(arguments, 41) if owner == "party"
              else getattr(arguments["bundle"], owner))
    return record, field


def test_readonly_strided_arrays_preserve_values_flags_and_headers():
    arguments = _arguments()
    expected = build_training_trace(**arguments)
    snapshots = []
    for path in CONSUMED_PATHS:
        record, name = _location(arguments, path)
        original = getattr(record, name)
        backing = np.zeros(original.size * 2, dtype=np.int64)
        backing[::2] = original
        view = backing[::2]
        view.setflags(write=False)
        setattr(record, name, view)
        snapshots.append((record, name, view, view.copy(), view.shape, view.strides))
    assert build_training_trace(**arguments) == expected
    assert _check(expected, arguments) is None
    for record, name, view, copy, shape, strides in snapshots:
        assert getattr(record, name) is view
        assert view.shape == shape and view.strides == strides
        assert not view.flags.writeable
        np.testing.assert_array_equal(view, copy)


class _Int(int):
    pass


class _Str(str):
    pass


class _Tuple(tuple):
    pass


class _Array(np.ndarray):
    pass


class _Bundle(DatasetBundle):
    pass


class _Party(PartyData):
    pass


@pytest.mark.parametrize("field", [
    "run_seed", "round_index", "local_epoch", "negatives_per_positive",
])
@pytest.mark.parametrize("value,error", [
    (True, TypeError), (np.int64(1), TypeError), (_Int(1), TypeError),
    (1.0, TypeError), ("1", TypeError), (-1, ValueError), (LIMIT + 1, ValueError),
])
def test_scalar_preflight_rejects_without_rng_or_sampler(monkeypatch, field, value, error):
    arguments = _arguments()
    arguments[field] = value
    monkeypatch.setattr(trace_module, "sample_negative_items", _forbid)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    with pytest.raises(error):
        build_training_trace(**arguments)


@pytest.mark.parametrize("selection,error", [
    ("7", TypeError), (b"7", TypeError), (bytearray(b"7"), TypeError),
    ({7}, TypeError), (iter((7,)), TypeError),
    ((True,), TypeError), ((np.int64(7),), TypeError), ((_Int(7),), TypeError),
    ((7, 7), ValueError), ((-1,), ValueError), ((LIMIT + 1,), ValueError),
    ((999,), ValueError),
])
def test_selected_party_preflight(monkeypatch, selection, error):
    arguments = _arguments()
    arguments["selected_party_ids"] = selection
    monkeypatch.setattr(trace_module, "sample_negative_items", _forbid)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    with pytest.raises(error):
        build_training_trace(**arguments)


@pytest.mark.parametrize("replace", [0, 1, np.bool_(True), "false", None])
def test_replacement_requires_exact_bool(replace):
    arguments = _arguments()
    arguments["replace"] = replace
    with pytest.raises(TypeError):
        build_training_trace(**arguments)


@pytest.mark.parametrize("excluded,error", [
    (["train", "valid", "test"], TypeError),
    ("train", TypeError), (_Tuple(EXCLUDED), TypeError),
    ((_Str("train"), "valid", "test"), TypeError),
    (("test", "valid", "train"), ValueError), (("train",), ValueError),
    (("train", "train", "test"), ValueError), ((), ValueError),
])
def test_only_exact_three_split_protocol_is_accepted(excluded, error):
    arguments = _arguments()
    arguments["excluded_splits"] = excluded
    with pytest.raises(error):
        build_training_trace(**arguments)


@pytest.mark.parametrize("case,error", [
    ("bundle-subclass", TypeError), ("not-bundle", TypeError),
    ("party-subclass", TypeError), ("not-party", TypeError),
    ("roster-generator", TypeError), ("roster-string", TypeError),
    ("roster-duplicate", ValueError), ("roster-bool", TypeError),
    ("roster-numpy-int", TypeError), ("roster-int-subclass", TypeError),
    ("roster-negative", ValueError), ("roster-too-large", ValueError),
])
def test_exact_top_level_types_and_all_roster_keys(case, error):
    arguments = _arguments(bundle_type=_Bundle if case == "bundle-subclass" else DatasetBundle)
    if case == "not-bundle":
        arguments["bundle"] = object()
    elif case == "party-subclass":
        arguments["parties"] += (_party(101, (), (), party_type=_Party),)
    elif case == "not-party":
        arguments["parties"] += (object(),)
    elif case == "roster-generator":
        arguments["parties"] = iter(arguments["parties"])
    elif case == "roster-string":
        arguments["parties"] = "parties"
    elif case == "roster-duplicate":
        arguments["parties"] += (_party(13, (), ()),)
    elif case.startswith("roster-"):
        _find_party(arguments, 13).party_id = {
            "roster-bool": True, "roster-numpy-int": np.int64(13),
            "roster-int-subclass": _Int(13), "roster-negative": -1,
            "roster-too-large": LIMIT + 1,
        }[case]
    with pytest.raises(error):
        build_training_trace(**arguments)


@pytest.mark.parametrize("path", CONSUMED_PATHS)
@pytest.mark.parametrize("representation,error", [
    ("list", TypeError), ("int32", TypeError), ("float64", TypeError),
    ("non-native", TypeError), ("metadata", TypeError),
    ("subclass", TypeError), ("masked", TypeError), ("rank-two", ValueError),
])
def test_consumed_array_representation_is_revalidated(path, representation, error):
    arguments = _arguments()
    trace = build_training_trace(**arguments)
    record, name = _location(arguments, path)
    values = getattr(record, name)
    if representation == "list":
        bad = values.tolist()
    elif representation == "int32":
        bad = values.astype(np.int32)
    elif representation == "float64":
        bad = values.astype(np.float64)
    elif representation == "non-native":
        bad = values.astype(np.dtype(np.int64).newbyteorder("S"))
    elif representation == "metadata":
        bad = values.astype(np.dtype(np.int64, metadata={}))
    elif representation == "subclass":
        bad = values.view(_Array)
    elif representation == "masked":
        bad = np.ma.array(values, mask=False)
    else:
        bad = values.reshape(1, -1)
    setattr(record, name, bad)
    with pytest.raises(error):
        build_training_trace(**arguments)
    with pytest.raises(error):
        _check(trace, arguments)


@pytest.mark.parametrize("case", [
    "interaction-positional", "interaction-length", "user-negative", "item-outside",
    "split-negative", "split-outside", "split-unsorted", "split-duplicate",
    "split-overlap", "split-hole", "train-nontraining", "train-unsorted",
    "train-duplicate", "train-negative", "train-overlap", "user-unsorted",
    "user-duplicate", "user-outside", "user-missing", "labels-false",
])
def test_invalid_current_domain_or_ownership_rejected_before_rng(monkeypatch, case):
    arguments = _arguments()
    trace = build_training_trace(**arguments)
    bundle = arguments["bundle"]
    party = _find_party(arguments, 41)
    if case == "interaction-positional":
        bundle.interactions.interaction_ids[1] = 0
    elif case == "interaction-length":
        bundle.interactions.user_ids = bundle.interactions.user_ids[:-1]
    elif case == "user-negative":
        bundle.interactions.user_ids[0] = -1
    elif case == "item-outside":
        bundle.interactions.item_ids[0] = 5
    elif case.startswith("split-"):
        bad_splits = {
            "split-negative": (-1, 6, 9, 11), "split-outside": (3, 6, 9, 13),
            "split-unsorted": (6, 3, 9, 11), "split-duplicate": (3, 3, 9, 11),
            "split-overlap": (0, 6, 9, 11), "split-hole": (3, 6, 9),
        }
        bundle.splits.valid_interaction_ids = _ids(bad_splits[case])
    elif case.startswith("train-"):
        if case == "train-overlap":
            _find_party(arguments, 7).train_interaction_ids = _ids((0, 8))
            _find_party(arguments, 7).user_ids = _ids((0, 2))
        else:
            party.train_interaction_ids = _ids({
                "train-nontraining": (0, 1, 2, 3),
                "train-unsorted": (1, 0, 2, 5), "train-duplicate": (0, 1, 2, 2),
                "train-negative": (-1, 1, 2, 5),
            }[case])
    elif case.startswith("user-"):
        party.user_ids = _ids({
            "user-unsorted": (1, 0), "user-duplicate": (0, 0),
            "user-outside": (0, 1, 4), "user-missing": (0,),
        }[case])
    else:
        party.owns_labels = False
    monkeypatch.setattr(trace_module, "sample_negative_items", _forbid)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    with pytest.raises(ValueError):
        build_training_trace(**arguments)
    with pytest.raises(ValueError):
        _check(trace, arguments)


@pytest.mark.parametrize("map_name", ["user_id_map", "item_id_map"])
@pytest.mark.parametrize("case,error", [
    ("not-dict", TypeError), ("raw-bool", TypeError), ("raw-numpy", TypeError),
    ("raw-int-subclass", TypeError), ("raw-str-subclass", TypeError),
    ("processed-bool", TypeError), ("processed-numpy", TypeError),
    ("processed-subclass", TypeError), ("duplicate", ValueError),
    ("gap", ValueError),
])
def test_mapping_keys_and_values_validate_before_normalization(map_name, case, error):
    arguments = _arguments()
    trace = build_training_trace(**arguments)
    bundle = arguments["bundle"]
    mapping = getattr(bundle, map_name)
    entries = list(mapping.items())
    raw, value = entries[0]
    if case == "not-dict":
        bad = UserDict(mapping)
    elif case.startswith("raw-"):
        bad_raw = {
            "raw-bool": True, "raw-numpy": np.int64(100),
            "raw-int-subclass": _Int(100), "raw-str-subclass": _Str("raw"),
        }[case]
        bad = dict([(bad_raw, value), *entries[1:]])
    else:
        bad_value = {
            "processed-bool": False, "processed-numpy": np.int64(0),
            "processed-subclass": _Int(0), "duplicate": 1, "gap": len(entries),
        }[case]
        bad = dict([(raw, bad_value), *entries[1:]])
    setattr(bundle, map_name, bad)
    with pytest.raises(error):
        build_training_trace(**arguments)
    with pytest.raises(error):
        _check(trace, arguments)


@pytest.mark.parametrize("owns_labels", [0, 1, np.bool_(True), "true"])
def test_owns_labels_exact_bool_for_selected_empty_and_nonempty(owns_labels):
    for party_id in (41, 90):
        arguments = _arguments()
        _find_party(arguments, party_id).owns_labels = owns_labels
        with pytest.raises(TypeError):
            build_training_trace(**arguments)


@pytest.mark.parametrize("change", [
    "raw-id-type", "raw-id-value", "map-permutation", "interaction-user",
    "interaction-item", "held-out-split", "selected-users", "selected-train",
    "empty-party-labels",
])
def test_context_binds_current_consumed_values_even_when_structurally_valid(change):
    arguments = _arguments()
    trace = build_training_trace(**arguments)
    bundle = arguments["bundle"]
    if change == "raw-id-type":
        bundle.user_id_map = {"zero": 0, 1: 1, "u2": 2, "u3": 3}
    elif change == "raw-id-value":
        bundle.user_id_map = {2: 0, "1": 1, "u2": 2, "u3": 3}
    elif change == "map-permutation":
        bundle.item_id_map["i0"], bundle.item_id_map["i1"] = 1, 0
    elif change == "interaction-user":
        bundle.interactions.user_ids[0] = 1
    elif change == "interaction-item":
        bundle.interactions.item_ids[0] = 1
    elif change == "held-out-split":
        bundle.splits.valid_interaction_ids = _ids(TEST)
        bundle.splits.test_interaction_ids = _ids(VALID)
    elif change == "selected-users":
        _find_party(arguments, 41).user_ids = _ids((0, 1, 2))
    elif change == "selected-train":
        _find_party(arguments, 41).train_interaction_ids = _ids((0, 1, 5))
    else:
        _find_party(arguments, 90).owns_labels = True
    with pytest.raises(ValueError):
        _check(trace, arguments)
    assert build_training_trace(**arguments).context_id != trace.context_id


@pytest.mark.parametrize("selected", [(), (90,)])
def test_empty_work_still_validates_global_domain_and_roster(selected):
    arguments = _arguments()
    arguments["selected_party_ids"] = selected
    trace = build_training_trace(**arguments)
    assert trace.ordered_positive_ids == tuple((p, ()) for p in selected)
    assert trace.negative_items == ()
    assert _check(trace, arguments) is None
    arguments["bundle"].splits.test_interaction_ids = _ids(())
    with pytest.raises(ValueError):
        build_training_trace(**arguments)
    with pytest.raises(ValueError):
        _check(trace, arguments)


def test_fully_empty_bundle_and_roster_remain_well_defined():
    arguments = _arguments()
    arguments["bundle"] = DatasetBundle(
        name="empty", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=_ids(()), user_ids=_ids(()), item_ids=_ids(()),
            timestamps=_ids(()), values=np.empty(0, dtype=np.float64),
        ),
        splits=InteractionSplits(
            train_interaction_ids=_ids(()), valid_interaction_ids=_ids(()),
            test_interaction_ids=_ids(()),
        ), user_id_map={}, item_id_map={}, metadata={},
    )
    arguments.update(parties=(), selected_party_ids=())
    trace = build_training_trace(**arguments)
    assert trace.selected_party_ids == trace.ordered_positive_ids == ()
    assert trace.negative_items == trace.positive_order_seeds == ()
    assert _check(trace, arguments) is None


def test_expanded_int64_overflow_fails_before_rng_or_sampling(monkeypatch):
    arguments = _arguments()
    arguments["negatives_per_positive"] = LIMIT
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    monkeypatch.setattr(trace_module, "sample_negative_items", _forbid)
    with pytest.raises(ValueError):
        build_training_trace(**arguments)


def test_empty_work_accepts_int64_maximum_scalar_values_without_large_allocation():
    arguments = _arguments()
    arguments.update(selected_party_ids=(), run_seed=LIMIT,
                     round_index=LIMIT, local_epoch=LIMIT,
                     negatives_per_positive=LIMIT)
    trace = build_training_trace(**arguments)
    assert trace.negatives_per_positive == LIMIT
    assert trace.negative_items == ()
    assert _check(trace, arguments) is None


def test_sampler_insufficient_pool_and_exact_exception_propagation(monkeypatch):
    arguments = _arguments()
    arguments["negatives_per_positive"] = 2
    with pytest.raises(ValueError, match="user 0 has only 1 eligible negative items"):
        build_training_trace(**arguments)
    sentinel = ValueError("existing sampler failure object")
    calls = []

    def failing_sampler(*args, **kwargs):
        calls.append(kwargs)
        raise sentinel

    monkeypatch.setattr(trace_module, "sample_negative_items", failing_sampler)
    with pytest.raises(ValueError) as raised:
        build_training_trace(**arguments)
    assert raised.value is sentinel
    assert len(calls) == 1
    assert calls[0]["negatives_per_positive"] == 2
    assert calls[0]["replace"] is False


@pytest.mark.parametrize("field,value,error", [
    ("protocol_version", True, TypeError), ("protocol_version", 2, ValueError),
    ("trace_id", "A" * 64, ValueError), ("trace_id", "0" * 63, ValueError),
    ("context_id", _Str("a" * 64), TypeError),
    ("run_seed", np.int64(2026), TypeError), ("round_index", -1, ValueError),
    ("local_epoch", LIMIT + 1, ValueError), ("replace", 0, TypeError),
    ("excluded_splits", list(EXCLUDED), TypeError),
    ("selected_party_ids", [7, 41, 90], TypeError),
    ("selected_party_ids", (True, 41, 90), TypeError),
    ("positive_order_seeds", ((7, True), (41, 0), (90, 0)), TypeError),
    ("negative_seed", 2**64, ValueError),
    ("ordered_positive_ids", ((7, (np.int64(8),)),), TypeError),
    ("negative_items", ((0, [4]),), TypeError),
])
def test_checker_rejects_malformed_record_fields_without_rng(
    monkeypatch, field, value, error
):
    arguments = _arguments()
    trace = build_training_trace(**arguments)
    bad = _corrupt(trace, **{field: value})
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    monkeypatch.setattr(trace_module, "sample_negative_items", _forbid)
    with pytest.raises(error):
        _check(bad, arguments)


@pytest.mark.parametrize("case", [
    "selected-order", "selected-duplicate", "order-party-key", "positive-missing",
    "positive-duplicate", "positive-wrong-owner", "negative-missing",
    "negative-extra", "negative-order", "negative-count", "negative-outside",
    "negative-target", "negative-known-positive", "positive-seed", "negative-seed",
])
def test_checker_rejects_rehashed_incomplete_or_inconsistent_trace(monkeypatch, case):
    arguments = _arguments()
    trace = build_training_trace(**arguments)
    order = dict(trace.ordered_positive_ids)
    negatives = dict(trace.negative_items)
    changes = {}
    if case == "selected-order":
        changes["selected_party_ids"] = (41, 7, 90)
    elif case == "selected-duplicate":
        changes["selected_party_ids"] = (7, 41, 41)
    elif case == "order-party-key":
        changes["ordered_positive_ids"] = ((7, order[7]), (40, order[41]), (90, ()))
    elif case.startswith("positive-") and case != "positive-seed":
        order[41] = {
            "positive-missing": (0, 1, 5), "positive-duplicate": (0, 0, 2, 5),
            "positive-wrong-owner": (0, 1, 2, 8),
        }[case]
        changes["ordered_positive_ids"] = tuple(order.items())
    elif case == "positive-seed":
        changes["positive_order_seeds"] = tuple(
            (p, s + (p == 7)) for p, s in trace.positive_order_seeds
        )
    elif case == "negative-seed":
        changes["negative_seed"] = trace.negative_seed ^ 1
    else:
        if case == "negative-missing":
            negatives.pop(0)
        elif case == "negative-extra":
            negatives[3] = (4,)
        elif case == "negative-order":
            negatives = dict(reversed(tuple(negatives.items())))
        else:
            negatives[0] = {
                "negative-count": (), "negative-outside": (5,),
                "negative-target": (0,), "negative-known-positive": (2,),
            }[case]
        changes["negative_items"] = tuple(negatives.items())
    bad = _corrupt(trace, rehash=True, **changes)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    monkeypatch.setattr(trace_module, "sample_negative_items", _forbid)
    with pytest.raises(ValueError):
        _check(bad, arguments)


def test_checker_rejects_plain_wrong_type_and_stale_trace_hash():
    arguments = _arguments()
    trace = build_training_trace(**arguments)
    with pytest.raises(TypeError):
        _check(object(), arguments)
    with pytest.raises(ValueError):
        _check(_corrupt(trace, trace_id="0" * 64), arguments)


@pytest.mark.parametrize("row", [(4, 3), (3, 3)])
def test_checker_rejects_unsorted_or_unapproved_duplicate_negative_rows(row):
    arguments = _arguments()
    arguments.update(selected_party_ids=(7,), negatives_per_positive=2)
    trace = build_training_trace(**arguments)
    assert trace.negative_items == ((8, (3, 4)),)
    bad = _corrupt(trace, rehash=True, negative_items=((8, row),))
    with pytest.raises(ValueError):
        _check(bad, arguments)


@pytest.mark.parametrize("q,replace", [(0, False), (1, False), (3, True)])
def test_trace_planner_packer_tiny_group_major_tail_and_replacement(q, replace):
    # Three duplicate-positive interaction rows for user 0 and explicit support
    # rows for user 1 establish the full two-item catalog. Its only negative
    # candidate is item 1; all payload expectations below are hand-worked.
    arguments = _arguments()
    arguments["bundle"] = DatasetBundle(
        name="tiny-chain", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=_ids((0, 1, 2, 3, 4)),
            user_ids=_ids((0, 0, 0, 1, 1)), item_ids=_ids((0, 0, 0, 1, 1)),
            timestamps=_ids((0, 1, 2, 3, 4)), values=np.ones(5, dtype=np.float32),
        ),
        splits=InteractionSplits(train_interaction_ids=_ids((0, 1, 2)),
                                 valid_interaction_ids=_ids((3,)),
                                 test_interaction_ids=_ids((4,))),
        user_id_map={"receiver": 0, "support": 1},
        item_id_map={"positive": 0, "negative": 1}, metadata={},
    )
    arguments.update(parties=(_party(41, (0,), (0, 1, 2)), _party(90, (), ())),
                     selected_party_ids=(90, 41),
                     negatives_per_positive=q, replace=replace)
    trace = build_training_trace(**arguments)
    assert _check(trace, arguments) is None
    assert trace.ordered_positive_ids == ((41, (0, 1, 2)), (90, ()))
    plan = prepare_round_drain_plan(
        arguments["bundle"], arguments["parties"],
        selected_party_ids=trace.selected_party_ids,
        ordered_positive_ids=dict(trace.ordered_positive_ids),
        negative_items=dict(trace.negative_items), round_index=trace.round_index,
        batch_unit="positive_group", positive_groups_per_batch=2,
        negatives_per_positive=trace.negatives_per_positive,
        negative_replacement=trace.replace, trace_id=trace.trace_id,
    )
    cursor = start_round_drain(plan)
    all_positive_ids = []
    for index, count in enumerate((2, 1)):
        step = peek_round_drain_step(plan, cursor, requested_source_party_ids=(90,))
        assert step.source_only_party_ids == (90,)
        assert len(step.batches) == 1
        batch = step.batches[0]
        packed = pack_party_group_batch(batch, dtype=torch.float64)
        all_positive_ids.extend(packed.positive_ids)
        assert packed.party_id == 41
        assert packed.loss_coefficient == (count, 3)
        assert packed.user_ids.tolist() == [0] * (count * (q + 1))
        assert packed.item_ids.tolist() == ([0] + [1] * q) * count
        assert packed.labels.tolist() == ([1.0] + [0.0] * q) * count
        assert packed.group_offsets.tolist() == [i * (q + 1) for i in range(count + 1)]
        cursor = advance_round_drain_cursor(plan, cursor, completed_step_index=index)
    assert all_positive_ids == [0, 1, 2]
    assert peek_round_drain_step(plan, cursor, requested_source_party_ids=()) is None
