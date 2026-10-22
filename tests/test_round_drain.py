"""Independent synthetic contract checks; no sampler, training or real data."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
import hashlib
import importlib
import json
import random

import numpy as np
import pytest
import torch

import src.federation.round_drain as round_drain_module
from src.data.contracts import (
    DatasetBundle,
    InteractionSplits,
    InteractionTable,
    PartyData,
)
from src.federation.round_drain import (
    PartyGroupBatch,
    RoundDrainCursor,
    RoundDrainPlan,
    RoundDrainStep,
    advance_round_drain_cursor,
    peek_round_drain_step,
    prepare_round_drain_plan,
    start_round_drain,
)


LIMIT = 2**63 - 1
TRACE_ID = "a" * 64
POSITIVES = {7: (101, 109), 41: (203, 211, 227, 251, 277)}
NEGATIVES = {101: (7,), 109: (7,), **dict.fromkeys(POSITIVES[41], (8,))}


def _party(party_id, user_ids, train_ids):
    empty = np.empty(0, dtype=np.int64)
    return PartyData(
        party_id=party_id,
        train_interaction_ids=np.array(train_ids, dtype=np.int64),
        valid_interaction_ids=empty.copy(),
        test_interaction_ids=empty.copy(),
        user_ids=np.array(user_ids, dtype=np.int64),
        item_ids=empty.copy(),
        feature_view_names=(),
        graph_view_names=(),
        owns_labels=True,
        metadata={},
    )


def _inputs():
    # Exact literal traces from the approved historical fixed reference. Other
    # rows make its non-contiguous interaction IDs legal positional bundle IDs.
    ids = np.arange(278, dtype=np.int64)
    users, items = ids % 18, ids % 9
    users[list(POSITIVES[7])] = 17
    users[list(POSITIVES[41])] = 3
    items[list(POSITIVES[7])] = (0, 1)
    items[list(POSITIVES[41])] = (2, 3, 4, 5, 6)
    train = np.array((1, *POSITIVES[7], *POSITIVES[41]), dtype=np.int64)
    held_out = np.setdiff1d(ids, train)
    bundle = DatasetBundle(
        name="synthetic-round-drain",
        schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=ids,
            user_ids=users,
            item_ids=items,
            timestamps=ids.copy(),
            values=np.ones(278, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=train,
            valid_interaction_ids=held_out[:-1],
            test_interaction_ids=held_out[-1:],
        ),
        user_id_map={f"user-{i}": i for i in range(18)},
        item_id_map={f"item-{i}": i for i in range(9)},
        metadata={},
    )
    return dict(
        bundle=bundle,
        parties=(
            _party(41, (3,), POSITIVES[41]),
            _party(90, (), ()),
            _party(7, (17,), POSITIVES[7]),
            _party(13, (1,), (1,)),
        ),
        selected_party_ids=(41, 7),
        ordered_positive_ids=dict(POSITIVES),
        negative_items=dict(NEGATIVES),
        round_index=4,
        batch_unit="positive_group",
        positive_groups_per_batch=2,
        negatives_per_positive=1,
        negative_replacement=False,
        trace_id=TRACE_ID,
    )


def _find_party(arguments, party_id):
    return next(p for p in arguments["parties"] if p.party_id == party_id)


def _prepare(**changes):
    arguments = _inputs()
    arguments.update(changes)
    return prepare_round_drain_plan(**arguments)


def _drain(plan, sources=()):
    cursor = start_round_drain(plan)
    steps = []
    for index in range(plan.step_count):
        step = peek_round_drain_step(
            plan, cursor, requested_source_party_ids=sources
        )
        assert step is not None
        steps.append(step)
        cursor = advance_round_drain_cursor(
            plan, cursor, completed_step_index=index
        )
    assert peek_round_drain_step(
        plan, cursor, requested_source_party_ids=()
    ) is None
    return steps


def _identity_oracle(arguments):
    # Deliberately independent serialization, not a production/private helper.
    table = arguments["bundle"].interactions
    entries = []
    for party_id in sorted(arguments["selected_party_ids"]):
        positives = arguments["ordered_positive_ids"][party_id]
        entries.append(dict(
            party_id=party_id,
            positive_ids=list(positives),
            user_ids=[int(table.user_ids[i]) for i in positives],
            positive_item_ids=[int(table.item_ids[i]) for i in positives],
            negative_item_ids=[list(arguments["negative_items"][i]) for i in positives],
        ))
    payload = dict(
        format="thesisfedrec.round_drain_plan",
        version=1,
        protocol="round_drain_once_interaction_mass",
        protocol_version=1,
        trace_id=arguments["trace_id"],
        round_index=arguments["round_index"],
        batch_unit=arguments["batch_unit"],
        positive_groups_per_batch=arguments["positive_groups_per_batch"],
        negatives_per_positive=arguments["negatives_per_positive"],
        negative_replacement=arguments["negative_replacement"],
        parties=entries,
    )
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_fixed_reference_exact_batches_weights_identity_and_exposure():
    arguments = _inputs()
    plan = prepare_round_drain_plan(**arguments)
    assert isinstance(plan, RoundDrainPlan)
    assert plan.protocol == "round_drain_once_interaction_mass"
    assert plan.protocol_version == 1
    assert plan.plan_id == _identity_oracle(arguments)
    assert plan.trace_id == TRACE_ID
    assert plan.selected_party_ids == plan.loss_party_ids == (7, 41)
    assert plan.positive_counts == {7: 2, 41: 5}
    assert plan.batch_counts == {7: 1, 41: 3}
    assert (plan.round_positive_count, plan.step_count, plan.skipped) == (7, 3, False)
    assert plan.utilization_fraction == (4, 6)
    assert tuple(plan.negative_items) == (101, 109, 203, 211, 227, 251, 277)
    steps = _drain(plan)
    expected = (
        ((7, 0, 2, (101, 109), (17, 17), (0, 1), ((7,), (7,)), (2, 7)),
         (41, 0, 2, (203, 211), (3, 3), (2, 3), ((8,), (8,)), (2, 7))),
        ((41, 2, 4, (227, 251), (3, 3), (4, 5), ((8,), (8,)), (2, 7)),),
        ((41, 4, 5, (277,), (3,), (6,), ((8,),), (1, 7)),),
    )
    for index, (step, expected_batches) in enumerate(zip(steps, expected)):
        assert isinstance(step, RoundDrainStep)
        assert step.plan_id == plan.plan_id
        assert step.step_index == index
        assert tuple((b.party_id, b.start, b.stop, b.positive_ids, b.user_ids,
                      b.positive_item_ids, b.negative_item_ids, b.loss_coefficient)
                     for b in step.batches) == expected_batches
        assert all(isinstance(b, PartyGroupBatch) for b in step.batches)
    assert [s.bce_row_count for s in steps] == [8, 4, 2]
    exposure = [i for s in steps for b in s.batches for i in b.positive_ids]
    assert sorted(exposure) == [101, 109, 203, 211, 227, 251, 277]
    # Exact integer numerators against the frozen same-parameter reference;
    # no optimizer execution or fresh run of the old Fraction script.
    gradients = dict(zip(exposure, (1, 3, 2, 4, 6, 8, 10)))
    weighted_numerators = []
    for step in steps:
        numerator = 0
        for batch in step.batches:
            m, denominator = batch.loss_coefficient
            assert m == len(batch.positive_ids) and denominator == 7
            numerator += sum(gradients[i] for i in batch.positive_ids)
        weighted_numerators.append(numerator)
    assert weighted_numerators == [10, 14, 10]
    assert sum(weighted_numerators) == 34
    assert sum(s.bce_row_count for s in steps) == 14


def test_source_requests_change_roles_not_schedule_or_loss():
    plan = _prepare()
    idle = _drain(plan)
    sourcing = _drain(plan, (41, 7))
    assert [s.source_only_party_ids for s in sourcing] == [(), (7,), (7,)]
    assert [s.idle_party_ids for s in idle] == [(), (7,), (7,)]
    assert all(s.idle_party_ids == () for s in sourcing)
    assert all(s.requested_source_party_ids == (7, 41) for s in sourcing)
    for first, second in zip(idle, sourcing):
        assert (first.plan_id, first.step_index) == (second.plan_id, second.step_index)
        assert first.batches == second.batches
        assert first.bce_row_count == second.bce_row_count
    assert sum(any(b.party_id == 7 for b in s.batches) for s in sourcing) == 1
    assert sum(7 in s.source_only_party_ids or any(b.party_id == 7 for b in s.batches)
               for s in sourcing) == 3


@pytest.mark.parametrize("selected", [(), (90,)])
def test_skipped_round_preserves_empty_selected_parties(selected):
    plan = _prepare(selected_party_ids=selected,
                    ordered_positive_ids={i: () for i in selected}, negative_items={})
    assert plan.selected_party_ids == selected
    assert plan.loss_party_ids == ()
    assert plan.round_positive_count == plan.step_count == 0
    assert plan.skipped is True and plan.utilization_fraction is None
    assert dict(plan.positive_counts) == dict.fromkeys(selected, 0)
    assert dict(plan.batch_counts) == dict.fromkeys(selected, 0)
    assert dict(plan.positive_ids_by_party) == dict.fromkeys(selected, ())
    assert _drain(plan) == []
    cursor = start_round_drain(plan)
    with pytest.raises(ValueError):
        advance_round_drain_cursor(plan, cursor, completed_step_index=0)
    if selected:
        with pytest.raises(ValueError):
            peek_round_drain_step(plan, cursor, requested_source_party_ids=selected)


def test_empty_party_is_source_eligible_without_entering_denominator():
    plan = _prepare(selected_party_ids=(90, 7),
                    ordered_positive_ids={7: POSITIVES[7], 90: ()},
                    negative_items={i: NEGATIVES[i] for i in POSITIVES[7]},
                    positive_groups_per_batch=1)
    assert plan.loss_party_ids == (7,)
    assert plan.round_positive_count == 2 and plan.utilization_fraction == (2, 2)
    for step in _drain(plan, (90,)):
        assert step.source_only_party_ids == (90,)
        assert step.batches[0].loss_coefficient == (1, 2)


def test_grouped_users_and_unbounded_party_id_are_not_user_indices():
    arguments = _inputs()
    large_id = 2**70
    party = _find_party(arguments, 7)
    party.party_id = large_id
    party.user_ids = np.array([2, 17], dtype=np.int64)
    arguments["bundle"].interactions.user_ids[109] = 2
    arguments["selected_party_ids"] = (large_id,)
    arguments["ordered_positive_ids"] = {large_id: (109, 101)}
    arguments["negative_items"] = {i: NEGATIVES[i] for i in POSITIVES[7]}
    plan = prepare_round_drain_plan(**arguments)
    batch = _drain(plan)[0].batches[0]
    assert batch.party_id == large_id
    assert batch.positive_ids == (109, 101)
    assert batch.user_ids == (2, 17)


def test_q_zero_and_explicit_replacement():
    plan = _prepare(negatives_per_positive=0,
                    negative_items={i: () for i in NEGATIVES})
    assert [s.bce_row_count for s in _drain(plan)] == [4, 2, 1]
    for batch in _drain(plan)[0].batches:
        assert all(row == () for row in batch.negative_item_ids)
    duplicated = {i: values * 2 for i, values in NEGATIVES.items()}
    plan = _prepare(negatives_per_positive=2, negative_items=duplicated,
                    negative_replacement=True)
    assert _drain(plan)[0].batches[0].negative_item_ids == ((7, 7), (7, 7))
    with pytest.raises(ValueError):
        _prepare(negatives_per_positive=2, negative_items=duplicated)


@pytest.mark.parametrize("field,value", [
    ("round_index", LIMIT),
    ("positive_groups_per_batch", LIMIT),
    ("negatives_per_positive", LIMIT - 1),
])
def test_empty_plan_accepts_valid_int64_capacity_boundaries(field, value):
    settings = dict(selected_party_ids=(), ordered_positive_ids={}, negative_items={},
                    positive_groups_per_batch=1, negatives_per_positive=0)
    settings[field] = value
    assert _prepare(**settings).skipped


@pytest.mark.parametrize("settings", [
    {"positive_groups_per_batch": LIMIT, "negatives_per_positive": 1},
    {"positive_groups_per_batch": 1, "negatives_per_positive": LIMIT},
    {"positive_groups_per_batch": 2**32, "negatives_per_positive": 2**32},
])
def test_batch_capacity_rejected_even_when_trace_is_empty(settings):
    with pytest.raises(ValueError):
        _prepare(selected_party_ids=(), ordered_positive_ids={}, negative_items={}, **settings)


def test_total_expanded_capacity_is_checked_with_python_integers():
    # Must fail on total capacity, not the deliberately impossible q tuple size.
    with pytest.raises(ValueError, match="round.*capacity"):
        _prepare(selected_party_ids=(7,), ordered_positive_ids={7: POSITIVES[7]},
                 negative_items={i: () for i in POSITIVES[7]},
                 positive_groups_per_batch=1, negatives_per_positive=LIMIT // 2)


def test_empty_roster_and_empty_processed_dataset_are_valid():
    arguments = _inputs()
    empty = np.empty(0, dtype=np.int64)
    arguments["bundle"] = DatasetBundle(
        name="empty", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=empty.copy(), user_ids=empty.copy(),
            item_ids=empty.copy(), timestamps=empty.copy(),
            values=np.empty(0, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=empty.copy(),
            valid_interaction_ids=empty.copy(), test_interaction_ids=empty.copy(),
        ),
        user_id_map={}, item_id_map={}, metadata={},
    )
    arguments.update(parties=(), selected_party_ids=(),
                     ordered_positive_ids={}, negative_items={})
    plan = prepare_round_drain_plan(**arguments)
    assert plan.skipped and plan.selected_party_ids == ()
    assert plan.plan_id == _identity_oracle(arguments)
    assert _drain(plan) == []


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


class _TupleSubclass(tuple):
    pass


@pytest.mark.parametrize("field", ["round_index", "positive_groups_per_batch", "negatives_per_positive"])
@pytest.mark.parametrize("value", [True, np.int64(1), _IntSubclass(1), 1.0, "1", None])
def test_integer_setting_requires_exact_builtin_int(field, value):
    with pytest.raises(TypeError):
        _prepare(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("round_index", -1), ("round_index", 2**63),
    ("positive_groups_per_batch", 0), ("positive_groups_per_batch", -1),
    ("positive_groups_per_batch", 2**63),
    ("negatives_per_positive", -1), ("negatives_per_positive", 2**63),
])
def test_integer_setting_range(field, value):
    with pytest.raises(ValueError):
        _prepare(**{field: value})


@pytest.mark.parametrize("field,value,error", [
    ("batch_unit", "label_row", ValueError),
    ("batch_unit", _StrSubclass("positive_group"), TypeError),
    ("batch_unit", None, TypeError),
    ("negative_replacement", 0, TypeError),
    ("negative_replacement", np.bool_(True), TypeError),
    ("trace_id", "A" * 64, ValueError),
    ("trace_id", "a" * 63, ValueError),
    ("trace_id", "g" * 64, ValueError),
    ("trace_id", "a" * 64 + "\n", ValueError),
    ("trace_id", _StrSubclass(TRACE_ID), TypeError),
    ("trace_id", None, TypeError),
])
def test_strings_and_replacement_are_explicit(field, value, error):
    with pytest.raises(error):
        _prepare(**{field: value})


@pytest.mark.parametrize("field", ["parties", "selected_party_ids"])
@pytest.mark.parametrize("bad", ["", b"", bytearray(), None, {7}, iter((7,))])
def test_roster_and_selection_require_reusable_nonstring_sequences(field, bad):
    with pytest.raises(TypeError):
        _prepare(**{field: bad})


@pytest.mark.parametrize("field,bad", [
    ("bundle", object()), ("parties", (object(),)),
    ("ordered_positive_ids", []), ("negative_items", []),
])
def test_top_level_contract_types(field, bad):
    with pytest.raises(TypeError):
        _prepare(**{field: bad})


@pytest.mark.parametrize("bad", [True, np.int64(7), _IntSubclass(7), "7"])
@pytest.mark.parametrize("location", ["roster", "selection", "positive_key", "negative_key", "positive", "negative"])
def test_id_types_are_validated_before_sort_set_copy_or_coercion(bad, location):
    arguments = _inputs()
    if location == "roster":
        _find_party(arguments, 7).party_id = bad
    elif location == "selection":
        arguments["selected_party_ids"] = (bad, 41)
    elif location == "positive_key":
        arguments["ordered_positive_ids"] = {bad: POSITIVES[7], 41: POSITIVES[41]}
    elif location == "negative_key":
        arguments["negative_items"] = {bad: (7,), **{i: v for i, v in NEGATIVES.items() if i != 101}}
    elif location == "positive":
        arguments["ordered_positive_ids"][7] = (bad, 109)
    else:
        arguments["negative_items"][101] = (bad,)
    with pytest.raises(TypeError):
        prepare_round_drain_plan(**arguments)


@pytest.mark.parametrize("field", ["ordered_positive_ids", "negative_items"])
@pytest.mark.parametrize("value", [[101, 109], np.array([101, 109]), iter((101, 109)), _TupleSubclass((101, 109))])
def test_trace_values_must_be_tuples(field, value):
    arguments = _inputs()
    arguments[field][7 if field == "ordered_positive_ids" else 101] = value
    with pytest.raises(TypeError):
        prepare_round_drain_plan(**arguments)


@pytest.mark.parametrize("field", ["ordered_positive_ids", "negative_items"])
def test_bad_mapping_key_is_rejected_before_lookup_or_copy(field):
    class BadKeyMapping(Mapping):
        def __len__(self):
            return 1

        def __iter__(self):
            yield np.int64(7)

        def __getitem__(self, key):
            raise AssertionError("invalid key must be rejected before value lookup")

    with pytest.raises(TypeError):
        _prepare(**{field: BadKeyMapping()})


@pytest.mark.parametrize("selected", [(7, 7), (999,), (-1,)])
def test_invalid_selection_fails(selected):
    with pytest.raises(ValueError):
        _prepare(selected_party_ids=selected)


def test_duplicate_or_negative_roster_id_fails_even_when_unselected():
    arguments = _inputs()
    _find_party(arguments, 13).party_id = 90
    with pytest.raises(ValueError):
        prepare_round_drain_plan(**arguments)
    arguments = _inputs()
    _find_party(arguments, 13).party_id = -1
    with pytest.raises(ValueError):
        prepare_round_drain_plan(**arguments)


ARRAY_FIELDS = (
    "interaction_ids", "user_ids", "item_ids", "train", "valid", "test",
    "party_train", "party_users",
)


def _array_owner(arguments, field):
    if field in ("interaction_ids", "user_ids", "item_ids"):
        return arguments["bundle"].interactions, field
    if field in ("train", "valid", "test"):
        return arguments["bundle"].splits, field + "_interaction_ids"
    return _find_party(arguments, 7), {
        "party_train": "train_interaction_ids", "party_users": "user_ids"
    }[field]


class _ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize("field", ARRAY_FIELDS)
@pytest.mark.parametrize("kind,error", [
    ("list", TypeError), ("float", TypeError), ("int32", TypeError),
    ("metadata", TypeError), ("subclass", TypeError),
    ("matrix", ValueError), ("scalar", ValueError),
])
@pytest.mark.parametrize("empty", [False, True])
def test_consumed_array_exact_type_dtype_and_dimension(field, kind, error, empty):
    arguments = _inputs()
    owner, attribute = _array_owner(arguments, field)
    original = getattr(owner, attribute)
    values = original[:0] if empty else original
    replacement = {
        "list": lambda: values.tolist(),
        "float": lambda: values.astype(np.float64),
        "int32": lambda: values.astype(np.int32),
        "metadata": lambda: values.astype(np.dtype(np.int64, metadata={"tag": "bad"})),
        "subclass": lambda: values.view(_ArraySubclass),
        "matrix": lambda: values.reshape((1, -1)),
        "scalar": lambda: np.array(0, dtype=np.int64),
    }[kind]()
    setattr(owner, attribute, replacement)
    with pytest.raises(error):
        prepare_round_drain_plan(**arguments)


def test_noncontiguous_readonly_input_arrays_are_valid_and_identity_stable():
    arguments = _inputs()
    expected = prepare_round_drain_plan(**arguments).plan_id
    for field in ARRAY_FIELDS:
        owner, attribute = _array_owner(arguments, field)
        values = getattr(owner, attribute)
        storage = np.empty(2 * len(values), dtype=np.int64)
        view = storage[::2]
        view[:] = values
        view.setflags(write=False)
        setattr(owner, attribute, view)
    assert prepare_round_drain_plan(**arguments).plan_id == expected


@pytest.mark.parametrize("field,value", [
    ("interaction_ids", np.arange(278, dtype=np.int64)[::-1]),
    ("user_ids", np.zeros(277, dtype=np.int64)),
    ("item_ids", np.zeros(279, dtype=np.int64)),
    ("user_ids", np.full(278, -1, dtype=np.int64)),
    ("user_ids", np.full(278, 18, dtype=np.int64)),
    ("item_ids", np.full(278, 9, dtype=np.int64)),
    ("train", np.array([109, 101], dtype=np.int64)),
    ("valid", np.array([0, 0], dtype=np.int64)),
    ("test", np.array([278], dtype=np.int64)),
    ("party_train", np.array([109, 101], dtype=np.int64)),
    ("party_train", np.array([101, 101], dtype=np.int64)),
    ("party_train", np.array([-1], dtype=np.int64)),
    ("party_train", np.array([278], dtype=np.int64)),
    ("party_train", np.array([0], dtype=np.int64)),
    ("party_users", np.array([17, 3], dtype=np.int64)),
    ("party_users", np.array([17, 17], dtype=np.int64)),
    ("party_users", np.array([-1], dtype=np.int64)),
    ("party_users", np.array([18], dtype=np.int64)),
    ("party_users", np.array([3], dtype=np.int64)),
])
def test_invalid_context_and_ownership_array_values(field, value):
    arguments = _inputs()
    owner, attribute = _array_owner(arguments, field)
    setattr(owner, attribute, value)
    with pytest.raises(ValueError):
        prepare_round_drain_plan(**arguments)


@pytest.mark.parametrize("which", ["overlap", "missing"])
def test_global_split_coverage_and_disjointness(which):
    arguments = _inputs()
    splits = arguments["bundle"].splits
    if which == "overlap":
        splits.test_interaction_ids = np.array([0], dtype=np.int64)
    else:
        splits.test_interaction_ids = np.empty(0, dtype=np.int64)
    with pytest.raises(ValueError):
        prepare_round_drain_plan(**arguments)


@pytest.mark.parametrize("field", ["user_id_map", "item_id_map"])
@pytest.mark.parametrize("bad,error", [
    ([], TypeError), ({"raw": True}, TypeError),
    ({"raw": np.int64(0)}, TypeError), ({"raw": _IntSubclass(0)}, TypeError),
    ({"raw": 1}, ValueError), ({"a": 0, "b": 0}, ValueError),
])
def test_processed_mapping_values_without_reinterpreting_raw_ids(field, bad, error):
    arguments = _inputs()
    setattr(arguments["bundle"], field, bad)
    with pytest.raises(error):
        prepare_round_drain_plan(**arguments)


@pytest.mark.parametrize("value,error", [(False, ValueError), (1, TypeError), (np.bool_(True), TypeError)])
def test_selected_nonempty_party_requires_exact_true_label_ownership(value, error):
    arguments = _inputs()
    _find_party(arguments, 7).owns_labels = value
    with pytest.raises(error):
        prepare_round_drain_plan(**arguments)


def test_empty_selected_party_requires_bool_but_need_not_own_labels():
    arguments = _inputs()
    _find_party(arguments, 90).owns_labels = False
    arguments.update(selected_party_ids=(90,), ordered_positive_ids={90: ()}, negative_items={})
    assert prepare_round_drain_plan(**arguments).skipped
    _find_party(arguments, 90).owns_labels = 0
    with pytest.raises(TypeError):
        prepare_round_drain_plan(**arguments)


def test_selected_histories_cannot_share_an_interaction():
    arguments = _inputs()
    party = _find_party(arguments, 41)
    party.train_interaction_ids = np.array([101, *POSITIVES[41]], dtype=np.int64)
    party.user_ids = np.array([3, 17], dtype=np.int64)
    arguments["ordered_positive_ids"][41] = (101, *POSITIVES[41])
    with pytest.raises(ValueError):
        prepare_round_drain_plan(**arguments)


@pytest.mark.parametrize("positive_ids", [(101,), (101, 101), (101, 203), (101, -1), (101, 278), (101, 0)])
def test_positive_trace_is_exact_owned_train_permutation(positive_ids):
    arguments = _inputs()
    arguments["ordered_positive_ids"][7] = positive_ids
    with pytest.raises(ValueError):
        prepare_round_drain_plan(**arguments)


@pytest.mark.parametrize("field,operation", [
    ("ordered_positive_ids", "missing"), ("ordered_positive_ids", "extra"),
    ("negative_items", "missing"), ("negative_items", "extra"),
])
def test_trace_keys_must_match_exactly(field, operation):
    arguments = _inputs()
    key = 7 if field == "ordered_positive_ids" else 101
    if operation == "missing":
        del arguments[field][key]
    else:
        arguments[field][999] = ()
    with pytest.raises(ValueError):
        prepare_round_drain_plan(**arguments)


@pytest.mark.parametrize("negative", [(), (7, 8), (-1,), (9,), (0,)])
def test_negative_count_domain_and_unconditional_target_exclusion(negative):
    arguments = _inputs()
    arguments["negative_items"][101] = negative
    with pytest.raises(ValueError):
        prepare_round_drain_plan(**arguments)


def test_known_positive_exclusion_is_not_silently_reconstructed():
    arguments = _inputs()
    # Item1 is another positive for user17, but not this target (item0).
    # The trusted trace producer owns configured known-positive exclusion.
    arguments["negative_items"][101] = (1,)
    assert prepare_round_drain_plan(**arguments).negative_items[101] == (1,)


def test_unselected_payloads_and_unconsumed_bundle_fields_are_not_read(monkeypatch):
    arguments = _inputs()
    expected = prepare_round_drain_plan(**arguments).plan_id
    for party in arguments["parties"]:
        if party.party_id not in (7, 41):
            party.train_interaction_ids = object()
            party.user_ids = object()
            party.owns_labels = object()
        party.valid_interaction_ids = object()
        party.test_interaction_ids = object()
        party.item_ids = object()
        party.feature_view_names = object()
        party.graph_view_names = object()
        party.metadata = object()
    bundle = arguments["bundle"]
    bundle.interactions.values = object()
    bundle.interactions.timestamps = object()
    bundle.metadata = object()
    bundle.parties = object()
    bundle.feature_views = object()
    bundle.graph_views = object()
    bundle.continual_views = object()
    bundle.user_id_map = {object(): value for value in bundle.user_id_map.values()}
    bundle.item_id_map = {object(): value for value in bundle.item_id_map.values()}
    def fail(*args, **kwargs):
        raise AssertionError("whole-bundle/party revalidation is forbidden")
    monkeypatch.setattr(DatasetBundle, "__post_init__", fail)
    monkeypatch.setattr(PartyData, "__post_init__", fail)
    assert prepare_round_drain_plan(**arguments).plan_id == expected


def test_independent_identity_oracle_and_input_order_invariance():
    arguments = _inputs()
    baseline = prepare_round_drain_plan(**arguments)
    arguments["parties"] = tuple(reversed(arguments["parties"]))
    arguments["selected_party_ids"] = (7, 41)
    arguments["ordered_positive_ids"] = dict(reversed(list(arguments["ordered_positive_ids"].items())))
    arguments["negative_items"] = dict(reversed(list(arguments["negative_items"].items())))
    arguments["bundle"].metadata = {"different": "ignored"}
    reordered = prepare_round_drain_plan(**arguments)
    assert baseline.plan_id == reordered.plan_id == _identity_oracle(arguments)


@pytest.mark.parametrize("change", ["positive_order", "negative_order", "payload", "round", "batch", "q", "replacement", "trace", "empty_party", "user"])
def test_identity_sensitivity_matches_independent_oracle(change):
    arguments = _inputs()
    arguments["negatives_per_positive"] = 2
    arguments["negative_items"] = {i: (7, 8) for i in NEGATIVES}
    old = prepare_round_drain_plan(**arguments).plan_id
    if change == "positive_order":
        arguments["ordered_positive_ids"][7] = (109, 101)
    elif change == "negative_order":
        arguments["negative_items"][101] = (8, 7)
    elif change == "payload":
        arguments["negative_items"][101] = (6, 8)
    elif change == "round":
        arguments["round_index"] += 1
    elif change == "batch":
        arguments["positive_groups_per_batch"] = 1
    elif change == "q":
        arguments["negatives_per_positive"] = 1
        arguments["negative_items"] = dict(NEGATIVES)
    elif change == "replacement":
        arguments["negative_replacement"] = True
    elif change == "trace":
        arguments["trace_id"] = "b" * 64
    elif change == "empty_party":
        arguments["selected_party_ids"] = (7, 41, 90)
        arguments["ordered_positive_ids"][90] = ()
    else:
        arguments["bundle"].interactions.user_ids[101] = 2
        _find_party(arguments, 7).user_ids = np.array([2, 17], dtype=np.int64)
    plan = prepare_round_drain_plan(**arguments)
    assert plan.plan_id != old
    assert plan.plan_id == _identity_oracle(arguments)


def test_plan_defensively_owns_immutable_records_and_payloads():
    arguments = _inputs()
    arguments["selected_party_ids"] = [41, 7]
    plan = prepare_round_drain_plan(**arguments)
    expected_steps = _drain(plan)
    expected_id = plan.plan_id
    arguments["selected_party_ids"].clear()
    arguments["ordered_positive_ids"].clear()
    arguments["negative_items"].clear()
    arguments["bundle"].interactions.user_ids[:] = 0
    arguments["bundle"].interactions.item_ids[:] = 0
    arguments["bundle"].splits.train_interaction_ids[:] = 0
    for party in arguments["parties"]:
        party.train_interaction_ids[:] = 0
        party.user_ids[:] = 0
        party.party_id = 999
    assert _drain(plan) == expected_steps and plan.plan_id == expected_id
    for name in ("positive_ids_by_party", "user_ids_by_party", "positive_item_ids_by_party",
                 "negative_items", "positive_counts", "batch_counts"):
        mapping = getattr(plan, name)
        assert isinstance(mapping, Mapping)
        key = next(iter(mapping))
        with pytest.raises(TypeError):
            mapping[key] = ()
        with pytest.raises(TypeError):
            del mapping[key]
    records = (plan, start_round_drain(plan), expected_steps[0], expected_steps[0].batches[0])
    for record in records:
        assert not hasattr(record, "__dict__")
        field = "party_id" if isinstance(record, PartyGroupBatch) else "plan_id"
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, "changed")
    with pytest.raises(TypeError):
        plan.positive_ids_by_party[7][0] = 0
    with pytest.raises(TypeError):
        expected_steps[0].batches[0].negative_item_ids[0][0] = 0


def test_prepare_does_not_modify_inputs_and_source_sequence_is_not_retained():
    arguments = _inputs()
    original_arrays = []
    for field in ARRAY_FIELDS:
        owner, attribute = _array_owner(arguments, field)
        original_arrays.append((owner, attribute, getattr(owner, attribute).copy()))
    positives, negatives = dict(arguments["ordered_positive_ids"]), dict(arguments["negative_items"])
    plan = prepare_round_drain_plan(**arguments)
    for owner, attribute, before in original_arrays:
        assert np.array_equal(getattr(owner, attribute), before)
    assert arguments["ordered_positive_ids"] == positives
    assert arguments["negative_items"] == negatives
    requests = [41, 7]
    step = peek_round_drain_step(plan, start_round_drain(plan), requested_source_party_ids=requests)
    requests.clear()
    assert step.requested_source_party_ids == (7, 41)
    assert all(type(v) is int for v in plan.selected_party_ids)
    for mapping in (plan.positive_ids_by_party, plan.user_ids_by_party,
                    plan.positive_item_ids_by_party, plan.negative_items):
        assert all(type(key) is int and type(row) is tuple
                   and all(type(value) is int for value in row)
                   for key, row in mapping.items())


def test_public_plan_constructor_is_not_supported():
    with pytest.raises(TypeError):
        RoundDrainPlan()


def test_cursor_pure_replay_and_no_hidden_advancement():
    plan = _prepare()
    cursor = start_round_drain(plan)
    first = peek_round_drain_step(plan, cursor, requested_source_party_ids=())
    assert first == peek_round_drain_step(plan, cursor, requested_source_party_ids=())
    advanced = advance_round_drain_cursor(plan, cursor, completed_step_index=0)
    assert cursor.next_step_index == 0 and advanced.next_step_index == 1
    assert advanced is not cursor
    assert first == peek_round_drain_step(plan, cursor, requested_source_party_ids=())
    assert advanced == advance_round_drain_cursor(plan, cursor, completed_step_index=0)
    terminal = RoundDrainCursor(plan_id=plan.plan_id, next_step_index=plan.step_count)
    assert peek_round_drain_step(plan, terminal, requested_source_party_ids=()) is None
    with pytest.raises(ValueError):
        peek_round_drain_step(plan, terminal, requested_source_party_ids=(7,))
    with pytest.raises(ValueError):
        advance_round_drain_cursor(plan, terminal, completed_step_index=plan.step_count)


@pytest.mark.parametrize("value,error", [
    (-1, ValueError), (4, ValueError), (True, TypeError),
    (np.int64(0), TypeError), (_IntSubclass(0), TypeError), (0.0, TypeError),
])
def test_cursor_index_validation(value, error):
    plan = _prepare()
    cursor = RoundDrainCursor(plan_id=plan.plan_id, next_step_index=value)
    with pytest.raises(error):
        peek_round_drain_step(plan, cursor, requested_source_party_ids=())
    with pytest.raises(error):
        advance_round_drain_cursor(plan, cursor, completed_step_index=0)


@pytest.mark.parametrize("value,error", [
    ("b" * 64, ValueError), ("", ValueError), (None, TypeError),
    (_StrSubclass(TRACE_ID), TypeError),
])
def test_cursor_identity_validation(value, error):
    plan = _prepare()
    cursor = RoundDrainCursor(plan_id=value, next_step_index=0)
    with pytest.raises(error):
        peek_round_drain_step(plan, cursor, requested_source_party_ids=())
    with pytest.raises(error):
        advance_round_drain_cursor(plan, cursor, completed_step_index=0)


@pytest.mark.parametrize("value,error", [
    (-1, ValueError), (1, ValueError), (True, TypeError),
    (np.int64(0), TypeError), (_IntSubclass(0), TypeError),
])
def test_advance_requires_current_exact_index(value, error):
    plan = _prepare()
    with pytest.raises(error):
        advance_round_drain_cursor(plan, start_round_drain(plan), completed_step_index=value)


@pytest.mark.parametrize("which", ["plan", "cursor"])
def test_functions_reject_wrong_record_object(which):
    plan = _prepare()
    cursor = start_round_drain(plan)
    if which == "plan":
        with pytest.raises(TypeError):
            start_round_drain(object())
        plan = object()
    else:
        cursor = object()
    with pytest.raises(TypeError):
        peek_round_drain_step(plan, cursor, requested_source_party_ids=())
    with pytest.raises(TypeError):
        advance_round_drain_cursor(plan, cursor, completed_step_index=0)


@pytest.mark.parametrize("value,error", [
    ((7, 7), ValueError), ((90,), ValueError), ((999,), ValueError), ((-1,), ValueError),
    ((True,), TypeError), ((np.int64(7),), TypeError), ((_IntSubclass(7),), TypeError),
    ("", TypeError), (b"", TypeError), (bytearray(), TypeError),
    (iter((7,)), TypeError), ({7}, TypeError), (None, TypeError),
])
def test_source_request_validation(value, error):
    plan = _prepare()
    with pytest.raises(error):
        peek_round_drain_step(plan, start_round_drain(plan), requested_source_party_ids=value)


def test_peek_advance_never_rescan_original_inputs_or_rehash_plan(monkeypatch):
    arguments = _inputs()
    plan = prepare_round_drain_plan(**arguments)
    expected = _drain(plan)
    def fail(*args, **kwargs):
        raise AssertionError("original input access or whole-plan hashing after prepare")
    monkeypatch.setattr(DatasetBundle, "__getattribute__", fail)
    monkeypatch.setattr(PartyData, "__getattribute__", fail)
    monkeypatch.setattr(hashlib, "sha256", fail)
    monkeypatch.setattr(round_drain_module, "sha256", fail)
    monkeypatch.setattr(json, "dumps", fail)
    assert _drain(plan) == expected


def test_planning_calls_no_sampler_rng_seed_model_evaluation_delivery_or_io(monkeypatch):
    arguments = _inputs()
    targets = {
        "src.data.negative_sampling": ("sample_negative_items",),
        "src.utils.seed": ("set_random_seed",),
        "src.evaluation.ranking": ("build_evaluation_candidates", "evaluate_candidates"),
        "src.data.feature_runtime": ("project_party_features",),
        "src.data.graph_runtime": ("project_party_graphs",),
        "src.models.fedgmf": ("train_local_fedgmf", "score_fedgmf"),
        "src.methods.doi4": ("positive_group_bce", "local_recommendation_logits"),
    }
    modules = [(importlib.import_module(name), attrs) for name, attrs in targets.items()]
    def fail(*args, **kwargs):
        raise AssertionError("forbidden execution in pure round-drain planner")
    for module, attributes in modules:
        for name in attributes:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(np.random, "default_rng", fail)
    monkeypatch.setattr(np.random, "seed", fail)
    monkeypatch.setattr(random, "seed", fail)
    monkeypatch.setattr(torch, "manual_seed", fail)
    monkeypatch.setattr(torch.optim, "Adam", fail)
    monkeypatch.setattr("builtins.open", fail)
    plan = prepare_round_drain_plan(**arguments)
    assert len(_drain(plan)) == 3


def test_rng_states_unchanged_for_success_failure_and_cursor_replay():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    plan = _prepare()
    assert len(_drain(plan)) == 3
    with pytest.raises(ValueError):
        _prepare(positive_groups_per_batch=0)
    with pytest.raises(ValueError):
        peek_round_drain_step(plan, start_round_drain(plan), requested_source_party_ids=(999,))
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
