"""Pure, supplied-trace round draining; no sampling or execution side effects."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType

import numpy as np

from src.data.contracts import DatasetBundle, PartyData


_MAX_INDEX = 2**63 - 1
_PROTOCOL = "round_drain_once_interaction_mass"


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class RoundDrainPlan:
    """Owned immutable trace snapshot, constructed only by preparation.

    Identity is trusted-local bookkeeping, not authentication or a proof that
    the caller's trace producer excluded every known positive.
    """

    protocol: str
    protocol_version: int
    plan_id: str
    trace_id: str
    round_index: int
    batch_unit: str
    positive_groups_per_batch: int
    negatives_per_positive: int
    negative_replacement: bool
    selected_party_ids: tuple[int, ...]
    loss_party_ids: tuple[int, ...]
    positive_ids_by_party: Mapping[int, tuple[int, ...]]
    user_ids_by_party: Mapping[int, tuple[int, ...]]
    positive_item_ids_by_party: Mapping[int, tuple[int, ...]]
    negative_items: Mapping[int, tuple[int, ...]]
    positive_counts: Mapping[int, int]
    batch_counts: Mapping[int, int]
    round_positive_count: int
    step_count: int
    skipped: bool
    utilization_fraction: tuple[int, int] | None

    def __init__(self, **kwargs: object) -> None:
        raise TypeError("RoundDrainPlan must be built by prepare_round_drain_plan")


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundDrainCursor:
    plan_id: str
    next_step_index: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PartyGroupBatch:
    party_id: int
    start: int
    stop: int
    positive_ids: tuple[int, ...]
    user_ids: tuple[int, ...]
    positive_item_ids: tuple[int, ...]
    negative_item_ids: tuple[tuple[int, ...], ...]
    loss_coefficient: tuple[int, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundDrainStep:
    plan_id: str
    step_index: int
    batches: tuple[PartyGroupBatch, ...]
    requested_source_party_ids: tuple[int, ...]
    source_only_party_ids: tuple[int, ...]
    idle_party_ids: tuple[int, ...]
    bce_row_count: int


def _integer(value: int, name: str, *, minimum: int = 0,
             maximum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in int")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")


def _digest(value: str, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _sequence(value: Sequence, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a repeatable Sequence")


def _party_ids(value: Sequence[int], name: str) -> tuple[int, ...]:
    _sequence(value, name)
    for party_id in value:
        _integer(party_id, f"{name} element")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate party IDs")
    return tuple(sorted(value))


def _id_array(value: np.ndarray, name: str, *, ordered: bool) -> None:
    if (type(value) is not np.ndarray or value.dtype != np.dtype(np.int64)
            or value.dtype.metadata is not None):
        raise TypeError(f"{name} must be a plain metadata-free int64 ndarray")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(value < 0):
        raise ValueError(f"{name} must be nonnegative")
    if ordered and np.any(value[1:] <= value[:-1]):
        raise ValueError(f"{name} must be sorted and unique")


def _domain_size(mapping: dict, name: str) -> int:
    if not isinstance(mapping, dict):
        raise TypeError(f"{name} must be a dict")
    for value in mapping.values():
        _integer(value, f"{name} processed ID")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError(f"{name} processed IDs must bijectively cover 0..n-1")
    return len(mapping)


def _context(bundle: DatasetBundle) -> tuple[set[int], int, int]:
    """Once per plan: only consumed ID columns/domains and split membership."""
    table = bundle.interactions
    for name in ("interaction_ids", "user_ids", "item_ids"):
        _id_array(getattr(table, name), f"interactions.{name}", ordered=False)
    size = len(table.interaction_ids)
    if len(table.user_ids) != size or len(table.item_ids) != size:
        raise ValueError("interaction ID, user and item columns must align")
    if any(value != position for position, value in enumerate(table.interaction_ids)):
        raise ValueError("interaction_ids must equal their positional IDs")
    users = _domain_size(bundle.user_id_map, "user_id_map")
    items = _domain_size(bundle.item_id_map, "item_id_map")
    if np.any(table.user_ids >= users) or np.any(table.item_ids >= items):
        raise ValueError("interaction user/item IDs exceed the processed namespace")
    covered: set[int] = set()
    train: set[int] = set()
    for name in ("train", "valid", "test"):
        ids = getattr(bundle.splits, f"{name}_interaction_ids")
        _id_array(ids, f"splits.{name}", ordered=True)
        if np.any(ids >= size):
            raise ValueError(f"splits.{name} contains an unknown interaction")
        members = set(ids.tolist())
        if covered.intersection(members):
            raise ValueError("splits must be disjoint")
        covered.update(members)
        if name == "train":
            train = members
    if len(covered) != size:
        raise ValueError("splits must cover all interactions")
    return train, users, items


def _selected_histories(bundle: DatasetBundle, roster: dict[int, PartyData],
                        selected: tuple[int, ...], train: set[int],
                        num_users: int) -> dict[int, tuple[int, ...]]:
    histories: dict[int, tuple[int, ...]] = {}
    seen: set[int] = set()
    for party_id in selected:
        party = roster[party_id]
        _id_array(party.train_interaction_ids, f"party {party_id} train", ordered=True)
        _id_array(party.user_ids, f"party {party_id} user_ids", ordered=True)
        if type(party.owns_labels) is not bool:
            raise TypeError(f"party {party_id} owns_labels must be an exact bool")
        if len(party.train_interaction_ids) and not party.owns_labels:
            raise ValueError(f"party {party_id} nonempty training requires owns_labels")
        if np.any(party.user_ids >= num_users):
            raise ValueError(f"party {party_id} user_ids exceed the processed namespace")
        ids = tuple(party.train_interaction_ids.tolist())
        members = set(ids)
        if not members.issubset(train):
            raise ValueError(f"party {party_id} train must belong to global train")
        if seen.intersection(members):
            raise ValueError(f"party {party_id} shares training interactions with another selected party")
        users = set(party.user_ids.tolist())
        if any(int(bundle.interactions.user_ids[i]) not in users for i in ids):
            raise ValueError(f"party {party_id} does not own every training user")
        histories[party_id] = ids
        seen.update(members)
    return histories


def _trace_keys(mapping: Mapping, expected: set[int], name: str) -> None:
    # Validate individually BEFORE set/copy/sort: bool and NumPy keys can alias ints.
    for key in mapping:
        _integer(key, f"{name} key")
    if set(mapping) != expected:
        raise ValueError(f"{name} keys must match the selected trace exactly")


def _trace_tuple(value: tuple[int, ...], name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    for item in value:
        _integer(item, f"{name} element")


def prepare_round_drain_plan(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    *,
    selected_party_ids: Sequence[int],
    ordered_positive_ids: Mapping[int, tuple[int, ...]],
    negative_items: Mapping[int, tuple[int, ...]],
    round_index: int,
    batch_unit: str,
    positive_groups_per_batch: int,
    negatives_per_positive: int,
    negative_replacement: bool,
    trace_id: str,
) -> RoundDrainPlan:
    """Validate consumed domains and freeze a caller-supplied single-round trace.

    Inputs must be stable during this call. This is not an atomic concurrent
    snapshot, seed derivation, known-positive certification or execution engine.
    """
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle")
    _sequence(parties, "parties")
    _sequence(selected_party_ids, "selected_party_ids")
    for value, name in ((ordered_positive_ids, "ordered_positive_ids"),
                        (negative_items, "negative_items")):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a Mapping")
    _integer(round_index, "round_index", maximum=_MAX_INDEX)
    _integer(positive_groups_per_batch, "positive_groups_per_batch", minimum=1,
             maximum=_MAX_INDEX)
    _integer(negatives_per_positive, "negatives_per_positive", maximum=_MAX_INDEX)
    if type(batch_unit) is not str:
        raise TypeError("batch_unit must be an exact built-in str")
    if batch_unit != "positive_group":
        raise ValueError("batch_unit must be positive_group")
    if type(negative_replacement) is not bool:
        raise TypeError("negative_replacement must be an exact bool")
    _digest(trace_id, "trace_id")
    if positive_groups_per_batch * (negatives_per_positive + 1) > _MAX_INDEX:
        raise ValueError("batch expanded-row capacity exceeds int64 offsets")
    roster: dict[int, PartyData] = {}
    for party in parties:
        if not isinstance(party, PartyData):
            raise TypeError("parties must contain PartyData instances")
        _integer(party.party_id, "party_id")
        if party.party_id in roster:
            raise ValueError("parties contains duplicate party_id")
        roster[party.party_id] = party
    selected = _party_ids(selected_party_ids, "selected_party_ids")
    if not set(selected).issubset(roster):
        raise ValueError("selected_party_ids contains a party outside the roster")
    train, num_users, num_items = _context(bundle)
    histories = _selected_histories(bundle, roster, selected, train, num_users)
    counts = {p: len(histories[p]) for p in selected}
    total = sum(counts.values())
    if total > _MAX_INDEX or total * (negatives_per_positive + 1) > _MAX_INDEX:
        raise ValueError("round expanded-row capacity exceeds int64 offsets")
    _trace_keys(ordered_positive_ids, set(selected), "ordered_positive_ids")
    positives: dict[int, tuple[int, ...]] = {}
    for p in selected:
        ids = ordered_positive_ids[p]
        _trace_tuple(ids, f"party {p} ordered_positive_ids")
        if len(ids) != counts[p] or set(ids) != set(histories[p]):
            raise ValueError(f"party {p} positives must permute its entire train history")
        positives[p] = ids
    positive_union = {i for ids in positives.values() for i in ids}
    _trace_keys(negative_items, positive_union, "negative_items")
    negatives: dict[int, tuple[int, ...]] = {}
    for i in sorted(positive_union):
        values = negative_items[i]
        _trace_tuple(values, f"interaction {i} negative_items")
        if len(values) != negatives_per_positive:
            raise ValueError(f"interaction {i} negative count does not match q")
        if any(v >= num_items for v in values):
            raise ValueError(f"interaction {i} negative item outside processed namespace")
        if int(bundle.interactions.item_ids[i]) in values:
            raise ValueError(f"interaction {i} target item cannot be its own negative")
        if not negative_replacement and len(set(values)) != len(values):
            raise ValueError(f"interaction {i} repeated negative requires replacement")
        negatives[i] = values
    users = {p: tuple(int(bundle.interactions.user_ids[i]) for i in positives[p]) for p in selected}
    items = {p: tuple(int(bundle.interactions.item_ids[i]) for i in positives[p]) for p in selected}
    envelope = dict(
        format="thesisfedrec.round_drain_plan", version=1,
        protocol=_PROTOCOL, protocol_version=1, trace_id=trace_id,
        round_index=round_index, batch_unit=batch_unit,
        positive_groups_per_batch=positive_groups_per_batch,
        negatives_per_positive=negatives_per_positive,
        negative_replacement=negative_replacement,
        parties=[dict(party_id=p, positive_ids=positives[p], user_ids=users[p],
                      positive_item_ids=items[p],
                      negative_item_ids=tuple(negatives[i] for i in positives[p]))
                 for p in selected],
    )
    identity = sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    batches = {p: (counts[p] + positive_groups_per_batch - 1) // positive_groups_per_batch
               for p in selected}
    steps = max(batches.values(), default=0)
    losses = tuple(p for p in selected if counts[p])
    # Each proxy owns a new dict; only validated builtin ints/immutable tuples remain.
    fields = dict(
        protocol=_PROTOCOL, protocol_version=1, plan_id=identity, trace_id=trace_id,
        round_index=round_index, batch_unit=batch_unit,
        positive_groups_per_batch=positive_groups_per_batch,
        negatives_per_positive=negatives_per_positive,
        negative_replacement=negative_replacement, selected_party_ids=selected,
        loss_party_ids=losses, positive_ids_by_party=MappingProxyType(positives),
        user_ids_by_party=MappingProxyType(users), positive_item_ids_by_party=MappingProxyType(items),
        negative_items=MappingProxyType(negatives), positive_counts=MappingProxyType(counts),
        batch_counts=MappingProxyType(batches), round_positive_count=total,
        step_count=steps, skipped=total == 0,
        utilization_fraction=(sum(batches.values()), len(losses) * steps) if total else None,
    )
    plan = object.__new__(RoundDrainPlan)
    for name, value in fields.items():
        object.__setattr__(plan, name, value)
    return plan


def _plan(plan: RoundDrainPlan) -> None:
    if not isinstance(plan, RoundDrainPlan):
        raise TypeError("plan must be a prepared RoundDrainPlan")


def _cursor(plan: RoundDrainPlan, cursor: RoundDrainCursor) -> None:
    _plan(plan)
    if not isinstance(cursor, RoundDrainCursor):
        raise TypeError("cursor must be a RoundDrainCursor")
    _digest(cursor.plan_id, "cursor.plan_id")
    if cursor.plan_id != plan.plan_id:
        raise ValueError("cursor.plan_id does not match plan")
    _integer(cursor.next_step_index, "cursor.next_step_index", maximum=plan.step_count)


def start_round_drain(plan: RoundDrainPlan) -> RoundDrainCursor:
    """Return the initial cursor; a skipped plan is already terminal."""
    _plan(plan)
    return RoundDrainCursor(plan_id=plan.plan_id, next_step_index=0)


def peek_round_drain_step(
    plan: RoundDrainPlan,
    cursor: RoundDrainCursor,
    *,
    requested_source_party_ids: Sequence[int],
) -> RoundDrainStep | None:
    """Describe one step without advancing, rescanning inputs or certifying access."""
    _cursor(plan, cursor)
    sources = _party_ids(requested_source_party_ids, "requested_source_party_ids")
    if not set(sources).issubset(plan.selected_party_ids):
        raise ValueError("requested_source_party_ids must belong to the selected roster")
    if cursor.next_step_index == plan.step_count:
        if sources:
            raise ValueError("terminal/skipped round cannot request sources")
        return None
    start = cursor.next_step_index * plan.positive_groups_per_batch
    batches = []
    for p in plan.selected_party_ids:
        stop = min(start + plan.positive_groups_per_batch, plan.positive_counts[p])
        if start >= stop:
            continue
        ids = plan.positive_ids_by_party[p][start:stop]
        batches.append(PartyGroupBatch(
            party_id=p, start=start, stop=stop, positive_ids=ids,
            user_ids=plan.user_ids_by_party[p][start:stop],
            positive_item_ids=plan.positive_item_ids_by_party[p][start:stop],
            negative_item_ids=tuple(plan.negative_items[i] for i in ids),
            loss_coefficient=(stop - start, plan.round_positive_count),
        ))
    loss_ids = {batch.party_id for batch in batches}
    source_ids = set(sources)
    active_ids = loss_ids | source_ids
    return RoundDrainStep(
        plan_id=plan.plan_id, step_index=cursor.next_step_index, batches=tuple(batches),
        requested_source_party_ids=sources,
        source_only_party_ids=tuple(p for p in sources if p not in loss_ids),
        idle_party_ids=tuple(p for p in plan.selected_party_ids if p not in active_ids),
        bce_row_count=sum(batch.stop - batch.start for batch in batches) * (plan.negatives_per_positive + 1),
    )


def advance_round_drain_cursor(
    plan: RoundDrainPlan,
    cursor: RoundDrainCursor,
    *,
    completed_step_index: int,
) -> RoundDrainCursor:
    """Return the next cursor; caller's completion assertion is not commit proof."""
    _cursor(plan, cursor)
    _integer(completed_step_index, "completed_step_index", maximum=_MAX_INDEX)
    if cursor.next_step_index == plan.step_count:
        raise ValueError("cannot advance a terminal/skipped round")
    if completed_step_index != cursor.next_step_index:
        raise ValueError("completed_step_index must equal the current cursor index")
    return RoundDrainCursor(plan_id=plan.plan_id, next_step_index=cursor.next_step_index + 1)
