"""Owned, in-memory round/epoch traces using the existing shared sampler.

Inputs must remain stable during preparation/checking. Identities bind consumed
local data, not source files, authorization or concurrent transactions.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from src.data.contracts import DatasetBundle, PartyData
from src.data.negative_sampling import sample_negative_items


_MAX_ID = 2**63 - 1
_MAX_SEED = 2**64 - 1
_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class TrainingTrace:
    """Factory-only record of immutable builtins; no source or RNG references."""

    protocol_version: int
    trace_id: str
    context_id: str
    run_seed: int
    round_index: int
    local_epoch: int
    negatives_per_positive: int
    replace: bool
    excluded_splits: tuple[str, str, str]
    selected_party_ids: tuple[int, ...]
    positive_order_seeds: tuple[tuple[int, int], ...]
    negative_seed: int
    ordered_positive_ids: tuple[tuple[int, tuple[int, ...]], ...]
    negative_items: tuple[tuple[int, tuple[int, ...]], ...]

    def __init__(self, **kwargs: object) -> None:
        raise TypeError("TrainingTrace must be built by build_training_trace")


def _integer(value: int, name: str, maximum: int = _MAX_ID) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in int")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} is outside 0..{maximum}")


def _digest(value: str, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _sequence(value: Sequence, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a finite repeatable Sequence")


def _tuple(value: tuple, name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")


def _id_tuple(value: tuple[int, ...], name: str, *, ordered: bool) -> None:
    _tuple(value, name)
    for item in value:
        _integer(item, f"{name} element")
    if ordered and any(a >= b for a, b in zip(value, value[1:])):
        raise ValueError(f"{name} must be sorted and unique")


def _id_array(value: np.ndarray, name: str, *, ordered: bool) -> None:
    if (type(value) is not np.ndarray or value.dtype != np.dtype(np.int64)
            or not value.dtype.isnative or value.dtype.metadata is not None):
        raise TypeError(f"{name} must be a plain native metadata-free int64 array")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(value < 0):
        raise ValueError(f"{name} must be nonnegative")
    if ordered and np.any(value[1:] <= value[:-1]):
        raise ValueError(f"{name} must be sorted and unique")


def _mapping_entries(mapping: dict, name: str) -> list[dict]:
    if not isinstance(mapping, dict):
        raise TypeError(f"{name} must be a dict")
    # Validate before sorting, indexing or set conversion (bool/int aliases).
    for raw, processed in mapping.items():
        if type(raw) not in (str, int):
            raise TypeError(f"{name} raw keys must be exact str or int")
        _integer(processed, f"{name} processed ID")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError(f"{name} must bijectively cover 0..size-1")
    return [dict(processed_id=p, raw_type=type(raw).__name__, raw_id=raw)
            for raw, p in sorted(mapping.items(), key=lambda pair: pair[1])]


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _seed(run_seed: int, round_index: int, local_epoch: int,
          stream: str, party_id: int | None = None) -> int:
    envelope = dict(format="thesisfedrec.training_trace.seed", version=1,
                    run_seed=run_seed, round_index=round_index,
                    local_epoch=local_epoch, stream=stream)
    if stream == "positive-order":
        envelope["party_id"] = party_id
    return int.from_bytes(sha256(_canonical(envelope)).digest()[:8], "big", signed=False)


def _settings(run_seed: int, round_index: int, local_epoch: int,
              q: int, excluded_splits: tuple[str, str, str], replace: bool) -> None:
    for value, name in ((run_seed, "run_seed"), (round_index, "round_index"),
                        (local_epoch, "local_epoch"), (q, "negatives_per_positive")):
        _integer(value, name)
    if type(replace) is not bool:
        raise TypeError("replace must be an exact bool")
    _tuple(excluded_splits, "excluded_splits")
    if any(type(name) is not str for name in excluded_splits):
        raise TypeError("excluded_splits must contain exact str values")
    if excluded_splits != _SPLITS:
        raise ValueError("excluded_splits must equal ('train', 'valid', 'test')")


def _context(bundle: DatasetBundle, parties: Sequence[PartyData],
             selected_party_ids: Sequence[int], q: int
             ) -> tuple[str, dict[int, tuple[int, ...]], int]:
    """Validate/hash only consumed domains; temporary full-source lists released."""
    if type(bundle) is not DatasetBundle:
        raise TypeError("bundle must be an exact DatasetBundle")
    _sequence(parties, "parties")
    _sequence(selected_party_ids, "selected_party_ids")
    roster = {}
    for party in parties:
        if type(party) is not PartyData:
            raise TypeError("parties must contain exact PartyData records")
        _integer(party.party_id, "party_id")
        if party.party_id in roster:
            raise ValueError("roster contains a duplicate party ID")
        roster[party.party_id] = party
    for party_id in selected_party_ids:
        _integer(party_id, "selected_party_ids element")
    if len(set(selected_party_ids)) != len(selected_party_ids):
        raise ValueError("selected_party_ids contains duplicates")
    selected = tuple(sorted(selected_party_ids))
    if not set(selected).issubset(roster):
        raise ValueError("selected party is outside the roster")

    table = bundle.interactions
    for name in ("interaction_ids", "user_ids", "item_ids"):
        _id_array(getattr(table, name), f"interactions.{name}", ordered=False)
    size = len(table.interaction_ids)
    if len(table.user_ids) != size or len(table.item_ids) != size:
        raise ValueError("interaction ID/user/item columns must align")
    if any(value != position for position, value in enumerate(table.interaction_ids)):
        raise ValueError("interaction_ids must equal 0..T-1 in order")
    user_map = _mapping_entries(bundle.user_id_map, "user_id_map")
    item_map = _mapping_entries(bundle.item_id_map, "item_id_map")
    if np.any(table.user_ids >= len(user_map)) or np.any(table.item_ids >= len(item_map)):
        raise ValueError("interaction user/item ID is outside the processed namespace")
    split_lists = {}
    covered = set()
    for name in _SPLITS:
        ids = getattr(bundle.splits, f"{name}_interaction_ids")
        _id_array(ids, f"splits.{name}", ordered=True)
        if np.any(ids >= size):
            raise ValueError(f"splits.{name} has an out-of-domain interaction")
        split_lists[name] = ids.tolist()
        members = set(split_lists[name])
        if covered.intersection(members):
            raise ValueError("global splits must be disjoint")
        covered.update(members)
    if len(covered) != size:
        raise ValueError("global splits must cover every interaction")

    histories = {}
    party_entries = []
    seen = set()
    global_train = set(split_lists["train"])
    for party_id in selected:
        party = roster[party_id]
        _id_array(party.train_interaction_ids, "party train_interaction_ids", ordered=True)
        _id_array(party.user_ids, "party user_ids", ordered=True)
        if type(party.owns_labels) is not bool:
            raise TypeError("party owns_labels must be an exact bool")
        if len(party.train_interaction_ids) and not party.owns_labels:
            raise ValueError("nonempty party train history requires owns_labels=True")
        if np.any(party.user_ids >= len(user_map)):
            raise ValueError("party user ID is outside the processed namespace")
        ids = tuple(party.train_interaction_ids.tolist())
        members = set(ids)
        if not members.issubset(global_train):
            raise ValueError("party train must belong to global train")
        if seen.intersection(members):
            raise ValueError("selected parties share training interactions")
        users = party.user_ids.tolist()
        user_set = set(users)
        if any(int(table.user_ids[i]) not in user_set for i in ids):
            raise ValueError("party does not own the user of every training interaction")
        seen.update(members)
        histories[party_id] = ids
        party_entries.append(dict(party_id=party_id, user_ids=users,
                                  train_interaction_ids=list(ids), owns_labels=party.owns_labels))
    total = sum(len(ids) for ids in histories.values())
    if total > _MAX_ID or total * (q + 1) > _MAX_ID:
        raise ValueError("selected positive/expanded-row capacity exceeds int64")
    context = dict(format="thesisfedrec.training_trace.context", version=1,
                   user_map=user_map, item_map=item_map,
                   interaction_ids=table.interaction_ids.tolist(),
                   user_ids=table.user_ids.tolist(), item_ids=table.item_ids.tolist(),
                   splits=split_lists, parties=party_entries)
    return sha256(_canonical(context)).hexdigest(), histories, len(item_map)


def _semantic(fields: dict) -> dict:
    return dict(
        format="thesisfedrec.training_trace", version=1, context_id=fields["context_id"],
        run_seed=fields["run_seed"], round_index=fields["round_index"],
        local_epoch=fields["local_epoch"], negatives_per_positive=fields["negatives_per_positive"],
        replace=fields["replace"], excluded_splits=list(fields["excluded_splits"]),
        selected_party_ids=list(fields["selected_party_ids"]),
        positive_order_seeds=[dict(party_id=p, seed=s) for p, s in fields["positive_order_seeds"]],
        negative_seed=fields["negative_seed"],
        ordered_positive_ids=[dict(party_id=p, interaction_ids=list(ids))
                              for p, ids in fields["ordered_positive_ids"]],
        negative_items=[dict(interaction_id=i, item_ids=list(ids))
                        for i, ids in fields["negative_items"]],
    )


def build_training_trace(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    *,
    selected_party_ids: Sequence[int],
    round_index: int,
    local_epoch: int,
    run_seed: int,
    negatives_per_positive: int,
    excluded_splits: tuple[str, str, str],
    replace: bool,
) -> TrainingTrace:
    """Sample once per nonempty round/epoch with q>0; never per minibatch.

    The existing sampler still rebuilds all global known-positive sets. This
    producer neither indexes them persistently nor detects repeated builds.
    """
    _settings(run_seed, round_index, local_epoch, negatives_per_positive, excluded_splits, replace)
    context_id, histories, _ = _context(bundle, parties, selected_party_ids, negatives_per_positive)
    positive_seeds = tuple((p, _seed(run_seed, round_index, local_epoch, "positive-order", p))
                           for p in histories)
    negative_seed = _seed(run_seed, round_index, local_epoch, "negative-items")
    ordered = tuple((p, tuple(int(i) for i in np.random.default_rng(seed).permutation(histories[p])))
                    for p, seed in positive_seeds)
    positive_ids = sorted(i for ids in histories.values() for i in ids)
    if positive_ids and negatives_per_positive:
        negatives = sample_negative_items(
            bundle, positive_ids, negatives_per_positive=negatives_per_positive,
            strategy="uniform", seed=negative_seed, excluded_splits=excluded_splits, replace=replace,
        )
    else:
        negatives = {i: () for i in positive_ids}
    fields = dict(protocol_version=1, context_id=context_id,
                  run_seed=run_seed, round_index=round_index, local_epoch=local_epoch,
                  negatives_per_positive=negatives_per_positive, replace=replace,
                  excluded_splits=_SPLITS, selected_party_ids=tuple(histories),
                  positive_order_seeds=positive_seeds, negative_seed=negative_seed,
                  ordered_positive_ids=ordered,
                  negative_items=tuple((i, negatives[i]) for i in positive_ids))
    fields["trace_id"] = sha256(_canonical(_semantic(fields))).hexdigest()
    result = object.__new__(TrainingTrace)
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    return result


def _trace_fields(trace: TrainingTrace) -> dict:
    if type(trace) is not TrainingTrace:
        raise TypeError("trace must be an exact factory-created TrainingTrace")
    try:
        fields = {name: getattr(trace, name) for name in TrainingTrace.__slots__}
    except AttributeError as error:
        raise ValueError("trace has incomplete factory fields") from error
    _integer(trace.protocol_version, "protocol_version")
    if trace.protocol_version != 1:
        raise ValueError("unsupported trace protocol version")
    _digest(trace.trace_id, "trace_id")
    _digest(trace.context_id, "context_id")
    _settings(trace.run_seed, trace.round_index, trace.local_epoch,
              trace.negatives_per_positive, trace.excluded_splits, trace.replace)
    _id_tuple(trace.selected_party_ids, "selected_party_ids", ordered=True)
    _integer(trace.negative_seed, "negative_seed", _MAX_SEED)
    for name in ("positive_order_seeds", "ordered_positive_ids", "negative_items"):
        pairs = fields[name]
        _tuple(pairs, name)
        for pair in pairs:
            _tuple(pair, name + " pair")
            if len(pair) != 2:
                raise ValueError(f"{name} pairs must contain exactly two entries")
            _integer(pair[0], name + " key")
            if name == "positive_order_seeds":
                _integer(pair[1], "positive seed", _MAX_SEED)
            else:
                _id_tuple(pair[1], name + " IDs", ordered=False)
        _id_tuple(tuple(pair[0] for pair in pairs), name + " keys", ordered=True)
    for name in ("positive_order_seeds", "ordered_positive_ids"):
        if tuple(p for p, _ in fields[name]) != trace.selected_party_ids:
            raise ValueError(f"{name} must cover the selected parties exactly")
    expected_seeds = tuple((p, _seed(trace.run_seed, trace.round_index, trace.local_epoch,
                                    "positive-order", p)) for p in trace.selected_party_ids)
    if trace.positive_order_seeds != expected_seeds or trace.negative_seed != _seed(
            trace.run_seed, trace.round_index, trace.local_epoch, "negative-items"):
        raise ValueError("trace seeds do not match the named seed derivation")
    positives = [i for _, ids in trace.ordered_positive_ids for i in ids]
    if len(positives) != len(set(positives)):
        raise ValueError("trace positives must occur exactly once")
    if len(positives) > _MAX_ID or len(positives) * (trace.negatives_per_positive + 1) > _MAX_ID:
        raise ValueError("trace expanded-row capacity exceeds int64")
    if tuple(i for i, _ in trace.negative_items) != tuple(sorted(positives)):
        raise ValueError("trace negatives must cover the positive union exactly")
    for _, ids in trace.negative_items:
        if len(ids) != trace.negatives_per_positive:
            raise ValueError("trace negative row length does not match q")
        if any(a > b for a, b in zip(ids, ids[1:])):
            raise ValueError("trace negative rows must retain sampler sorted order")
        if not trace.replace and len(set(ids)) != len(ids):
            raise ValueError("duplicate negative requires replace=True")
    if sha256(_canonical(_semantic(fields))).hexdigest() != trace.trace_id:
        raise ValueError("trace_id does not match the realized trace")
    return fields


def check_training_trace_context(
    trace: TrainingTrace, bundle: DatasetBundle, parties: Sequence[PartyData],
) -> None:
    """Validate realized integrity and current consumed context without RNG.

    Call once at a context boundary, not per batch. No sampling/permutation is
    replayed; hashes are trusted local bindings, not authentication signatures.
    """
    _trace_fields(trace)
    context_id, histories, num_items = _context(
        bundle, parties, trace.selected_party_ids, trace.negatives_per_positive)
    if context_id != trace.context_id:
        raise ValueError("training trace context does not match current consumed data")
    for p, ids in trace.ordered_positive_ids:
        if len(ids) != len(histories[p]) or set(ids) != set(histories[p]):
            raise ValueError("trace positives must permute the complete selected train histories")
    # Structural consistency with current consumed data, not re-execution of RNG.
    known = {}
    if trace.negatives_per_positive and trace.negative_items:
        for user, item in zip(bundle.interactions.user_ids, bundle.interactions.item_ids):
            known.setdefault(int(user), set()).add(int(item))
    for interaction_id, items in trace.negative_items:
        user = int(bundle.interactions.user_ids[interaction_id])
        if any(item >= num_items or item in known.get(user, ()) for item in items):
            raise ValueError("trace negatives contain an out-of-domain or known-positive item")
