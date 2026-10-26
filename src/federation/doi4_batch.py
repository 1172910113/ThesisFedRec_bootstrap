"""Owned numerical DOI4 deliveries and explicit, single-step source geometry.

This module does not select sources or authenticate permissions, message versions,
or mathematical provenance. Callers keep inputs stable during each call and own
those checks before supplying this narrow numerical boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import torch


_MAX_INT = 2**63 - 1
_PAYLOAD_KEYS = {
    "entity_kind", "entity_ids", "local_field_ids", "binary_values",
    "valid_unit_mask",
}


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class BinaryTensorBatch:
    """Fresh CPU tensors; pass only the three unit tensors to mathematics."""

    entity_ids: torch.Tensor
    local_field_ids: torch.Tensor
    binary_values: torch.Tensor
    valid_unit_mask: torch.Tensor

    def __new__(cls, *args: object, **kwargs: object) -> BinaryTensorBatch:
        raise TypeError("use tensorize_binary_delivery()")


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class Doi4BatchRequests:
    """Immutable built-in geometry, not a source-version or permission proof."""

    context_id: str
    public_item_count: int
    online_party_ids: tuple[int, ...]
    receiver_party_ids: tuple[int, ...]
    receiver_item_ids: tuple[tuple[int, ...], ...]
    source_party_ids_by_receiver: tuple[tuple[int, ...], ...]
    source_party_ids: tuple[int, ...]
    source_item_ids: tuple[tuple[int, ...], ...]
    receiver_source_row_indices: tuple[tuple[tuple[int, ...], ...], ...]

    def __new__(cls, *args: object, **kwargs: object) -> Doi4BatchRequests:
        raise TypeError("use prepare_doi4_batch_requests()")


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class ReceiverSlotBatch:
    """Owned packed tensors plus trusted row/column sidecars, not features."""

    receiver_party_id: int
    entity_ids: tuple[int, ...]
    slot_source_party_ids: tuple[int, ...]
    slot_indices: tuple[int, ...]
    compatible_tokens: torch.Tensor
    valid_token_mask: torch.Tensor

    def __new__(cls, *args: object, **kwargs: object) -> ReceiverSlotBatch:
        raise TypeError("use pack_doi4_source_slots()")


def _record(record_type: type, **fields: object) -> object:
    record = object.__new__(record_type)
    for name, value in fields.items():
        object.__setattr__(record, name, value)
    return record


def _integer(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in int")
    if not minimum <= value <= _MAX_INT:
        raise ValueError(f"{name} must be in [{minimum}, 2**63-1]")


def _context(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase 64-character hex")


def _capacity(shape: tuple[int, ...], name: str) -> None:
    # Protect dimensions and necessary contiguous strides even with zero axes.
    product = 1
    for extent in shape:
        if extent < 0 or extent > _MAX_INT:
            raise ValueError(f"{name} has a dimension outside int64 capacity")
        product *= max(1, extent)
        if product > _MAX_INT:
            raise ValueError(f"{name} exceeds int64 dimension/stride capacity")


def _ids(
    values: object, name: str, *, item_count: int | None = None,
    sorted_unique: bool = False,
) -> tuple[int, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    for value in values:
        _integer(value, f"{name} member")
        if item_count is not None and value >= item_count:
            raise ValueError(f"{name} member is outside the public item domain")
    if sorted_unique and any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be sorted and unique")
    return tuple(value for value in values)


def _array(value: object, name: str, dtype: np.dtype, rank: int) -> None:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a plain ndarray")
    actual = value.dtype
    if (
        actual != dtype or not actual.isnative or actual.metadata is not None
        or actual.fields is not None or actual.subdtype is not None
    ):
        raise TypeError(f"{name} has an unsupported dtype")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")


def _owned_tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    # Canonicalize negative/read-only strides, then copy into independent storage.
    canonical = np.array(value, dtype=value.dtype, order="C", copy=True)
    return torch.tensor(canonical, dtype=dtype, device="cpu", requires_grad=False)


def tensorize_binary_delivery(
    payload: dict[str, object], *, expected_entity_ids: tuple[int, ...],
    public_item_count: int, local_field_count: int,
) -> BinaryTensorBatch:
    """Validate the exact adapter packet before creating four independent tensors.

    Entity IDs are trusted sidecars. Neither this packet nor its shape authenticates
    the runtime/map binding or ACL; callers verify them before delivery.
    """
    _integer(public_item_count, "public_item_count")
    _integer(local_field_count, "local_field_count")
    expected = _ids(expected_entity_ids, "expected_entity_ids", item_count=public_item_count)
    if type(payload) is not dict:
        raise TypeError("payload must be an exact dict")
    if any(type(key) is not str for key in payload):
        raise TypeError("payload keys must be exact built-in str")
    if set(payload) != _PAYLOAD_KEYS:
        raise ValueError("payload must contain exactly the five binary delivery keys")
    if type(payload["entity_kind"]) is not str:
        raise TypeError("entity_kind must be an exact built-in str")
    if payload["entity_kind"] != "item":
        raise ValueError("entity_kind must be item")
    ids = payload["entity_ids"]
    local_ids = payload["local_field_ids"]
    codes = payload["binary_values"]
    valid = payload["valid_unit_mask"]
    _array(ids, "entity_ids", np.dtype("int64"), 1)
    _array(local_ids, "local_field_ids", np.dtype("int64"), 2)
    _array(codes, "binary_values", np.dtype("int64"), 2)
    _array(valid, "valid_unit_mask", np.dtype("bool"), 2)
    shape = (len(expected), local_field_count)
    _capacity((len(expected),), "entity_ids")
    _capacity(shape, "binary unit output")
    if ids.shape != (len(expected),):
        raise ValueError("entity_ids shape does not match expected rows")
    if any(value.shape != shape for value in (local_ids, codes, valid)):
        raise ValueError("unit array shapes must match [E, J]")
    if any(int(actual) != wanted for actual, wanted in zip(ids, expected)):
        raise ValueError("entity_ids do not equal the expected original rows")
    if shape[0] and shape[1]:
        columns = np.arange(local_field_count, dtype=np.int64)[None, :]
        if np.any(valid & (local_ids != columns)):
            raise ValueError("valid units must use their dense local column ID")
        if np.any(valid & (codes != 0) & (codes != 1)):
            raise ValueError("valid binary values must be exactly 0 or 1")
        if np.any(~valid & ((local_ids != -1) | (codes != 0))):
            raise ValueError("invalid units must use exactly -1/0/False")
    return _record(
        BinaryTensorBatch,
        entity_ids=_owned_tensor(ids, torch.int64),
        local_field_ids=_owned_tensor(local_ids, torch.int64),
        binary_values=_owned_tensor(codes, torch.int64),
        valid_unit_mask=_owned_tensor(valid, torch.bool),
    )


def _geometry(
    receiver_items: tuple[tuple[int, ...], ...],
    receiver_sources: tuple[tuple[int, ...], ...],
) -> tuple[
    tuple[int, ...], tuple[tuple[int, ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
]:
    source_ids = tuple(sorted({source for sources in receiver_sources for source in sources}))
    items_by_source = {source: set() for source in source_ids}
    for items, sources in zip(receiver_items, receiver_sources):
        for source in sources:
            items_by_source[source].update(items)
    source_items = tuple(tuple(sorted(items_by_source[source])) for source in source_ids)
    row_by_source = {
        source: {item: row for row, item in enumerate(items)}
        for source, items in zip(source_ids, source_items)
    }
    inverses = tuple(
        tuple(tuple(row_by_source[source][item] for item in items) for source in sources)
        for items, sources in zip(receiver_items, receiver_sources)
    )
    return source_ids, source_items, inverses


def prepare_doi4_batch_requests(
    receiver_item_ids: dict[int, tuple[int, ...]], *,
    source_party_ids_by_receiver: dict[int, tuple[int, ...]],
    online_party_ids: tuple[int, ...], public_item_count: int, context_id: str,
) -> Doi4BatchRequests:
    """Own one caller-declared pool's sorted unique source rows and row inverses.

    No own source is added, and no source is selected from data or availability.
    Party IDs are independent of user IDs. Repeated receiver rows remain repeated.
    """
    _integer(public_item_count, "public_item_count")
    _context(context_id, "context_id")
    online = _ids(online_party_ids, "online_party_ids", sorted_unique=True)
    if type(receiver_item_ids) is not dict or type(source_party_ids_by_receiver) is not dict:
        raise TypeError("receiver item and source plans must be exact dicts")
    for receiver in receiver_item_ids:
        _integer(receiver, "receiver party ID")
    for receiver in source_party_ids_by_receiver:
        _integer(receiver, "source-plan receiver party ID")
    if set(receiver_item_ids) != set(source_party_ids_by_receiver):
        raise ValueError("receiver item and source plans must have identical keys")
    receivers = tuple(sorted(receiver_item_ids))
    online_set = set(online)
    if any(receiver not in online_set for receiver in receivers):
        raise ValueError("every receiver must be in the explicit online roster")
    receiver_items = tuple(
        _ids(receiver_item_ids[receiver], "receiver item IDs", item_count=public_item_count)
        for receiver in receivers
    )
    receiver_sources = tuple(
        _ids(source_party_ids_by_receiver[receiver], "receiver source IDs", sorted_unique=True)
        for receiver in receivers
    )
    if any(source not in online_set for sources in receiver_sources for source in sources):
        raise ValueError("every declared source must be in the explicit online roster")
    source_ids, source_items, inverses = _geometry(receiver_items, receiver_sources)
    return _record(
        Doi4BatchRequests,
        context_id=context_id, public_item_count=public_item_count,
        online_party_ids=online, receiver_party_ids=receivers,
        receiver_item_ids=receiver_items, source_party_ids_by_receiver=receiver_sources,
        source_party_ids=source_ids, source_item_ids=source_items,
        receiver_source_row_indices=inverses,
    )


def _validate_requests(requests: object) -> None:
    if type(requests) is not Doi4BatchRequests:
        raise TypeError("requests must be an exact Doi4BatchRequests")
    try:
        values = (
            requests.context_id, requests.public_item_count, requests.online_party_ids,
            requests.receiver_party_ids, requests.receiver_item_ids,
            requests.source_party_ids_by_receiver, requests.source_party_ids,
            requests.source_item_ids, requests.receiver_source_row_indices,
        )
    except AttributeError as error:
        raise ValueError("requests is an unprepared record") from error
    context, item_count, online, receivers, items, pools, sources, source_items, inverses = values
    _context(context, "requests.context_id")
    _integer(item_count, "requests.public_item_count")
    _ids(online, "requests.online_party_ids", sorted_unique=True)
    _ids(receivers, "requests.receiver_party_ids", sorted_unique=True)
    _ids(sources, "requests.source_party_ids", sorted_unique=True)
    for value, name in (
        (items, "receiver_item_ids"), (pools, "source_party_ids_by_receiver"),
        (source_items, "source_item_ids"), (inverses, "receiver_source_row_indices"),
    ):
        if type(value) is not tuple:
            raise TypeError(f"requests.{name} must be an exact tuple")
    if len(items) != len(receivers) or len(pools) != len(receivers) or len(inverses) != len(receivers):
        raise ValueError("requests receiver fields have inconsistent lengths")
    if len(source_items) != len(sources):
        raise ValueError("requests source fields have inconsistent lengths")
    online_set = set(online)
    if any(receiver not in online_set for receiver in receivers):
        raise ValueError("requests receiver is outside the online roster")
    for rows, pool, receiver_inverse in zip(items, pools, inverses):
        _ids(rows, "requests receiver item IDs", item_count=item_count)
        _ids(pool, "requests receiver source IDs", sorted_unique=True)
        if any(source not in online_set for source in pool):
            raise ValueError("requests source is outside the online roster")
        if type(receiver_inverse) is not tuple:
            raise TypeError("requests receiver inverse must be an exact tuple")
        if len(receiver_inverse) != len(pool):
            raise ValueError("requests inverse source count is inconsistent")
        for inverse in receiver_inverse:
            _ids(inverse, "requests inverse row indices")
            if len(inverse) != len(rows):
                raise ValueError("requests inverse row count is inconsistent")
    for rows in source_items:
        _ids(rows, "requests source item IDs", item_count=item_count, sorted_unique=True)
    expected_sources, expected_items, expected_inverses = _geometry(items, pools)
    if sources != expected_sources or source_items != expected_items:
        raise ValueError("requests must contain the exact declared source/item unions")
    if inverses != expected_inverses:
        raise ValueError("requests inverse must point to each receiver's actual item row")


def _cpu_tensor(value: object, name: str, dtype: torch.dtype, rank: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.is_nested:
        raise TypeError(f"{name} must be a dense, non-nested Tensor")
    if value.layout is not torch.strided or value.device.type != "cpu" or value.dtype != dtype:
        raise TypeError(f"{name} must be a dense strided CPU {dtype} Tensor")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")


def _validate_publications(
    requests: Doi4BatchRequests, publications: object, *, slot_count: int,
    compatible_dim: int, dtype: torch.dtype,
) -> None:
    if type(publications) is not dict:
        raise TypeError("source_publications must be an exact dict")
    for source in publications:
        _integer(source, "publication source party ID")
    if set(publications) != set(requests.source_party_ids):
        raise ValueError("publications must contain exactly the planned sources")
    for source, expected_ids in zip(requests.source_party_ids, requests.source_item_ids):
        publication = publications[source]
        if type(publication) is not tuple:
            raise TypeError("each source publication must be an exact tuple")
        if len(publication) != 3:
            raise ValueError("each source publication must contain IDs, tokens and availability")
        entity_ids, tokens, available = publication
        actual_ids = _ids(entity_ids, "publication entity IDs", item_count=requests.public_item_count)
        if actual_ids != expected_ids:
            raise ValueError("publication entity IDs must equal the full source unique rows")
        _cpu_tensor(available, "source availability", torch.bool, 1)
        if available.shape != (len(expected_ids),):
            raise ValueError("source availability shape must match source unique rows")
        if tokens is None:
            if bool(torch.any(available)):
                raise ValueError("a None publication must have entirely false availability")
            continue
        _cpu_tensor(tokens, "source tokens", dtype, 3)
        if tokens.shape != (len(expected_ids), slot_count, compatible_dim):
            raise ValueError("source tokens shape must match [U_s, K, h]")
        if not bool(torch.all(torch.isfinite(tokens))):
            raise ValueError("all source token cells must be finite, including inactive cells")


def pack_doi4_source_slots(
    requests: Doi4BatchRequests, *,
    source_publications: dict[int, tuple[tuple[int, ...], torch.Tensor | None, torch.Tensor]],
    slot_count: int, compatible_dim: int, dtype: torch.dtype, expected_context_id: str,
) -> tuple[ReceiverSlotBatch, ...]:
    """Validate all sources once, then gather original rows into ordered slots.

    Present H tensors retain their gather/where/cat autograd path, including empty
    or all-false selections. None declares absence and contributes plain zeros.
    The function neither runs backward nor decides any downstream kernel bypass.
    """
    _integer(slot_count, "slot_count", minimum=1)
    _integer(compatible_dim, "compatible_dim", minimum=1)
    if type(dtype) is not torch.dtype or dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    _context(expected_context_id, "expected_context_id")
    _validate_requests(requests)
    if requests.context_id != expected_context_id:
        raise ValueError("requests context does not equal the expected caller tag")
    # K participates only in declared source shapes and packed S*K columns.
    for source_items in requests.source_item_ids:
        _capacity((len(source_items), slot_count, compatible_dim), "source token shape")
    for items, sources in zip(requests.receiver_item_ids, requests.source_party_ids_by_receiver):
        columns = len(sources) * slot_count
        _capacity((len(items), columns, compatible_dim), "receiver token shape")
        _capacity((len(items), columns), "receiver mask shape")
    _validate_publications(
        requests, source_publications, slot_count=slot_count,
        compatible_dim=compatible_dim, dtype=dtype,
    )

    results = []
    for receiver, items, sources, inverses in zip(
        requests.receiver_party_ids, requests.receiver_item_ids,
        requests.source_party_ids_by_receiver, requests.receiver_source_row_indices,
    ):
        token_parts = []
        mask_parts = []
        for source, inverse in zip(sources, inverses):
            _, tokens, available = source_publications[source]
            row_index = torch.tensor(inverse, dtype=torch.int64, device="cpu")
            selected_available = torch.index_select(available, 0, row_index)
            mask_parts.append(torch.repeat_interleave(selected_available, slot_count, dim=0).reshape(
                len(items), slot_count,
            ))
            if tokens is None:
                selected = torch.zeros((len(items), slot_count, compatible_dim), dtype=dtype, device="cpu")
            else:
                selected = torch.index_select(tokens, 0, row_index)
                selected = torch.where(
                    selected_available[:, None, None], selected,
                    torch.zeros((), dtype=dtype, device="cpu"),
                )
            token_parts.append(selected)
        if sources:
            packed = torch.cat(token_parts, dim=1)
            mask = torch.cat(mask_parts, dim=1)
        else:
            packed = torch.empty((len(items), 0, compatible_dim), dtype=dtype, device="cpu")
            mask = torch.empty((len(items), 0), dtype=torch.bool, device="cpu")
        results.append(_record(
            ReceiverSlotBatch,
            receiver_party_id=receiver, entity_ids=tuple(item for item in items),
            slot_source_party_ids=tuple(source for source in sources for _ in range(slot_count)),
            slot_indices=tuple(slot for _ in sources for slot in range(slot_count)),
            compatible_tokens=packed, valid_token_mask=mask,
        ))
    return tuple(results)
