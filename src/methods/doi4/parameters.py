"""Pure shape and ownership declarations for the partial pri-none DOI4 path."""

from __future__ import annotations

from dataclasses import dataclass


_INT64_MAX = 2**63 - 1
_CENTER_NAMES = (
    "item_embeddings",
    "queries",
    "fusion_w_query",
    "fusion_w_key",
    "fusion_w_value",
    "fusion_w_output",
    "w_fused",
    "score_weight",
    "score_bias",
)
_PARTY_NAMES = (
    "user_embeddings",
    "field_embeddings",
    "value_embeddings",
    "readout_w_key",
    "readout_w_value",
)
_ZERO_ROW_TABLES = ("item_embeddings", "user_embeddings", "field_embeddings")


def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in int")
    if not minimum <= value <= _INT64_MAX:
        raise ValueError(f"{name} must be in [{minimum}, 2**63 - 1]")


@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterSpec:
    """One immutable declaration, without tensors or parameter ownership state.

    Names fix each record's owner and rank. Only item/user/field table row
    counts may be zero; all other dimensions are positive. Every extent and
    the exact Python-integer element count must fit signed int64 capacity.
    Declarations do not allocate tables, initialize state, or bind field maps.
    """

    name: str
    shape: tuple[int, ...]
    owner_scope: str
    party_id: int | None
    trainable: bool

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("name must be an exact built-in str")
        if self.name not in _CENTER_NAMES and self.name not in _PARTY_NAMES:
            raise ValueError("name must be one of the 14 DOI4 parameter labels")
        if type(self.owner_scope) is not str:
            raise TypeError("owner_scope must be an exact built-in str")
        expected_owner = "center" if self.name in _CENTER_NAMES else "party"
        if self.owner_scope != expected_owner:
            raise ValueError(f"{self.name} must have owner_scope={expected_owner!r}")
        if self.owner_scope == "center":
            if self.party_id is not None:
                _validate_integer("party_id", self.party_id, minimum=0)
                raise ValueError("center parameters must have party_id=None")
        else:
            _validate_integer("party_id", self.party_id, minimum=0)
        if type(self.trainable) is not bool:
            raise TypeError("trainable must be an exact built-in bool")
        if not self.trainable:
            raise ValueError("this DOI4 unit declares only trainable parameters")
        if type(self.shape) is not tuple:
            raise TypeError("shape must be an exact built-in tuple")
        rank = 0 if self.name == "score_bias" else 1 if self.name == "score_weight" else 2
        if len(self.shape) != rank:
            raise ValueError(f"{self.name} must have rank {rank}")
        numel = 1
        for axis, extent in enumerate(self.shape):
            minimum = 0 if axis == 0 and self.name in _ZERO_ROW_TABLES else 1
            _validate_integer(f"shape[{axis}]", extent, minimum=minimum)
            numel *= extent
        if self.name == "value_embeddings" and self.shape[0] != 2:
            raise ValueError("value_embeddings must have exactly two rows")
        if self.name in ("fusion_w_key", "fusion_w_value", "fusion_w_output"):
            if self.shape[0] != self.shape[1]:
                raise ValueError(f"{self.name} must have square shape [h,h]")
        if numel > _INT64_MAX:
            raise ValueError("shape element count must not exceed 2**63 - 1")


def doi4_parameter_specs(
    *,
    party_id: int,
    public_item_count: int,
    local_user_count: int,
    local_field_count: int,
    item_dim: int,
    unit_dim: int,
    compatible_dim: int,
    slot_count: int,
) -> tuple[ParameterSpec, ...]:
    """Declare the fixed nine center and five party-local parameters.

    All arguments are exact built-in integers within signed int64 capacity;
    IDs/counts may be zero and widths/slot count must be positive. Zero-row
    declarations establish no trainable-loss guarantee. The nine center
    records are equal across party calls, not instructions to allocate copies.
    User row counts and party IDs belong to separate namespaces. This pure
    declaration chooses no dtype, initialization, RNG, optimizer, map binding,
    resource budget, or scientific profile and declares no private parameters.
    """
    for name, value in (
        ("party_id", party_id),
        ("public_item_count", public_item_count),
        ("local_user_count", local_user_count),
        ("local_field_count", local_field_count),
    ):
        _validate_integer(name, value, minimum=0)
    for name, value in (
        ("item_dim", item_dim),
        ("unit_dim", unit_dim),
        ("compatible_dim", compatible_dim),
        ("slot_count", slot_count),
    ):
        _validate_integer(name, value, minimum=1)

    return (
        ParameterSpec(
            name="item_embeddings", shape=(public_item_count, item_dim),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="queries", shape=(slot_count, compatible_dim),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="fusion_w_query", shape=(compatible_dim, item_dim),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="fusion_w_key", shape=(compatible_dim, compatible_dim),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="fusion_w_value", shape=(compatible_dim, compatible_dim),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="fusion_w_output", shape=(compatible_dim, compatible_dim),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="w_fused", shape=(item_dim, compatible_dim),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="score_weight", shape=(item_dim,),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="score_bias", shape=(),
            owner_scope="center", party_id=None, trainable=True,
        ),
        ParameterSpec(
            name="user_embeddings", shape=(local_user_count, item_dim),
            owner_scope="party", party_id=party_id, trainable=True,
        ),
        ParameterSpec(
            name="field_embeddings", shape=(local_field_count, unit_dim),
            owner_scope="party", party_id=party_id, trainable=True,
        ),
        ParameterSpec(
            name="value_embeddings", shape=(2, unit_dim),
            owner_scope="party", party_id=party_id, trainable=True,
        ),
        ParameterSpec(
            name="readout_w_key", shape=(compatible_dim, unit_dim),
            owner_scope="party", party_id=party_id, trainable=True,
        ),
        ParameterSpec(
            name="readout_w_value", shape=(compatible_dim, unit_dim),
            owner_scope="party", party_id=party_id, trainable=True,
        ),
    )
