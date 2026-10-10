"""FedGMF shared state, party-local user state, scoring, and local training."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.contracts import DatasetBundle, PartyData
from src.data.negative_sampling import sample_negative_items
from src.federation.fedavg import ClientUpdate

ITEM_EMBEDDINGS_KEY = "item_embeddings"
PREDICTION_WEIGHT_KEY = "prediction_weight"
PREDICTION_BIAS_KEY = "prediction_bias"
FEDGMF_SHARED_KEYS = (
    ITEM_EMBEDDINGS_KEY,
    PREDICTION_WEIGHT_KEY,
    PREDICTION_BIAS_KEY,
)

_SHARED_INIT_STREAM = 0x53485244
_LOCAL_INIT_STREAM = 0x4C4F4341
_SHUFFLE_STREAM = 0x53485546


@dataclass(frozen=True, slots=True, kw_only=True)
class FedGMFLocalState:
    """CPU-resident user IDs and embedding rows owned by one party."""

    user_ids: torch.Tensor
    user_embeddings: torch.Tensor

    def __post_init__(self) -> None:
        if self.user_ids.device.type != "cpu":
            raise ValueError("FedGMFLocalState.user_ids must be on CPU")
        if self.user_ids.dtype != torch.long or self.user_ids.ndim != 1:
            raise ValueError("user_ids must be a one-dimensional torch.long tensor")
        if not self.user_ids.numel():
            raise ValueError("local state must own at least one user")
        if torch.any(self.user_ids < 0).item():
            raise ValueError("user_ids must be non-negative")
        if not torch.equal(self.user_ids, torch.unique(self.user_ids, sorted=True)):
            raise ValueError("user_ids must be sorted and unique")

        if self.user_embeddings.device.type != "cpu":
            raise ValueError("FedGMFLocalState.user_embeddings must be on CPU")
        if (
            self.user_embeddings.ndim != 2
            or not self.user_embeddings.is_floating_point()
        ):
            raise ValueError(
                "user_embeddings must be a two-dimensional floating tensor"
            )
        if self.user_embeddings.shape[0] != self.user_ids.numel():
            raise ValueError("user_embeddings must have one row per user_id")
        if self.user_embeddings.shape[1] <= 0:
            raise ValueError("user_embeddings must have a positive embedding dimension")
        if not torch.isfinite(self.user_embeddings).all().item():
            raise ValueError("user_embeddings must be finite")

    def row_indices(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Map processed user IDs to deterministic party-local row indices."""
        if user_ids.device.type != "cpu":
            user_ids = user_ids.detach().cpu()
        if user_ids.dtype != torch.long or user_ids.ndim != 1:
            raise ValueError("user_ids must be a one-dimensional torch.long tensor")
        if not user_ids.numel():
            return torch.empty(0, dtype=torch.long)

        rows = torch.searchsorted(self.user_ids, user_ids)
        in_bounds = rows < self.user_ids.numel()
        safe_rows = rows.clamp(max=self.user_ids.numel() - 1)
        owned = in_bounds & (self.user_ids[safe_rows] == user_ids)
        if not torch.all(owned).item():
            unknown = torch.unique(user_ids[~owned], sorted=True).tolist()
            raise ValueError(f"local state does not own user IDs: {unknown}")
        return rows


class _FedGMFSharedModule(nn.Module):
    """Trainable working module containing shared parameters only."""

    def __init__(
        self,
        shared_state: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        super().__init__()
        _validate_shared_state(shared_state)
        self.item_embeddings = nn.Parameter(
            shared_state[ITEM_EMBEDDINGS_KEY].detach().to(device).clone()
        )
        self.prediction_weight = nn.Parameter(
            shared_state[PREDICTION_WEIGHT_KEY].detach().to(device).clone()
        )
        self.prediction_bias = nn.Parameter(
            shared_state[PREDICTION_BIAS_KEY].detach().to(device).clone()
        )

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        item_embeddings = F.embedding(item_ids, self.item_embeddings)
        interactions = user_embeddings * item_embeddings
        return F.linear(
            interactions,
            self.prediction_weight,
            self.prediction_bias,
        ).squeeze(-1)

    def export_shared_state(self) -> dict[str, torch.Tensor]:
        """Return detached CPU copies of shared parameters only."""
        return {
            ITEM_EMBEDDINGS_KEY: self.item_embeddings.detach().cpu().clone(),
            PREDICTION_WEIGHT_KEY: self.prediction_weight.detach().cpu().clone(),
            PREDICTION_BIAS_KEY: self.prediction_bias.detach().cpu().clone(),
        }


def initialize_fedgmf_shared_state(
    num_items: int,
    embedding_dim: int,
    *,
    run_seed: int,
) -> dict[str, torch.Tensor]:
    """Create deterministic CPU shared item and prediction parameters."""
    _positive_int("num_items", num_items)
    _positive_int("embedding_dim", embedding_dim)
    _run_seed(run_seed)
    generator = _torch_generator(run_seed, _SHARED_INIT_STREAM)

    item_embeddings = torch.empty(num_items, embedding_dim)
    nn.init.normal_(item_embeddings, generator=generator)
    prediction_weight = torch.empty(1, embedding_dim)
    nn.init.kaiming_uniform_(
        prediction_weight,
        a=sqrt(5),
        generator=generator,
    )
    prediction_bias = torch.empty(1)
    bound = 1 / sqrt(embedding_dim)
    nn.init.uniform_(
        prediction_bias,
        -bound,
        bound,
        generator=generator,
    )
    return {
        ITEM_EMBEDDINGS_KEY: item_embeddings,
        PREDICTION_WEIGHT_KEY: prediction_weight,
        PREDICTION_BIAS_KEY: prediction_bias,
    }


def initialize_fedgmf_local_state(
    user_ids: Sequence[int] | np.ndarray | torch.Tensor,
    embedding_dim: int,
    *,
    run_seed: int,
    party_id: int,
) -> FedGMFLocalState:
    """Create deterministic CPU embeddings for exactly one party's users."""
    _positive_int("embedding_dim", embedding_dim)
    _run_seed(run_seed)
    _nonnegative_int("party_id", party_id)
    owned_user_ids = torch.sort(_id_tensor(user_ids, "user_ids"))[0]
    if not owned_user_ids.numel():
        raise ValueError("user_ids must not be empty")
    if torch.unique(owned_user_ids).numel() != owned_user_ids.numel():
        raise ValueError("user_ids must not contain duplicates")

    generator = _torch_generator(run_seed, _LOCAL_INIT_STREAM, party_id)
    user_embeddings = torch.empty(owned_user_ids.numel(), embedding_dim)
    nn.init.normal_(user_embeddings, generator=generator)
    return FedGMFLocalState(
        user_ids=owned_user_ids,
        user_embeddings=user_embeddings,
    )


def score_fedgmf(
    shared_state: Mapping[str, torch.Tensor],
    local_state: FedGMFLocalState,
    user_ids: Sequence[int] | np.ndarray | torch.Tensor,
    item_ids: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return one unnormalized relevance logit per processed user/item pair."""
    users = _id_tensor(user_ids, "user_ids")
    items = _id_tensor(item_ids, "item_ids")
    if users.ndim != 1 or items.ndim != 1 or users.numel() != items.numel():
        raise ValueError("user_ids and item_ids must be aligned one-dimensional IDs")
    if not users.numel():
        raise ValueError("at least one user/item pair is required")

    num_items, embedding_dim, _ = _validate_shared_state(shared_state)
    _validate_local_dimension(local_state, embedding_dim)
    if torch.any(items >= num_items).item():
        raise ValueError("item_ids contain an out-of-range processed item ID")
    local_rows = local_state.row_indices(users)

    selected_device = torch.device(device)
    module = _FedGMFSharedModule(shared_state, selected_device)
    module.eval()
    with torch.no_grad():
        local_embeddings = local_state.user_embeddings.to(selected_device)
        logits = module(
            local_embeddings[local_rows.to(selected_device)],
            items.to(selected_device),
        )
    return logits.detach().cpu()


def train_local_fedgmf(
    shared_state: Mapping[str, torch.Tensor],
    local_state: FedGMFLocalState,
    bundle: DatasetBundle,
    party: PartyData,
    *,
    round_index: int,
    local_epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    negatives_per_positive: int,
    negative_strategy: str,
    excluded_splits: Sequence[str],
    run_seed: int,
    device: str | torch.device,
) -> tuple[FedGMFLocalState, ClientUpdate]:
    """Train one party and return separate local state and shared upload."""
    _run_seed(run_seed)
    _nonnegative_int("round_index", round_index)
    _positive_int("local_epochs", local_epochs)
    _positive_int("batch_size", batch_size)
    _positive_int("negatives_per_positive", negatives_per_positive)
    _positive_number("learning_rate", learning_rate)
    _nonnegative_number("weight_decay", weight_decay)

    num_items, embedding_dim, _ = _validate_shared_state(shared_state)
    if num_items != len(bundle.item_id_map):
        raise ValueError("shared item state does not match the dataset item count")
    _validate_local_dimension(local_state, embedding_dim)
    positive_ids = _validate_party_training_data(bundle, party, local_state)

    selected_device = torch.device(device)
    shared_module = _FedGMFSharedModule(shared_state, selected_device)
    local_embeddings = nn.Parameter(
        local_state.user_embeddings.detach().to(selected_device).clone()
    )
    optimizer = torch.optim.Adam(
        [*shared_module.parameters(), local_embeddings],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()

    for local_epoch in range(local_epochs):
        negative_seed = _component_seed(
            run_seed,
            round_index,
            party.party_id,
            local_epoch,
        )
        negatives = sample_negative_items(
            bundle,
            positive_ids,
            negatives_per_positive=negatives_per_positive,
            strategy=negative_strategy,
            seed=negative_seed,
            excluded_splits=excluded_splits,
            replace=False,
        )
        example_users, example_items, labels = _labeled_examples(
            bundle,
            positive_ids,
            negatives,
        )
        local_rows = local_state.row_indices(example_users)
        shuffle_seed = _component_seed(
            run_seed,
            round_index,
            party.party_id,
            local_epoch,
            _SHUFFLE_STREAM,
        )
        order = np.random.default_rng(shuffle_seed).permutation(labels.numel())
        order_tensor = torch.from_numpy(order.astype(np.int64, copy=False))

        shared_module.train()
        for start in range(0, labels.numel(), batch_size):
            batch = order_tensor[start : start + batch_size]
            batch_rows = local_rows[batch].to(selected_device)
            batch_items = example_items[batch].to(selected_device)
            batch_labels = labels[batch].to(selected_device)

            optimizer.zero_grad()
            logits = shared_module(local_embeddings[batch_rows], batch_items)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss).item():
                raise ValueError("local FedGMF loss became non-finite")
            loss.backward()
            optimizer.step()

    updated_local_state = FedGMFLocalState(
        user_ids=local_state.user_ids.detach().cpu().clone(),
        user_embeddings=local_embeddings.detach().cpu().clone(),
    )
    client_update = ClientUpdate(
        party_id=party.party_id,
        shared_state=shared_module.export_shared_state(),
        num_positive_interactions=int(positive_ids.size),
    )
    return updated_local_state, client_update


def _validate_shared_state(
    shared_state: Mapping[str, torch.Tensor],
) -> tuple[int, int, torch.dtype]:
    if not isinstance(shared_state, Mapping):
        raise TypeError("shared_state must be a mapping")
    if set(shared_state) != set(FEDGMF_SHARED_KEYS):
        raise ValueError(
            "FedGMF shared_state must contain item embeddings and prediction "
            "weight/bias only"
        )
    for key in FEDGMF_SHARED_KEYS:
        value = shared_state[key]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"FedGMF shared tensor {key!r} must be floating")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"FedGMF shared tensor {key!r} must be finite")

    items = shared_state[ITEM_EMBEDDINGS_KEY]
    weight = shared_state[PREDICTION_WEIGHT_KEY]
    bias = shared_state[PREDICTION_BIAS_KEY]
    if items.ndim != 2 or not items.shape[0] or not items.shape[1]:
        raise ValueError("item_embeddings must have shape (num_items, embedding_dim)")
    num_items, embedding_dim = items.shape
    if weight.shape != (1, embedding_dim) or bias.shape != (1,):
        raise ValueError("prediction tensors do not match the embedding dimension")
    if weight.dtype != items.dtype or bias.dtype != items.dtype:
        raise TypeError("all FedGMF shared tensors must have the same dtype")
    return int(num_items), int(embedding_dim), items.dtype


def _validate_local_dimension(
    local_state: FedGMFLocalState,
    embedding_dim: int,
) -> None:
    if not isinstance(local_state, FedGMFLocalState):
        raise TypeError("local_state must be a FedGMFLocalState")
    if local_state.user_embeddings.shape[1] != embedding_dim:
        raise ValueError("local and shared embedding dimensions do not match")


def _validate_party_training_data(
    bundle: DatasetBundle,
    party: PartyData,
    local_state: FedGMFLocalState,
) -> np.ndarray:
    if not isinstance(party, PartyData):
        raise TypeError("party must be a PartyData")
    if not np.array_equal(local_state.user_ids.numpy(), party.user_ids):
        raise ValueError("local state user IDs must exactly match party.user_ids")
    positive_ids = party.train_interaction_ids
    if not positive_ids.size:
        raise ValueError("party has no positive training interactions")
    if not np.all(np.isin(positive_ids, bundle.splits.train_interaction_ids)):
        raise ValueError("party training IDs must belong to the dataset train split")
    training_users = torch.from_numpy(
        bundle.interactions.user_ids[positive_ids].astype(np.int64, copy=False)
    )
    local_state.row_indices(training_users)
    return positive_ids


def _labeled_examples(
    bundle: DatasetBundle,
    positive_ids: np.ndarray,
    negatives: Mapping[int, tuple[int, ...]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    user_ids: list[int] = []
    item_ids: list[int] = []
    labels: list[float] = []
    for interaction_id in positive_ids.tolist():
        user_id = int(bundle.interactions.user_ids[interaction_id])
        positive_item_id = int(bundle.interactions.item_ids[interaction_id])
        user_ids.append(user_id)
        item_ids.append(positive_item_id)
        labels.append(1.0)
        for negative_item_id in negatives[interaction_id]:
            user_ids.append(user_id)
            item_ids.append(negative_item_id)
            labels.append(0.0)
    return (
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(item_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def _id_tensor(
    values: Sequence[int] | np.ndarray | torch.Tensor,
    name: str,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.dtype != torch.long:
            raise ValueError(f"{name} tensor must use torch.long")
        result = values.detach().cpu()
    else:
        array = np.asarray(values)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must contain one-dimensional integer IDs")
        result = torch.from_numpy(array.astype(np.int64, copy=False))
    if result.ndim != 1 or torch.any(result < 0).item():
        raise ValueError(f"{name} must contain one-dimensional non-negative IDs")
    return result


def _component_seed(*components: int) -> int:
    seed_sequence = np.random.SeedSequence(list(components))
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def _torch_generator(*components: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_component_seed(*components))
    return generator


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _run_seed(run_seed: int) -> None:
    if (
        isinstance(run_seed, bool)
        or not isinstance(run_seed, int)
        or not 0 <= run_seed < 2**32
    ):
        raise ValueError("run_seed must be an integer in [0, 2**32)")


def _positive_number(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")


def _nonnegative_number(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
