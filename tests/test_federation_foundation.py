"""Bounded synthetic qualification; no new federation or optimization policy."""

from copy import deepcopy
from dataclasses import asdict, replace
from math import ceil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.contracts import DatasetBundle, InteractionTable
from src.data.partition import build_hfl_partition
from src.data.split import leave_one_out_split
from src.federation.hfl import run_synchronous_hfl_round
from src.federation.runner import fit_fedgmf_hfl
from src.models.fedgmf import (
    FEDGMF_SHARED_KEYS,
    initialize_fedgmf_local_state,
    initialize_fedgmf_shared_state,
    score_fedgmf,
    train_local_fedgmf,
)
from src.utils.config import load_yaml, validate_phase3_experiment_config


def _bundle(histories: tuple[tuple[int, ...], ...]) -> DatasetBundle:
    """Small disjoint histories in first-appearance processed-ID order."""
    items = np.array([item for history in histories for item in history], dtype=np.int64)
    table = InteractionTable(
        interaction_ids=np.arange(items.size, dtype=np.int64),
        user_ids=np.repeat(np.arange(len(histories), dtype=np.int64),
                           [len(history) for history in histories]),
        item_ids=items,
        timestamps=np.arange(items.size, dtype=np.int64),
        values=np.ones(items.size, dtype=np.float32),
    )
    return DatasetBundle(
        name="synthetic_foundation", schema_version="1.0",
        interactions=table, splits=leave_one_out_split(table),
        user_id_map={f"user-{i}": i for i in range(len(histories))},
        item_id_map={f"item-{i}": i for i in range(items.size)},
        metadata={"fixture": "in-memory, disjoint histories; no real source"},
    )


def _initial_states(bundle, parties):
    shared = initialize_fedgmf_shared_state(
        len(bundle.item_id_map), 4, run_seed=2026,
    )
    local = {
        party.party_id: initialize_fedgmf_local_state(
            party.user_ids, 4, run_seed=2026, party_id=party.party_id,
        ) for party in parties
    }
    return shared, local


@pytest.mark.parametrize("fraction", [1.0, 0.5], ids=["full", "partial"])
def test_actual_round_weights_positive_counts_not_optimizer_work(
    monkeypatch: pytest.MonkeyPatch, fraction: float,
) -> None:
    bundle = _bundle(((0, 1, 2), (3, 4, 5, 6, 7), tuple(range(8, 15))))
    parties = tuple(
        replace(party, party_id=party_id)
        for party, party_id in zip(
            build_hfl_partition(bundle), (41, 7, 103), strict=True,
        )
    )
    shared, local = _initial_states(bundle, parties)
    before = deepcopy((bundle, parties, shared, local))
    # Deliberately different workloads make incorrect weighting distinguishable.
    # party: (positive count, local epochs, negatives per positive, batch size)
    workloads = {41: (1, 1, 1, 2), 7: (3, 2, 2, 4), 103: (5, 3, 4, 4)}
    uploads, step_counts, example_counts = {}, {}, {}
    active_party = []
    real_step = torch.optim.Adam.step
    real_loss = torch.nn.BCEWithLogitsLoss.forward

    def observed_step(optimizer, *args, **kwargs):
        step_counts[active_party[-1]] += 1
        return real_step(optimizer, *args, **kwargs)

    def observed_loss(criterion, logits, labels):
        example_counts[active_party[-1]] += labels.numel()
        return real_loss(criterion, logits, labels)

    monkeypatch.setattr(torch.optim.Adam, "step", observed_step)
    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", observed_loss)

    def update(round_shared, party_local, party):
        for key in FEDGMF_SHARED_KEYS:
            assert torch.equal(round_shared[key], before[2][key])
        count, epochs, negatives, batch_size = workloads[party.party_id]
        assert party.train_interaction_ids.size == count
        step_counts[party.party_id] = example_counts[party.party_id] = 0
        active_party.append(party.party_id)
        try:
            next_local, upload = train_local_fedgmf(
                round_shared, party_local, bundle, party,
                round_index=0, local_epochs=epochs, learning_rate=0.03,
                batch_size=batch_size, weight_decay=0.0,
                negatives_per_positive=negatives, negative_strategy="uniform",
                excluded_splits=("train", "valid", "test"),
                run_seed=2026, device="cpu",
            )
        finally:
            active_party.pop()
        assert upload.num_positive_interactions == count
        assert tuple(upload.shared_state) == FEDGMF_SHARED_KEYS
        uploads[party.party_id] = upload
        return next_local, upload

    result = run_synchronous_hfl_round(
        shared, tuple(reversed(parties)), local,
        client_fraction=fraction, run_seed=2026, round_index=0,
        local_update=update,
    )
    selected = result.selected_party_ids
    assert len(selected) == ceil(fraction * 3)
    assert selected == tuple(sorted(selected)) == tuple(uploads)
    weights = np.array([workloads[pid][0] for pid in selected], dtype=np.float64)
    assert result.total_positive_interactions == int(weights.sum())
    for pid in selected:
        count, epochs, negatives, batch_size = workloads[pid]
        assert example_counts[pid] == count * (1 + negatives) * epochs
        assert step_counts[pid] == ceil(count * (1 + negatives) / batch_size) * epochs

    # Independent NumPy arithmetic on real uploads, not aggregate_fedavg again.
    for key in FEDGMF_SHARED_KEYS:
        values = np.stack([
            uploads[pid].shared_state[key].numpy().astype(np.float64)
            for pid in selected
        ])
        broadcast_weights = weights.reshape((-1,) + (1,) * (values.ndim - 1))
        expected = (values * broadcast_weights).sum(axis=0) / weights.sum()
        torch.testing.assert_close(
            result.next_shared_state[key],
            torch.from_numpy(expected).to(shared[key].dtype), rtol=1e-6, atol=1e-8,
        )
        assert torch.isfinite(result.next_shared_state[key]).all()
        assert torch.equal(shared[key], before[2][key])

    # Check sensitivity: these plausible wrong normalizations do NOT coincide.
    biases = np.array([uploads[pid].shared_state["prediction_bias"].item()
                       for pid in selected])
    correct = np.dot(weights, biases) / weights.sum()
    wrong_weights = (
        np.array([step_counts[pid] for pid in selected]),
        np.array([workloads[pid][0] * workloads[pid][2] for pid in selected]),
        np.array([workloads[pid][0] * workloads[pid][1] for pid in selected]),
        np.array([example_counts[pid] for pid in selected]),
    )
    for wrong in wrong_weights:
        assert not np.isclose(correct, np.dot(wrong, biases) / wrong.sum(),
                              rtol=1e-6, atol=1e-8)
    if fraction < 1:
        all_party_denominator = sum(values[0] for values in workloads.values())
        assert not np.isclose(correct, np.dot(weights, biases) / all_party_denominator,
                              rtol=1e-6, atol=1e-8)

    for party in parties:
        pid = party.party_id
        assert torch.equal(local[pid].user_embeddings, before[3][pid].user_embeddings)
        assert torch.equal(result.local_states[pid].user_ids, local[pid].user_ids)
        if pid in selected:
            assert not torch.equal(result.local_states[pid].user_embeddings,
                                   local[pid].user_embeddings)
        else:
            assert result.local_states[pid] is local[pid]
    np.testing.assert_equal(asdict(bundle), asdict(before[0]))
    for party, old_party in zip(parties, before[1], strict=True):
        np.testing.assert_equal(asdict(party), asdict(old_party))
    print({"fraction": fraction, "selected": selected,
           "positive_counts": weights.tolist(), "optimizer_steps": step_counts,
           "labeled_examples": example_counts, "weighted_bias": correct})


def _diagnostic(shared, local):
    # The same four training positives and their entire four-item negative pools.
    # Held-out positives are never diagnostic positives or negatives for their user.
    users = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    items = np.array([0, 4, 5, 6, 7, 1, 4, 5, 6, 7,
                      4, 0, 1, 2, 3, 5, 0, 1, 2, 3], dtype=np.int64)
    labels = np.array([1, 0, 0, 0, 0] * 4, dtype=np.float64)
    logits = score_fedgmf(shared, local, users, items).numpy().astype(np.float64)
    loss = float(np.mean(np.logaddexp(0, logits) - labels * logits))
    margin = float(logits[labels == 1].mean() - logits[labels == 0].mean())
    return logits, loss, margin


def test_fixed_learnable_fit_with_finite_gradients_and_fresh_adam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Frozen before execution: 2 users, 8 items; seed 2026, dim 4; 12 rounds
    # x 5 epochs x one full batch = 60 Adam steps. No tuning or sampling stub.
    bundle = _bundle(((0, 1, 2, 3), (4, 5, 6, 7)))
    party = replace(build_hfl_partition(
        bundle, mode="grouped_round_robin", num_parties=1, seed=11,
    )[0], party_id=73)
    config = load_yaml(Path(__file__).parents[1] / "configs/experiments/ml1m_fedgmf_fedavg.yaml")
    config["federation"].update(partition="grouped_round_robin", num_parties=1,
                                rounds=12, local_epochs=5, client_fraction=1.0)
    config["model"]["embedding_dim"] = 4
    config["optimization"].update(learning_rate=0.05, batch_size=20,
                                 train_negatives_per_positive=4)
    config["evaluation"].update(k=[1, 3], negatives_per_positive=4)
    config["runtime"].update(seed=2026, device="cpu")
    validate_phase3_experiment_config(config)
    before = deepcopy((bundle, party, config))
    shared, local = _initial_states(bundle, (party,))
    no_update_shared, no_update_local = deepcopy((shared, local[73]))
    initial_logits, initial_loss, initial_margin = _diagnostic(shared, local[73])

    optimizers, steps, losses = [], [], []
    real_step = torch.optim.Adam.step
    real_loss = torch.nn.BCEWithLogitsLoss.forward

    def observed_loss(criterion, logits, labels):
        loss = real_loss(criterion, logits, labels)
        assert logits.numel() == labels.numel() == 20
        assert int(labels.sum().item()) == 4
        assert torch.isfinite(loss).item()
        losses.append(float(loss.detach()))
        return loss

    def observed_step(optimizer, *args, **kwargs):
        if not any(optimizer is existing for existing in optimizers):
            assert not optimizer.state, "Adam state must start fresh per local update"
            optimizers.append(optimizer)  # Bounded: retain 12 tiny diagnostic objects.
            steps.append(0)
        assert optimizer is optimizers[-1]
        steps[-1] += 1
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(parameters) == 4  # Three shared tensors plus local user rows.
        for parameter in parameters:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all().item()
        return real_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", observed_loss)
    monkeypatch.setattr(torch.optim.Adam, "step", observed_step)
    result = fit_fedgmf_hfl(bundle, (party,), config)

    assert len(optimizers) == 12
    assert steps == [5] * 12
    assert len(losses) == 60
    assert len(result.round_summaries) == 12
    assert all(summary.selected_party_ids == (73,)
               and summary.total_positive_interactions == 4
               and summary.validation_metrics["num_targets"] == 2
               for summary in result.round_summaries)
    assert result.validation_candidates.split == "valid"
    for candidate in result.validation_candidates.candidates:
        assert len(candidate.negative_item_ids) == 4  # Enough catalogue capacity.
    _, final_loss, final_margin = _diagnostic(
        result.final_shared_state, result.final_local_states[73],
    )
    reference_logits, reference_loss, reference_margin = _diagnostic(
        no_update_shared, no_update_local,
    )
    np.testing.assert_array_equal(reference_logits, initial_logits)
    assert reference_loss == initial_loss
    assert reference_margin == initial_margin
    assert final_loss < initial_loss - 0.15
    assert final_margin > initial_margin + 0.5
    for key in FEDGMF_SHARED_KEYS:
        assert torch.equal(shared[key], no_update_shared[key])
        assert not torch.equal(result.final_shared_state[key], shared[key])
    assert torch.equal(local[73].user_embeddings, no_update_local.user_embeddings)
    assert torch.equal(result.final_local_states[73].user_ids, local[73].user_ids)
    assert torch.all(torch.any(result.final_local_states[73].user_embeddings
                              != local[73].user_embeddings, dim=1))
    np.testing.assert_equal(asdict(bundle), asdict(before[0]))
    np.testing.assert_equal(asdict(party), asdict(before[1]))
    assert config == before[2]
    print({"initial_loss": initial_loss, "final_loss": final_loss,
           "initial_margin": initial_margin, "final_margin": final_margin,
           "no_update_loss": reference_loss, "no_update_margin": reference_margin,
           "adam_instances": len(optimizers), "adam_steps": sum(steps),
           "finite_losses": len(losses), "finite_gradient_tensors": 4 * sum(steps)})
