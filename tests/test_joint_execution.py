"""Unit2 bounded CPU integration; an independently connected six-step oracle.

The fixture, counterexamples, tolerances and caps below are frozen before the
first authorized pytest invocation. No executor helper constructs expectations.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math

import numpy as np
import pytest
import torch

from src.data.contracts import (
    DatasetBundle, FeatureAccess, FeaturePartition, FeatureView,
    InteractionSplits, InteractionTable, PartyData,
)
from src.data.feature_runtime import (
    check_feature_runtime_binding, close_feature_runtime,
    prepare_feature_runtime, project_party_features,
)
from src.data.local_field_maps import prepare_local_field_map
from src.data.model_inputs import (
    close_binary_input_adapter, deliver_binary_field_units,
    prepare_binary_input_adapter,
)
from src.data.training_trace import build_training_trace, check_training_trace_context
from src.evaluation.ranking import (
    EvaluationCandidate, EvaluationCandidateSet, build_evaluation_candidates,
)
from src.evaluation.terminal import evaluate_terminal_once, observe_terminal_evaluation
from src.federation import joint_execution as joint
from src.federation.optimizer_owners import OptimizerStepError, snapshot_optimizer_owners
from src.federation.round_drain import prepare_round_drain_plan
from src.methods.doi4.composition import (
    encode_readout_binary_units, group_loss_pri_none_from_fused,
)
from src.methods.doi4.fusion import center_fusion
from src.methods.doi4.parameters import doi4_parameter_specs


FROZEN_FIXTURE = {
    "fixture_id": "UNIT2-JOINT-2x5-v1",
    "dimensions": {"I": 12, "d": 3, "h": 3, "K": 2},
    "parties": [7, 23, 41], "users": {"7": [0, 2], "23": [1], "41": []},
    "J": [2, 2, 0], "r": [2, 3, 2], "rounds": [0, 2, 5],
    "interactions": [[0, 0], [2, 1], [1, 0], [1, 2], [1, 3], [1, 4],
                     [1, 5], [0, 6], [1, 7], [2, 8], [0, 9], [1, 10], [2, 11]],
    "train": list(range(7)), "valid": [7, 8, 9], "test": [10, 11, 12],
    "run_seed": 20261028, "epoch": 0, "q": 1, "batch_size": 2,
    "excluded_splits": ["train", "valid", "test"], "replace": False,
    "parameter_order": "center once then party:7,party:23,party:41; spec name order",
    "values": "(((7*j+3*s)%23)-11)/64+1/32; s begins at 1; score_bias=1/8",
    "binary_values": "(item+field_index+party_index)%2",
    "adam": {"lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
    "validation": {"negatives": 3, "k": [1, 3]},
    "tolerance": {"float64": {"rtol": 1e-10, "atol": 1e-12},
                  "float32": {"rtol": 2e-5, "atol": 2e-6}},
    "expected": {"applies": 6, "validations": 3, "generation": [0, 3, 3, 6],
                 "groups": 7, "steps_per_nonempty_round": 3,
                 "user_clocks": {"7": 2, "23": 6, "41": 0}},
    "counterexamples": [
        "all_false", "zero_H", "empty_pools", "bad_pool_key", "bad_receiver_key",
        "bad_online", "bad_user", "bad_domain", "bad_grant", "closed_runtime",
        "closed_adapter", "caller_mutation", "observation_mutation", "missing",
        "duplicate", "stale", "unexpected", "wrong_geometry", "nonfinite_gradient",
        "physical_apply_failure", "bad_apply_receipt", "monitor_failure", "zero_rounds",
    ],
    "caps": {"parties": 8, "items": 128, "users": 32, "rounds": 4, "steps": 32,
             "groups": 256, "label_rows": 2048, "fields": 16, "width": 16,
             "K": 4, "validation_targets": 16, "candidate_items": 32},
}
PARTIES = (7, 23, 41)
ROUTING = {7: (0, 2), 23: (1,), 41: ()}
CONTEXT = "d" * 64


def _ids(values):
    return np.array(values, dtype=np.int64)


def _values(dtype):
    """Literal recipe in task allocation order, not optimizer sorted order."""
    result = {"center": {}}
    serial = 0
    for party, fields, width in zip(PARTIES, (2, 2, 0), (2, 3, 2)):
        result[f"party:{party}"] = {}
        specs = doi4_parameter_specs(
            party_id=party, public_item_count=12, local_user_count=len(ROUTING[party]),
            local_field_count=fields, item_dim=3, unit_dim=width,
            compatible_dim=3, slot_count=2,
        )
        for spec in specs:
            if spec.owner_scope == "center" and party != 7:
                continue
            serial += 1
            owner = "center" if spec.owner_scope == "center" else f"party:{party}"
            count = math.prod(spec.shape)
            flat = [(((7 * j + 3 * serial) % 23) - 11) / 64 + 1 / 32
                    for j in range(count)]
            tensor = torch.tensor(flat, dtype=dtype).reshape(spec.shape)
            if spec.name == "score_bias":
                tensor.fill_(1 / 8)
            result[owner][spec.name] = tensor
    return result


def _fixture(dtype=torch.float64, *, mode="normal", denied=False):
    """Real canonical views, access, borrowed runtime, adapters and traces."""
    interactions = FROZEN_FIXTURE["interactions"]
    views = {}
    accesses = {}
    for index, party in enumerate(PARTIES):
        if party == 41:
            accesses[party] = ()
            continue
        name = f"opaque-view-{party}"
        views[name] = FeatureView(
            name=name, entity_kind="item", modality="tabular", entity_ids=_ids(range(12)),
            values=np.array([[(item + field + index) % 2 for field in range(2)]
                             for item in range(12)], dtype=np.int64),
            feature_names=("opaque-a", "opaque-b"),
            missing_mask=np.full((12, 2), mode == "all_false", dtype=bool), metadata={},
        )
        rows = tuple(range(1, 12)) if denied and party == 7 else tuple(range(12))
        accesses[party] = (FeatureAccess(view_name=name, entity_ids=_ids(rows),
                                         feature_names=None),)
    bundle = DatasetBundle(
        name="UNIT2-JOINT-2x5-v1", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=_ids(range(13)), user_ids=_ids([row[0] for row in interactions]),
            item_ids=_ids([row[1] for row in interactions]), timestamps=_ids(range(13)),
            values=np.ones(13, dtype=np.float32),
        ),
        splits=InteractionSplits(train_interaction_ids=_ids(range(7)),
                                 valid_interaction_ids=_ids((7, 8, 9)),
                                 test_interaction_ids=_ids((10, 11, 12))),
        user_id_map={f"u{u}": u for u in range(3)},
        item_id_map={f"i{i}": i for i in range(12)}, metadata={}, feature_views=views,
    )
    parties = []
    for party in PARTIES:
        users = ROUTING[party]
        own = [i for i, (user, _) in enumerate(interactions) if user in users]
        parties.append(PartyData(
            party_id=party, train_interaction_ids=_ids([i for i in own if i < 7]),
            valid_interaction_ids=_ids([i for i in own if 7 <= i < 10]),
            test_interaction_ids=_ids([i for i in own if i >= 10]), user_ids=_ids(users),
            item_ids=_ids(sorted({interactions[i][1] for i in own})),
            feature_view_names=tuple(a.view_name for a in accesses[party]),
            graph_view_names=(), owns_labels=bool(users), metadata={},
        ))
    partition = FeaturePartition(name="explicit", party_access=accesses, metadata={})
    runtime = prepare_feature_runtime(
        bundle, partition, parties, source_id="unit2-source-v1",
        split_id="unit2-split-v1", recipe_id="unit2-fields-v1",
    )
    adapters, bindings, maps = {}, {}, {}
    for party in PARTIES:
        projected = project_party_features(runtime, party_id=party, binding_id=runtime.binding_id)
        assignments = tuple((name, field) for name, view in projected.items()
                            for field in view.feature_names)
        local_map = prepare_local_field_map(party_id=party, schema_epoch=0,
                                            field_assignments=assignments)
        permissions = {entry.view_name: tuple(entry.entity_ids.tolist())
                       for entry in accesses[party]}
        adapter = prepare_binary_input_adapter(
            projected, party_id=party, public_item_count=12, runtime_binding_id=runtime.binding_id,
            local_map=local_map, authorized_rows_by_view=permissions,
        )
        adapters[party], maps[party] = adapter, local_map
        bindings[party] = (runtime.binding_id, local_map.map_id, 0, adapter.binding_id)
    plans, traces, pools = [], [], {}
    for round_id in (0, 2, 5):
        selected = () if round_id == 2 else PARTIES
        trace = build_training_trace(
            bundle, parties, selected_party_ids=selected, round_index=round_id,
            local_epoch=0, run_seed=20261028, negatives_per_positive=1,
            excluded_splits=("train", "valid", "test"), replace=False,
        )
        check_training_trace_context(trace, bundle, parties)
        plan = prepare_round_drain_plan(
            bundle, parties, selected_party_ids=selected,
            ordered_positive_ids=dict(trace.ordered_positive_ids),
            negative_items=dict(trace.negative_items), round_index=round_id,
            batch_unit="positive_group", positive_groups_per_batch=2,
            negatives_per_positive=1, negative_replacement=False, trace_id=trace.trace_id,
        )
        plans.append(plan)
        traces.append(trace)
        for step in range(plan.step_count):
            pools[(round_id, step)] = {
                party: (() if mode == "empty_pools" else PARTIES)
                for party in plan.loss_party_ids if step < plan.batch_counts[party]
            }
    initial = _values(dtype)
    if mode == "zero_H":
        for party in PARTIES:
            initial[f"party:{party}"]["readout_w_value"].zero_()
    validation = build_evaluation_candidates(
        bundle, (7, 8, 9), split="valid", negatives_per_positive=3, base_seed=20261028,
    )
    arguments = dict(
        rounds=tuple(plans), initial_values=initial, user_ids_by_party=dict(ROUTING),
        adapters=adapters, feature_runtimes={p: runtime for p in PARTIES}, input_bindings=bindings,
        allowed_sources_by_receiver={p: PARTIES for p in PARTIES}, step_source_pools=pools,
        inference_source_pools={p: (() if mode == "empty_pools" else PARTIES) for p in PARTIES},
        inference_online_party_ids=PARTIES, context_id=CONTEXT,
        initialization_id="UNIT2-JOINT-2x5-v1-initial", profile_id="UNIT2-JOINT-2x5-v1-profile",
        learning_rate=0.01, betas=(0.9, 0.999), epsilon=1e-8,
        validation_candidates=validation, k_values=(1, 3),
    )
    # These explicit bounds are part of the test fixture, not measured scale.
    assert len(parties) == 3 and len(bundle.item_id_map) == 12 and len(bundle.user_id_map) == 3
    assert sum(plan.step_count for plan in plans) == 6
    assert sum(plan.round_positive_count for plan in plans) == 14
    return dict(args=arguments, bundle=bundle, parties=parties, runtime=runtime,
                adapters=adapters, maps=maps, traces=traces, mode=mode)


def _clone(groups):
    return {owner: {name: None if value is None else value.detach().clone()
                    for name, value in group.items()} for owner, group in groups.items()}


def _batches(plan, step):
    """Independent descriptor expansion, without new executor or packer helpers."""
    result = {}
    for party in plan.selected_party_ids:
        begin = step * plan.positive_groups_per_batch
        positives = plan.positive_ids_by_party[party][begin:begin + plan.positive_groups_per_batch]
        if not positives:
            continue
        users = plan.user_ids_by_party[party][begin:begin + len(positives)]
        positive_items = plan.positive_item_ids_by_party[party][begin:begin + len(positives)]
        row_users, row_items, labels, offsets = [], [], [], [0]
        for interaction, user, positive in zip(positives, users, positive_items):
            items = (positive, *plan.negative_items[interaction])
            row_users.extend([user] * len(items))
            row_items.extend(items)
            labels.extend([1.0] + [0.0] * (len(items) - 1))
            offsets.append(len(row_items))
        result[party] = (tuple(row_users), tuple(row_items), labels, offsets, len(positives))
    return result


def _oracle(fixture):
    """Fully connected accepted math, explicit routing/scatters, persistent Adam.

    No new fit, contribution, receipt, detached-boundary or state helper is used.
    Both item paths gather from the SAME authoritative table; all H and z stay
    connected to their source parameters until the one complete scalar backward.
    """
    arguments = fixture["args"]
    parameters = {o: {n: torch.nn.Parameter(t.detach().clone()) for n, t in group.items()}
                  for o, group in arguments["initial_values"].items()}
    optimizers = {o: torch.optim.Adam(
        list(group.values()), lr=0.01, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False, fused=False,
    ) for o, group in parameters.items()}
    steps = []
    dtype = parameters["center"]["item_embeddings"].dtype
    for plan in arguments["rounds"]:
        for step in range(plan.step_count):
            batches = _batches(plan, step)
            pools = arguments["step_source_pools"][(plan.round_index, step)]
            source_ids = sorted({source for pool in pools.values() for source in pool})
            source_items = {source: tuple(sorted({item for receiver, batch in batches.items()
                                                if source in pools[receiver] for item in batch[1]}))
                            for source in source_ids}
            publications = {}
            center = parameters["center"]
            for source in source_ids:
                party_values = parameters[f"party:{source}"]
                fields = party_values["field_embeddings"].shape[0]
                rows = source_items[source]
                codes = torch.tensor(
                    [[(item + field + PARTIES.index(source)) % 2 for field in range(fields)]
                     for item in rows], dtype=torch.int64,
                ).reshape(len(rows), fields)
                ids = torch.arange(fields, dtype=torch.int64).expand(len(rows), fields).clone()
                mask = torch.full((len(rows), fields), fixture["mode"] != "all_false", dtype=torch.bool)
                if fixture["mode"] == "all_false":
                    ids.fill_(-1)
                    codes.zero_()
                H, available, _ = encode_readout_binary_units(
                    ids, codes, mask, field_embeddings=party_values["field_embeddings"],
                    value_embeddings=party_values["value_embeddings"], queries=center["queries"],
                    w_key=party_values["readout_w_key"], w_value=party_values["readout_w_value"],
                )
                publications[source] = (H, available)
            losses, path_rows = [], []
            for receiver, (users, items, labels, offsets, count) in batches.items():
                token_parts, masks = [], []
                for source in pools[receiver]:
                    H, available = publications[source]
                    inverse = torch.tensor([source_items[source].index(item) for item in items])
                    selected_mask = available[inverse]
                    token_parts.append(torch.where(selected_mask[:, None, None], H[inverse], 0))
                    masks.append(selected_mask[:, None].expand(len(items), 2))
                tokens = (torch.cat(token_parts, dim=1) if token_parts
                          else torch.empty((len(items), 0, 3), dtype=dtype))
                valid = (torch.cat(masks, dim=1) if masks
                         else torch.empty((len(items), 0), dtype=torch.bool))
                indices = torch.tensor(items, dtype=torch.int64)
                direct = center["item_embeddings"][indices]
                query = center["item_embeddings"][indices]
                fused, available, _ = center_fusion(
                    tokens, valid, query, center["fusion_w_query"], center["fusion_w_key"],
                    center["fusion_w_value"], center["fusion_w_output"],
                )
                user_rows = torch.tensor([ROUTING[receiver].index(user) for user in users])
                loss, _ = group_loss_pri_none_from_fused(
                    fused, available, torch.tensor(labels, dtype=dtype), torch.tensor(offsets),
                    user_embeddings=parameters[f"party:{receiver}"]["user_embeddings"][user_rows],
                    item_embeddings=direct, w_fused=center["w_fused"],
                    score_weight=center["score_weight"], score_bias=center["score_bias"],
                )
                losses.append(loss * (count / plan.round_positive_count))
                path_rows.append((indices, direct, query))
            loss = sum(losses)
            item_contributions = []
            for indices, direct, query in path_rows:
                gradients = torch.autograd.grad(loss, (direct, query), retain_graph=True)
                for gradient in gradients:
                    dense = torch.zeros_like(center["item_embeddings"])
                    dense.index_add_(0, indices, gradient)
                    item_contributions.append(dense)
            shared_contributions = []
            for source in source_ids:
                H = publications[source][0]
                derivative, = torch.autograd.grad(loss, (H,), retain_graph=True)
                shared, = torch.autograd.grad(H, (center["queries"],), derivative, retain_graph=True)
                shared_contributions.append(shared)
            loss.backward()
            gradients = {o: {n: None if p.grad is None else p.grad.detach().clone()
                             for n, p in group.items()} for o, group in parameters.items()}
            # Independently verify the decomposition before comparing production.
            torch.testing.assert_close(sum(item_contributions), gradients["center"]["item_embeddings"],
                                       **FROZEN_FIXTURE["tolerance"][str(dtype).split(".")[-1]])
            if shared_contributions:
                torch.testing.assert_close(sum(shared_contributions), gradients["center"]["queries"],
                                           **FROZEN_FIXTURE["tolerance"][str(dtype).split(".")[-1]])
            else:
                assert gradients["center"]["queries"] is None
            for owner, optimizer in optimizers.items():
                if any(value is not None for value in gradients[owner].values()):
                    optimizer.step()
            clocks = {o: {n: int(optimizers[o].state[p]["step"].item())
                           if p in optimizers[o].state else 0 for n, p in group.items()}
                      for o, group in parameters.items()}
            moments = {
                moment: {o: {n: optimizers[o].state[p][moment].detach().clone()
                             if p in optimizers[o].state else None for n, p in group.items()}
                         for o, group in parameters.items()}
                for moment in ("exp_avg", "exp_avg_sq")
            }
            steps.append(dict(gradients=gradients, values=_clone(parameters), clocks=clocks,
                              item_paths=item_contributions, shared_paths=shared_contributions,
                              moments=moments, receivers=tuple(batches), sources=tuple(source_ids),
                              receiver_items={party: batch[1] for party, batch in batches.items()}))
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
    return steps, _clone(parameters)


def _assert_groups(actual, expected, tolerance):
    assert set(actual) == set(expected)
    maximum = 0.0
    for owner, group in expected.items():
        assert set(actual[owner]) == set(group)
        for name, wanted in group.items():
            got = actual[owner][name]
            assert (got is None) == (wanted is None), (owner, name)
            if wanted is not None:
                assert got.shape == wanted.shape and got.dtype == wanted.dtype
                torch.testing.assert_close(got, wanted, **tolerance)
                if got.numel():
                    maximum = max(maximum, float((got - wanted).abs().max()))
    return maximum


def _capture(monkeypatch):
    observed = dict(steps=[], handles=[], monitors=[], item_paths=[], shared_paths=[], deliveries=[],
                    encodes=[], source_vjps=[], source_graphs={}, monitoring=False)
    prepare, apply = joint.prepare_optimizer_owners, joint.apply_optimizer_gradients
    accumulate, evaluate = joint._accumulate_gradients, joint.evaluate_candidates
    delivery = joint.deliver_binary_field_units
    encode, autograd_grad = joint.encode_readout_binary_units, torch.autograd.grad

    def prepare_spy(parameters, **kwargs):
        handle = prepare(parameters, **kwargs)
        observed["handles"].append(handle)
        observed["parameter_objects"] = {o: dict(group) for o, group in parameters.items()}
        return handle

    def accumulate_spy(groups, owner, name, gradient):
        if owner == "center" and name in ("item_embeddings", "queries"):
            destination = "item_paths" if name == "item_embeddings" else "shared_paths"
            observed[destination].append(None if gradient is None else gradient.detach().clone())
        return accumulate(groups, owner, name, gradient)

    def apply_spy(handle, gradients, **kwargs):
        assert handle is observed["handles"][0]
        before = _clone(gradients)
        receipt = apply(handle, gradients, **kwargs)
        observed["steps"].append(dict(gradients=before, receipt=receipt,
                                      snapshot=snapshot_optimizer_owners(handle),
                                      item_paths=observed["item_paths"],
                                      shared_paths=observed["shared_paths"],
                                      encodes=observed["encodes"], source_vjps=observed["source_vjps"]))
        observed["item_paths"], observed["shared_paths"] = [], []
        observed["encodes"], observed["source_vjps"], observed["source_graphs"] = [], [], {}
        return receipt

    def evaluation_spy(candidates, scorer, *, k_values):
        observed["monitors"].append((candidates.split, tuple(c.target_interaction_id
                                                          for c in candidates.candidates), tuple(k_values)))
        observed["monitoring"] = True
        try:
            return evaluate(candidates, scorer, k_values=k_values)
        finally:
            observed["monitoring"] = False

    def encode_spy(*args, **kwargs):
        result = encode(*args, **kwargs)
        if not observed["monitoring"] and observed["handles"][0].status == "active":
            owner = next(o for o, group in observed["parameter_objects"].items()
                         if group.get("field_embeddings") is kwargs["field_embeddings"])
            party = int(owner.split(":")[1])
            observed["encodes"].append(party)
            observed["source_graphs"][id(result[0])] = party
        return result

    def gradient_spy(*args, **kwargs):
        outputs = args[0] if args else kwargs["outputs"]
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if len(outputs) == 1 and id(outputs[0]) in observed["source_graphs"]:
            observed["source_vjps"].append(observed["source_graphs"][id(outputs[0])])
        return autograd_grad(*args, **kwargs)

    def delivery_spy(adapter, **kwargs):
        result = delivery(adapter, **kwargs)
        observed["deliveries"].append((adapter, tuple(kwargs["entity_ids"]),
                                       result["local_field_ids"].shape))
        return result

    monkeypatch.setattr(joint, "prepare_optimizer_owners", prepare_spy)
    monkeypatch.setattr(joint, "apply_optimizer_gradients", apply_spy)
    monkeypatch.setattr(joint, "_accumulate_gradients", accumulate_spy)
    monkeypatch.setattr(joint, "evaluate_candidates", evaluation_spy)
    monkeypatch.setattr(joint, "deliver_binary_field_units", delivery_spy)
    monkeypatch.setattr(joint, "encode_readout_binary_units", encode_spy)
    monkeypatch.setattr(torch.autograd, "grad", gradient_spy)
    return observed


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_six_joint_steps_match_connected_gradients_and_persistent_adam(dtype, monkeypatch, record_property):
    fixture = _fixture(dtype)
    expected_steps, final = _oracle(fixture)
    observed = _capture(monkeypatch)
    original = _clone(fixture["args"]["initial_values"])
    fit = joint.fit_doi4_pri_none(**fixture["args"])
    tolerance = FROZEN_FIXTURE["tolerance"][str(dtype).split(".")[-1]]
    maximum = dict(gradient=0.0, parameters=0.0, item_paths=0.0, shared_paths=0.0, moments=0.0)
    assert len(observed["steps"]) == len(expected_steps) == 6
    assert len(observed["handles"]) == 1 and observed["handles"][0].status == "closed"
    assert observed["monitors"] == [("valid", (7, 8, 9), (1, 3))] * 3
    for index, (actual, expected) in enumerate(zip(observed["steps"], expected_steps)):
        maximum["gradient"] = max(maximum["gradient"], _assert_groups(
            actual["gradients"], expected["gradients"], tolerance))
        maximum["parameters"] = max(maximum["parameters"], _assert_groups(
            actual["snapshot"].parameters, expected["values"], tolerance))
        assert actual["snapshot"].parameter_steps == expected["clocks"]
        assert actual["receipt"].generation_before == index
        assert actual["receipt"].generation_after == index + 1
        assert tuple(actual["encodes"]) == PARTIES
        assert tuple(actual["source_vjps"]) == PARTIES
        for path in ("item_paths", "shared_paths"):
            assert len(actual[path]) == len(expected[path])
            for got, wanted in zip(actual[path], expected[path]):
                assert got is not None
                torch.testing.assert_close(got, wanted, **tolerance)
                maximum[path] = max(maximum[path], float((got - wanted).abs().max()))
        for moment in ("exp_avg", "exp_avg_sq"):
            maximum["moments"] = max(maximum["moments"], _assert_groups(
                getattr(actual["snapshot"], moment), expected["moments"][moment], tolerance))
        assert actual["gradients"]["party:41"]["user_embeddings"] is None
        assert actual["gradients"]["party:41"]["field_embeddings"] is not None
        if index % 3:
            assert actual["gradients"]["party:7"]["user_embeddings"] is None
    _assert_groups(fit.observe_parameters(), final, tolerance)
    _assert_groups(fixture["args"]["initial_values"], original, {"atol": 0, "rtol": 0})
    assert tuple(r.round_index for r in fit.round_receipts) == (0, 2, 5)
    assert tuple((r.generation_before, r.generation_after) for r in fit.round_receipts) == ((0, 3), (3, 3), (3, 6))
    assert tuple(r.completed_step_count for r in fit.round_receipts) == (3, 0, 3)
    assert fit.round_receipts[1].skipped
    for index, receipt in enumerate(fit.step_receipts):
        assert receipt.round_positive_count == 7
        assert receipt.positive_group_counts == (((7, 2), (23, 2)) if index % 3 == 0
                                                  else ((23, 2 if index % 3 == 1 else 1),))
        assert receipt.expected_contribution_digest == receipt.completed_contribution_digest
        assert receipt.source_ids == PARTIES
    final_clocks = observed["steps"][-1]["snapshot"].parameter_steps
    assert [final_clocks[f"party:{p}"]["user_embeddings"] for p in PARTIES] == [2, 6, 0]
    assert all(final_clocks[f"party:{p}"]["field_embeddings"] == 6 for p in PARTIES)
    assert any(adapter is fixture["adapters"][41] and shape[1] == 0 and shape[0] > 0
               for adapter, _, shape in observed["deliveries"])
    assert fit.observe_bindings()["trace_ids"] == tuple(t.trace_id for t in fixture["traces"])
    assert len({t.context_id for t in fixture["traces"]}) == 2
    assert all(t.context_id != CONTEXT for t in fixture["traces"])
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        fit.step_receipts[0].step_index = 123
    record_property("unit2_connected_oracle", json.dumps(dict(
        dtype=str(dtype), fixture=FROZEN_FIXTURE, maximum_absolute_errors=maximum,
        actual_apply_count=len(observed["steps"]), actual_valid_count=len(observed["monitors"]),
        receiver_item_sequences=[step["receiver_items"] for step in expected_steps],
        traces=[dict(trace_id=t.trace_id, context_id=t.context_id,
                     ordered_positive_ids=t.ordered_positive_ids, negative_items=t.negative_items)
                for t in fixture["traces"]],
    ), sort_keys=True))


@pytest.mark.parametrize("mode", ["all_false", "zero_H", "empty_pools"])
def test_structural_none_is_not_numeric_zero(mode, monkeypatch, record_property):
    fixture = _fixture(mode=mode)
    oracle_steps, final = _oracle(fixture)
    observed = _capture(monkeypatch)
    fit = joint.fit_doi4_pri_none(**fixture["args"])
    tolerance = FROZEN_FIXTURE["tolerance"]["float64"]
    maximum = 0.0
    for step_index, (actual, expected) in enumerate(zip(observed["steps"], oracle_steps)):
        maximum = max(maximum, _assert_groups(actual["gradients"], expected["gradients"], tolerance))
        assert actual["snapshot"].parameter_steps == expected["clocks"]
        query = actual["gradients"]["center"]["queries"]
        if mode == "empty_pools":
            assert query is None
            assert actual["encodes"] == [] and actual["source_vjps"] == []
            assert all(actual["gradients"][f"party:{p}"]["field_embeddings"] is None for p in PARTIES)
        else:
            assert tuple(actual["encodes"]) == PARTIES and tuple(actual["source_vjps"]) == PARTIES
            assert query is not None
            # Adam may change zero-H readout weights after the first global step.
            if mode == "all_false" or step_index == 0:
                assert torch.count_nonzero(query) == 0
            assert actual["gradients"]["party:41"]["field_embeddings"] is not None
    _assert_groups(fit.observe_parameters(), final, tolerance)
    record_property("unit2_zero_paths", json.dumps({"mode": mode, "maximum_gradient_error": maximum}))


def _assert_inputs_open(fixture):
    check_feature_runtime_binding(fixture["runtime"], binding_id=fixture["args"]["input_bindings"][7][0])
    for party, adapter in fixture["adapters"].items():
        assert adapter.binding_id == fixture["args"]["input_bindings"][party][3]


@pytest.mark.parametrize("case", [
    "zero_rounds", "missing_pool", "extra_pool", "wrong_receiver", "bad_online",
    "bad_user", "bad_domain", "bad_grant", "closed_runtime", "closed_adapter",
    "bad_runtime_binding", "bad_map_id", "wrong_dtype", "test_candidates", "empty_valid",
])
def test_static_failures_precede_optimizer_creation(case, monkeypatch):
    fixture = _fixture(denied=case == "bad_grant")
    args = fixture["args"]
    if case == "zero_rounds":
        args["rounds"] = ()
    elif case == "missing_pool":
        del args["step_source_pools"][(0, 0)]
    elif case == "extra_pool":
        args["step_source_pools"][(2, 0)] = {}
    elif case == "wrong_receiver":
        args["step_source_pools"][(0, 0)][41] = PARTIES
    elif case == "bad_online":
        args["inference_online_party_ids"] = (7, 23)
    elif case == "bad_user":
        args["user_ids_by_party"][7] = (0, 1)
    elif case == "bad_domain":
        old = args["validation_candidates"]
        args["validation_candidates"] = replace(old, candidates=(
            EvaluationCandidate(7, 0, 12, (1, 2, 3)), *old.candidates[1:],
        ))
    elif case == "closed_runtime":
        close_feature_runtime(fixture["runtime"])
    elif case == "closed_adapter":
        close_binary_input_adapter(fixture["adapters"][7])
    elif case == "bad_runtime_binding":
        old = args["input_bindings"][7]
        args["input_bindings"][7] = ("0" * 64, *old[1:])
    elif case == "bad_map_id":
        old = args["input_bindings"][7]
        args["input_bindings"][7] = (old[0], "not-a-digest", old[2], old[3])
    elif case == "wrong_dtype":
        args["initial_values"]["party:7"]["field_embeddings"] = args["initial_values"]["party:7"]["field_embeddings"].float()
    elif case == "test_candidates":
        args["validation_candidates"] = replace(args["validation_candidates"], split="test")
    elif case == "empty_valid":
        args["validation_candidates"] = replace(args["validation_candidates"], candidates=())
    prepared = []
    original = joint.prepare_optimizer_owners
    def prepare(*values, **keywords):
        prepared.append(True)
        return original(*values, **keywords)
    monkeypatch.setattr(joint, "prepare_optimizer_owners", prepare)
    with pytest.raises((TypeError, ValueError)):
        joint.fit_doi4_pri_none(**args)
    assert prepared == []


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "stale", "unexpected", "wrong_geometry"])
def test_contribution_barrier_rejects_corruption_before_apply(corruption, monkeypatch):
    fixture = _fixture()
    real_record = joint._record_contribution
    fired, applied, handles = [], [], []
    original_prepare = joint.prepare_optimizer_owners
    def prepare(*args, **kwargs):
        owner = original_prepare(*args, **kwargs)
        handles.append(owner)
        return owner
    def record(records, key, geometry):
        if not fired:
            fired.append(True)
            if corruption == "missing":
                return
            if corruption == "duplicate":
                real_record(records, key, geometry)
            elif corruption == "stale":
                dependency = key[-1][0]
                versions = ((*dependency[:-1], dependency[-1] + 1), *key[-1][1:])
                key = (*key[:-1], versions)
            elif corruption == "unexpected":
                key = (key[0], "unrequested-phase", *key[2:])
            else:
                geometry = ("wrong-geometry",)
        return real_record(records, key, geometry)
    def apply(*args, **kwargs):
        applied.append(True)
        raise AssertionError("invalid contribution barrier reached optimizer")
    monkeypatch.setattr(joint, "prepare_optimizer_owners", prepare)
    monkeypatch.setattr(joint, "_record_contribution", record)
    monkeypatch.setattr(joint, "apply_optimizer_gradients", apply)
    with pytest.raises(ValueError):
        joint.fit_doi4_pri_none(**fixture["args"])
    assert fired == [True] and applied == []
    assert handles[0].status == "closed"
    _assert_inputs_open(fixture)


def test_nonfinite_completed_gradient_never_reaches_apply(monkeypatch):
    fixture = _fixture()
    real = joint._accumulate_gradients
    poisoned, calls = [], []
    def accumulate(groups, owner, name, gradient):
        result = real(groups, owner, name, gradient)
        if not poisoned and owner == "center" and name == "item_embeddings":
            poisoned.append(True)
            groups[owner][name].fill_(float("nan"))
        return result
    def apply(*args, **kwargs):
        calls.append(True)
        raise AssertionError("nonfinite final gradients reached apply")
    monkeypatch.setattr(joint, "_accumulate_gradients", accumulate)
    monkeypatch.setattr(joint, "apply_optimizer_gradients", apply)
    with pytest.raises(ValueError):
        joint.fit_doi4_pri_none(**fixture["args"])
    assert poisoned == [True] and calls == []
    _assert_inputs_open(fixture)


def test_physical_step_failure_is_fail_stop_without_cursor_advance(monkeypatch):
    fixture = _fixture()
    original_prepare = joint.prepare_optimizer_owners
    original_step = torch.optim.Adam.step
    handles, parameters, attempted, advanced = [], [], [], []
    def prepare(groups, **kwargs):
        parameters.extend((p, p.detach().clone()) for group in groups.values() for p in group.values())
        result = original_prepare(groups, **kwargs)
        handles.append(result)
        return result
    def step(optimizer, *args, **kwargs):
        attempted.append(True)
        original_step(optimizer, *args, **kwargs)
        raise RuntimeError("unit2 injected failure after physical Adam step")
    def advance(*args, **kwargs):
        advanced.append(True)
        raise AssertionError("failed physical apply advanced the round cursor")
    monkeypatch.setattr(joint, "prepare_optimizer_owners", prepare)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(joint, "advance_round_drain_cursor", advance)
    with pytest.raises(OptimizerStepError) as caught:
        joint.fit_doi4_pri_none(**fixture["args"])
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert attempted == [True] and advanced == [] and handles[0].status == "closed"
    assert any(not torch.equal(value, old) for value, old in parameters)
    _assert_inputs_open(fixture)


def test_successful_physical_apply_with_invalid_receipt_does_not_advance(monkeypatch):
    fixture = _fixture()
    original = joint.apply_optimizer_gradients
    applied, advanced = [], []
    def apply(*args, **kwargs):
        receipt = original(*args, **kwargs)
        applied.append(receipt)
        return replace(receipt, generation_after=receipt.generation_after + 1)
    def advance(*args, **kwargs):
        advanced.append(True)
        raise AssertionError("invalid receipt advanced cursor")
    monkeypatch.setattr(joint, "apply_optimizer_gradients", apply)
    monkeypatch.setattr(joint, "advance_round_drain_cursor", advance)
    with pytest.raises((ValueError, RuntimeError)):
        joint.fit_doi4_pri_none(**fixture["args"])
    assert len(applied) == 1 and advanced == []
    _assert_inputs_open(fixture)


def test_monitor_failure_does_not_return_a_terminal_or_retry(monkeypatch):
    fixture = _fixture()
    observed = _capture(monkeypatch)
    calls = []
    def monitor(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("unit2 monitor failure")
    monkeypatch.setattr(joint, "evaluate_candidates", monitor)
    with pytest.raises(RuntimeError, match="unit2 monitor failure"):
        joint.fit_doi4_pri_none(**fixture["args"])
    assert calls == [True] and len(observed["steps"]) == 3
    assert observed["handles"][0].status == "closed"
    _assert_inputs_open(fixture)


def test_caller_and_observation_isolation_and_real_terminal_once(monkeypatch):
    fixture = _fixture()
    observed = _capture(monkeypatch)
    fit = joint.fit_doi4_pri_none(**fixture["args"])
    users, items = _ids((2, 0, 1, 2)), _ids((4, 4, 4, 4))
    score = joint.score_doi4_pri_none(fit, users, items, expected_context_id=CONTEXT)
    assert score.dtype == np.float64 and score.flags.owndata and np.isfinite(score).all()
    assert score[0] == score[3]
    values, bindings = fit.observe_parameters(), fit.observe_bindings()
    frozen_values, frozen_bindings = _clone(values), deepcopy(bindings)
    for group in values.values():
        for value in group.values():
            value.add_(999)
    bindings["user_ids_by_party"][7] = (999,)
    bindings["input_bindings"].clear()
    for group in fixture["args"]["initial_values"].values():
        for value in group.values():
            value.fill_(123)
    borrowed = fixture["runtime"]
    fixture["args"]["feature_runtimes"].clear()
    fixture["args"]["adapters"] = {party: object() for party in PARTIES}
    fixture["args"]["input_bindings"].clear()
    fixture["args"]["inference_source_pools"].clear()
    fixture["args"]["user_ids_by_party"].clear()
    _assert_groups(fit.observe_parameters(), frozen_values, {"atol": 0, "rtol": 0})
    assert fit.observe_bindings() == frozen_bindings
    np.testing.assert_array_equal(
        joint.score_doi4_pri_none(fit, users, items, expected_context_id=CONTEXT), score)
    empty = joint.score_doi4_pri_none(fit, _ids(()), _ids(()), expected_context_id=CONTEXT)
    assert empty.shape == (0,) and empty.dtype == np.float64 and empty.flags.owndata
    # Genuine TEST construction occurs only after successful fit finalization.
    candidates = build_evaluation_candidates(
        fixture["bundle"], (10, 11, 12), split="test", negatives_per_positive=3, base_seed=20261028,
    )
    handle = fit.terminal_handle
    assert handle is fit.terminal_handle
    receipt = observe_terminal_evaluation(handle)
    receipt["state"] = "complete"
    assert observe_terminal_evaluation(handle)["state"] == "ready"
    result = evaluate_terminal_once(handle, candidates, expected_terminal_id=fit.terminal_id)
    assert result["num_targets"] == 3
    result["num_targets"] = -1
    assert observe_terminal_evaluation(handle)["metrics"]["num_targets"] == 3
    with pytest.raises((ValueError, RuntimeError)):
        evaluate_terminal_once(handle, candidates, expected_terminal_id=fit.terminal_id)
    assert observed["monitors"] == [("valid", (7, 8, 9), (1, 3))] * 3
    close_feature_runtime(borrowed)
    with pytest.raises(ValueError):
        joint.score_doi4_pri_none(fit, users, items, expected_context_id=CONTEXT)


def test_scoring_rows_do_not_clone_the_whole_terminal_item_table(monkeypatch):
    fixture = _fixture()
    fit = joint.fit_doi4_pri_none(**fixture["args"])
    original, whole_table_clones = torch.Tensor.clone, []
    def clone(tensor, *args, **kwargs):
        if tensor.shape == (12, 3):
            whole_table_clones.append(True)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "clone", clone)
    for items in ((1, 2), (3, 3)):
        scores = joint.score_doi4_pri_none(fit, _ids((0, 0)), _ids(items), expected_context_id=CONTEXT)
        assert scores.shape == (2,)
    assert whole_table_clones == []


def test_nonleaf_noncontiguous_initial_values_are_copied_without_backward_to_caller(monkeypatch):
    fixture = _fixture()
    leaf = fixture["args"]["initial_values"]["center"]["item_embeddings"].detach().clone().requires_grad_()
    initial = leaf.T.contiguous().T
    assert not initial.is_leaf and not initial.is_contiguous()
    fixture["args"]["initial_values"]["center"]["item_embeddings"] = initial
    before, version, pointer = initial.detach().clone(), initial._version, initial.data_ptr()
    observed = _capture(monkeypatch)
    fit = joint.fit_doi4_pri_none(**fixture["args"])
    assert leaf.grad is None and initial._version == version and initial.data_ptr() == pointer
    torch.testing.assert_close(initial, before, rtol=0, atol=0)
    owned = observed["parameter_objects"]["center"]["item_embeddings"]
    assert owned is not initial and owned.data_ptr() != initial.data_ptr()
    assert owned.is_leaf and owned.requires_grad
    public = fit.observe_parameters()["center"]["item_embeddings"]
    assert not public.requires_grad and public.data_ptr() != owned.data_ptr()


@pytest.mark.parametrize("case", ["context", "unknown_user", "unknown_item", "float_ids", "rank", "length"])
def test_score_rejects_invalid_public_inputs(case):
    fixture = _fixture()
    # One already-completed empty round still must monitor and finalize.
    fixture["args"]["rounds"] = (fixture["args"]["rounds"][1],)
    fixture["args"]["step_source_pools"] = {}
    fit = joint.fit_doi4_pri_none(**fixture["args"])
    users, items, context = _ids((0,)), _ids((0,)), CONTEXT
    if case == "context":
        context = "e" * 64
    elif case == "unknown_user":
        users = _ids((7,))
    elif case == "unknown_item":
        items = _ids((12,))
    elif case == "float_ids":
        users = np.array([0.0], dtype=np.float64)
    elif case == "rank":
        users = users.reshape(1, 1)
    else:
        items = _ids(())
    with pytest.raises((TypeError, ValueError)):
        joint.score_doi4_pri_none(fit, users, items, expected_context_id=context)


def test_empty_round_has_one_valid_and_zero_apply_and_stable_terminal_identity(monkeypatch):
    fixture = _fixture()
    fixture["args"]["rounds"] = (fixture["args"]["rounds"][1],)
    fixture["args"]["step_source_pools"] = {}
    observed = _capture(monkeypatch)
    fit = joint.fit_doi4_pri_none(**fixture["args"])
    assert observed["steps"] == [] and len(observed["monitors"]) == 1
    assert fit.step_receipts == () and len(fit.round_receipts) == 1
    receipt = fit.round_receipts[0]
    assert receipt.skipped and receipt.round_positive_count == 0
    assert receipt.generation_before == receipt.generation_after == 0
    assert receipt.completed_step_count == receipt.expected_step_count == 0
    _assert_groups(fit.observe_parameters(), fixture["args"]["initial_values"], {"atol": 0, "rtol": 0})
    for identity in (fit.terminal_id, fit.scheduled_end_id):
        assert len(identity) == 64 and all(char in "0123456789abcdef" for char in identity)
    assert fit.context_id == CONTEXT
    assert fit.initialization_id == "UNIT2-JOINT-2x5-v1-initial"
    assert fit.profile_id == "UNIT2-JOINT-2x5-v1-profile"
