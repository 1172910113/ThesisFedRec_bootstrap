"""Bounded CPU joint pri-none execution with explicit split-graph barriers.

This is not a distributed transport, checkpoint, source selector or initializer.
Caller coordination must keep inputs stable through validation and copying.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch

from src.data.feature_runtime import FeatureRuntime, check_feature_runtime_binding
from src.data.model_inputs import BinaryInputAdapter, deliver_binary_field_units
from src.evaluation.ranking import EvaluationCandidateSet, evaluate_candidates
from src.evaluation.terminal import (
    TerminalEvaluationHandle, _bind_terminal_evaluation, _candidate_digest,
    _copy_candidates, _validated_k_values,
)
from src.federation.doi4_batch import (
    Doi4BatchRequests, pack_doi4_source_slots, prepare_doi4_batch_requests,
    tensorize_binary_delivery,
)
from src.federation.optimizer_owners import (
    OptimizerStepReceipt, apply_optimizer_gradients, close_optimizer_owners,
    prepare_optimizer_owners,
)
from src.federation.round_drain import (
    RoundDrainCursor, RoundDrainPlan, advance_round_drain_cursor,
    peek_round_drain_step, start_round_drain,
)
from src.federation.trace_batch import pack_party_group_batch
from src.methods.doi4.composition import (
    encode_readout_binary_units, group_loss_pri_none_from_fused,
    score_pri_none_from_fused,
)
from src.methods.doi4.fusion import center_fusion
from src.methods.doi4.parameters import doi4_parameter_specs


_FUSION = ('fusion_w_query', 'fusion_w_key', 'fusion_w_value', 'fusion_w_output')
_HEAD = ('w_fused', 'score_weight', 'score_bias')
_SOURCE = ('field_embeddings', 'value_embeddings', 'readout_w_key', 'readout_w_value')


@dataclass(frozen=True, slots=True, kw_only=True)
class StepReceipt:
    step_id: str
    round_index: int
    step_index: int
    plan_id: str
    trace_id: str
    request_id: str
    expected_contribution_digest: str
    completed_contribution_digest: str
    receiver_ids: tuple[int, ...]
    source_ids: tuple[int, ...]
    positive_group_counts: tuple[tuple[int, int], ...]
    round_positive_count: int
    optimizer_receipt: OptimizerStepReceipt
    cursor_after: RoundDrainCursor


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundReceipt:
    round_index: int
    plan_id: str
    trace_id: str
    expected_step_count: int
    completed_step_count: int
    round_positive_count: int
    skipped: bool
    generation_before: int
    generation_after: int
    validation_metrics: tuple[tuple[str, float | int], ...]
    validation_candidate_digest: str


class Doi4PriNoneFit:
    """Prepare-only terminal result; observations never expose scoring storage."""

    __slots__ = ('_values', '_config', '_bindings', '_terminal_id', '_scheduled_end_id',
                 '_initialization_id', '_profile_id', '_rounds', '_steps', '_handle')

    def __new__(cls, *args: object, **kwargs: object) -> Doi4PriNoneFit:
        raise TypeError('use fit_doi4_pri_none to create a fit')

    @property
    def terminal_id(self) -> str:
        return self._terminal_id

    @property
    def scheduled_end_id(self) -> str:
        return self._scheduled_end_id

    @property
    def context_id(self) -> str:
        return self._config['context_id']

    @property
    def initialization_id(self) -> str:
        return self._initialization_id

    @property
    def profile_id(self) -> str:
        return self._profile_id

    @property
    def terminal_handle(self) -> TerminalEvaluationHandle:
        return self._handle

    @property
    def round_receipts(self) -> tuple[RoundReceipt, ...]:
        return self._rounds

    @property
    def step_receipts(self) -> tuple[StepReceipt, ...]:
        return self._steps

    def observe_parameters(self) -> dict[str, dict[str, torch.Tensor]]:
        return {o: {n: t.detach().clone() for n, t in group.items()}
                for o, group in self._values.items()}

    def observe_bindings(self) -> dict[str, object]:
        return deepcopy(self._bindings)


def _integer(value: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f'{name} must be an exact built-in int')
    if not 0 <= value <= 2**63 - 1:
        raise ValueError(f'{name} must be a nonnegative int64 ID')


def _ids(value: tuple[int, ...], name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f'{name} must be an exact tuple')
    for v in value:
        _integer(v, name)
    if any(a >= b for a, b in zip(value, value[1:])):
        raise ValueError(f'{name} must be sorted and unique')
    return value


def _mapping(value: dict, name: str, keys: tuple | set, *, integers: bool) -> None:
    if type(value) is not dict:
        raise TypeError(f'{name} must be an exact dict')
    for key in value:
        if integers:
            _integer(key, name + ' key')
        elif type(key) is not str:
            raise TypeError(f'{name} keys must be exact str')
    if set(value) != set(keys):
        raise ValueError(f'{name} has missing or extra keys')


def _digest(value: str, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f'{name} must be exact str')
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be lowercase 64-hex')


def _hash(tag: str, content: object) -> str:
    raw = json.dumps(dict(format=tag, version=1, content=content), sort_keys=True,
                     ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _values_digest(values: dict) -> str:
    content = []
    for owner in sorted(values):
        for name in sorted(values[owner]):
            tensor = values[owner][name]
            arr = tensor.detach().cpu().numpy()
            dtype = arr.dtype.newbyteorder('<')
            raw = arr.astype(dtype, copy=False).tobytes(order='C')
            content.append((owner, name, tuple(tensor.shape), dtype.str,
                            hashlib.sha256(raw).hexdigest()))
    return _hash('thesisfedrec.joint_values', content)


def _floating(value: torch.Tensor, dtype: torch.dtype | None = None) -> None:
    if (not isinstance(value, torch.Tensor) or value.is_nested
            or value.layout != torch.strided or value.device.type != 'cpu'
            or value.dtype not in (torch.float32, torch.float64)):
        raise TypeError('parameters/gradients must be dense strided CPU float32/64')
    if dtype is not None and value.dtype != dtype:
        raise TypeError('all floating tensors must share one dtype')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('parameters/gradients must be finite')


def _check_inputs(cfg: dict) -> None:
    for party, binding in cfg['input_bindings'].items():
        check_feature_runtime_binding(cfg['feature_runtimes'][party], binding_id=binding[0])
        if cfg['adapters'][party].binding_id != binding[3]:
            raise ValueError('binary adapter binding mismatch')


def _deliver(cfg: dict, party: int, ids: tuple[int, ...]):
    payload = deliver_binary_field_units(
        cfg['adapters'][party], entity_ids=ids,
        expected_binding_id=cfg['input_bindings'][party][3],
    )
    return tensorize_binary_delivery(
        payload, expected_entity_ids=ids, public_item_count=cfg['I'],
        local_field_count=cfg['field_counts'][party],
    )


def _requests(cfg: dict, rows: dict, pools: dict, online: tuple) -> Doi4BatchRequests:
    return prepare_doi4_batch_requests(
        rows, source_party_ids_by_receiver=pools, online_party_ids=online,
        public_item_count=cfg['I'], context_id=cfg['context_id'],
    )


def _candidate_preflight(cfg: dict, candidates: EvaluationCandidateSet) -> None:
    _check_inputs(cfg)
    # Include empty receivers so fixed online geometry is checked independently
    # of which registered parties happen to have current targets.
    rows = {p: [] for p in cfg['user_ids_by_party']}
    for candidate in candidates.candidates:
        if candidate.user_id not in cfg['routing']:
            raise ValueError('candidate user is not owned by a registered party')
        if any(i >= cfg['I'] for i in candidate.item_ids):
            raise ValueError('candidate item is outside the processed domain')
        rows[cfg['routing'][candidate.user_id][0]].extend(candidate.item_ids)
    request = _requests(cfg, {p: tuple(v) for p, v in rows.items()},
                        cfg['inference_source_pools'], cfg['inference_online_party_ids'])
    for party, items in zip(request.source_party_ids, request.source_item_ids):
        _deliver(cfg, party, items)


def _validate_parameters(values: dict, users: dict) -> tuple[int, int, int, torch.dtype, dict]:
    owners = {'center', *(f'party:{p}' for p in users)}
    _mapping(values, 'initial_values', owners, integers=False)
    for group in values.values():
        if type(group) is not dict or any(type(n) is not str for n in group):
            raise TypeError('parameter groups must be exact dicts with str keys')
        for t in group.values():
            _floating(t)
    c = values['center']
    if 'item_embeddings' not in c or 'queries' not in c:
        raise ValueError('center requires item_embeddings and queries')
    if c['item_embeddings'].ndim != 2 or c['queries'].ndim != 2:
        raise ValueError('item embeddings and queries must be rank two')
    I, d = c['item_embeddings'].shape
    K, h = c['queries'].shape
    if not (1 <= I <= 128 and 1 <= d <= 16 and 1 <= h <= 16 and 1 <= K <= 4):
        raise ValueError('item/width/slot dimensions exceed bounded Unit2 capacity')
    dtype = c['item_embeddings'].dtype
    fields = {}
    for p in users:
        group = values[f'party:{p}']
        if 'field_embeddings' not in group or group['field_embeddings'].ndim != 2:
            raise ValueError('party field_embeddings must have shape [J,r]')
        J, r = group['field_embeddings'].shape
        if not 0 <= J <= 16 or not 1 <= r <= 16:
            raise ValueError('field dimensions exceed bounded Unit2 capacity')
        fields[p] = J
        specs = doi4_parameter_specs(party_id=p, public_item_count=I,
            local_user_count=len(users[p]), local_field_count=J, item_dim=d,
            unit_dim=r, compatible_dim=h, slot_count=K)
        for owner, subset in [('center', specs[:9]), (f'party:{p}', specs[9:])]:
            if set(values[owner]) != {s.name for s in subset}:
                raise ValueError('exact nine center/five party parameter names required')
            for spec in subset:
                t = values[owner][spec.name]
                _floating(t, dtype)
                if tuple(t.shape) != spec.shape:
                    raise ValueError(f'{owner}/{spec.name} has incompatible shape')
    return I, K, h, dtype, fields


def _prepare_config(rounds, values, users, adapters, runtimes, bindings,
                    ceilings, step_pools, inference_pools, online, context, valid, ks):
    _digest(context, 'context_id')
    if type(users) is not dict:
        raise TypeError('user_ids_by_party must be exact dict')
    for p in users:
        _integer(p, 'party_id')
    parties = tuple(sorted(users))
    if not 1 <= len(parties) <= 8:
        raise ValueError('bounded fit requires 1..8 registered parties')
    routing = {}
    owned_users = {}
    for p in parties:
        owned_users[p] = _ids(users[p], 'owned users')
        for row, user in enumerate(owned_users[p]):
            if user in routing:
                raise ValueError('users must be disjoint across parties')
            routing[user] = (p, row)
    if len(routing) > 32:
        raise ValueError('bounded fit supports at most 32 users')
    I, K, h, dtype, fields = _validate_parameters(values, owned_users)
    for name, mapping in [('adapters', adapters), ('feature_runtimes', runtimes),
                          ('input_bindings', bindings), ('allowed_sources', ceilings),
                          ('inference_source_pools', inference_pools)]:
        _mapping(mapping, name, parties, integers=True)
    cfg = dict(I=I, K=K, h=h, dtype=dtype, field_counts=fields,
               user_ids_by_party=owned_users, routing=routing,
               adapters=dict(adapters), feature_runtimes=dict(runtimes),
               input_bindings={}, context_id=context, inference_source_pools={})
    permission = {}
    for p in parties:
        if type(adapters[p]) is not BinaryInputAdapter or type(runtimes[p]) is not FeatureRuntime:
            raise TypeError('prepared adapter/runtime handles required')
        b = bindings[p]
        if type(b) is not tuple:
            raise TypeError('each binding must be an exact tuple')
        if len(b) != 4:
            raise ValueError('binding requires runtime, map, epoch, adapter')
        for index in (0, 1, 3):
            _digest(b[index], 'input binding')
        _integer(b[2], 'schema_epoch')
        cfg['input_bindings'][p] = b
        permission[p] = _ids(ceilings[p], 'permission ceiling')
        if not set(permission[p]).issubset(parties):
            raise ValueError('permission ceiling includes unknown parties')
        pool = _ids(inference_pools[p], 'inference pool')
        if not set(pool).issubset(permission[p]):
            raise ValueError('inference pool exceeds permission ceiling')
        cfg['inference_source_pools'][p] = pool
    cfg['inference_online_party_ids'] = _ids(online, 'inference online roster')
    if cfg['inference_online_party_ids'] != parties:
        raise ValueError('inference online roster must contain exactly registered parties')
    _check_inputs(cfg)
    # Even empty requests verify the adapter's actual field width against J.
    for p in parties:
        _deliver(cfg, p, ())
    if type(rounds) is not tuple:
        raise TypeError('rounds must be an exact tuple')
    if not 1 <= len(rounds) <= 4:
        raise ValueError('bounded fit requires 1..4 rounds, not zero plans')
    if any(type(p) is not RoundDrainPlan for p in rounds):
        raise TypeError('rounds must contain prepared RoundDrainPlan objects')
    try:
        round_ids = tuple(plan.round_index for plan in rounds)
        _ids(round_ids, 'round indices')
        keys = {(p.round_index, step) for p in rounds for step in range(p.step_count)}
    except AttributeError as exc:
        raise ValueError('unprepared RoundDrainPlan') from exc
    if type(step_pools) is not dict:
        raise TypeError('step_source_pools must be exact dict')
    for key in step_pools:
        if type(key) is not tuple:
            raise TypeError('step pool key must be tuple')
        if len(key) != 2:
            raise ValueError('step pool key must contain round and step')
        for value in key:
            _integer(value, 'step pool key')
    if set(step_pools) != keys:
        raise ValueError('step pools must match every nonterminal plan step exactly')
    if (len(keys) > 32 or sum(p.round_positive_count for p in rounds) > 256
            or sum(p.round_positive_count * (p.negatives_per_positive + 1) for p in rounds) > 2048):
        raise ValueError('fit exceeds bounded step/group/label-row capacity')
    copied_pools = {}
    for plan in rounds:
        if not set(plan.selected_party_ids).issubset(parties):
            raise ValueError('selected roster includes unknown party')
        for party in plan.selected_party_ids:
            if any(u not in owned_users[party] for u in plan.user_ids_by_party[party]):
                raise ValueError('plan user is not owned by the corresponding party')
        cursor = start_round_drain(plan)
        for step in range(plan.step_count):
            key = (plan.round_index, step)
            descriptor = peek_round_drain_step(plan, cursor, requested_source_party_ids=())
            receivers = tuple(b.party_id for b in descriptor.batches)
            _mapping(step_pools[key], 'step pools', receivers, integers=True)
            pools = {p: _ids(step_pools[key][p], 'source pool') for p in receivers}
            for p, sources in pools.items():
                if not set(sources).issubset(set(permission[p]) & set(plan.selected_party_ids)):
                    raise ValueError('training pool exceeds permission or online roster')
            copied_pools[key] = pools
            packed = [pack_party_group_batch(b, dtype=dtype) for b in descriptor.batches]
            request = _requests(cfg, {b.party_id: tuple(b.item_ids.tolist()) for b in packed},
                                pools, plan.selected_party_ids)
            for p, items in zip(request.source_party_ids, request.source_item_ids):
                _deliver(cfg, p, items)
            # Preflight walks geometry, not the authoritative training cursor.
            cursor = RoundDrainCursor(plan_id=plan.plan_id, next_step_index=step + 1)
    candidates = _copy_candidates(valid, split='valid')
    if len(candidates.candidates) > 16 or any(len(c.item_ids) > 32 for c in candidates.candidates):
        raise ValueError('VALID exceeds bounded candidate capacity')
    ks = _validated_k_values(ks)
    _candidate_preflight(cfg, candidates)
    return cfg, copied_pools, permission, candidates, ks


def _source_outputs(cfg: dict, values: dict, request: Doi4BatchRequests) -> dict:
    result = {}
    for source, items in zip(request.source_party_ids, request.source_item_ids):
        delivery = _deliver(cfg, source, items)
        p = values[f'party:{source}']
        H, available, _ = encode_readout_binary_units(
            delivery.local_field_ids, delivery.binary_values, delivery.valid_unit_mask,
            field_embeddings=p['field_embeddings'], value_embeddings=p['value_embeddings'],
            queries=values['center']['queries'], w_key=p['readout_w_key'], w_value=p['readout_w_value'],
        )
        if H is None:
            raise ValueError('every requested source must execute a Tensor publication')
        result[source] = (items, H, available)
    return result


def _fuse(slot, item_rows, center):
    return center_fusion(slot.compatible_tokens, slot.valid_token_mask, item_rows,
                         *(center[name] for name in _FUSION))


def _score_values(cfg: dict, values: dict, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    _check_inputs(cfg)
    for ids in (user_ids, item_ids):
        if (type(ids) is not np.ndarray or ids.dtype != np.dtype('int64')
                or not ids.dtype.isnative or ids.dtype.metadata is not None):
            raise TypeError('scoring IDs must be plain native metadata-free int64 arrays')
        if ids.ndim != 1:
            raise ValueError('scoring IDs must be rank one')
    if user_ids.shape != item_ids.shape:
        raise ValueError('scoring arrays must align')
    if np.any(item_ids < 0) or np.any(item_ids >= cfg['I']):
        raise ValueError('scoring item outside domain')
    if any(int(u) not in cfg['routing'] for u in user_ids):
        raise ValueError('scoring user is unowned')
    output = np.empty(len(user_ids), dtype=np.float64)
    if not len(user_ids):
        return output
    positions = {p: [] for p in cfg['user_ids_by_party']}
    for row, user in enumerate(user_ids):
        positions[cfg['routing'][int(user)][0]].append(row)
    positions = {p: rows for p, rows in positions.items() if rows}
    request = _requests(cfg, {p: tuple(int(item_ids[i]) for i in rows) for p, rows in positions.items()},
                        {p: cfg['inference_source_pools'][p] for p in positions},
                        cfg['inference_online_party_ids'])
    with torch.no_grad():
        pubs = _source_outputs(cfg, values, request)
        slots = pack_doi4_source_slots(request, source_publications=pubs, slot_count=cfg['K'],
            compatible_dim=cfg['h'], dtype=cfg['dtype'], expected_context_id=cfg['context_id'])
        center = values['center']
        for slot in slots:
            p = slot.receiver_party_id
            rows = positions[p]
            items = center['item_embeddings'][list(slot.entity_ids)]
            users = values[f'party:{p}']['user_embeddings'][[cfg['routing'][int(user_ids[i])][1] for i in rows]]
            z, available, _ = _fuse(slot, items, center)
            logits = score_pri_none_from_fused(z, available, user_embeddings=users,
                item_embeddings=items, **{n: center[n] for n in _HEAD})
            output[rows] = logits.detach().numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(output)):
        raise ValueError('scoring produced nonfinite output')
    return output


def score_doi4_pri_none(fit: Doi4PriNoneFit, user_ids: np.ndarray, item_ids: np.ndarray,
                        *, expected_context_id: str) -> np.ndarray:
    if type(fit) is not Doi4PriNoneFit or not hasattr(fit, '_handle'):
        raise TypeError('fit must be a completed prepared Doi4PriNoneFit')
    _digest(expected_context_id, 'expected_context_id')
    if expected_context_id != fit.context_id:
        raise ValueError('scoring context mismatch')
    return _score_values(fit._config, fit._values, user_ids, item_ids)


def _record_contribution(records: list, key: tuple, geometry: tuple) -> None:
    records.append((key, geometry))


def _validate_contributions(expected: dict, records: list) -> None:
    seen = {}
    for key, geometry in records:
        if key in seen:
            raise ValueError('duplicate contribution')
        if key not in expected or geometry != expected[key]:
            raise ValueError('unexpected/stale contribution or incorrect geometry')
        seen[key] = geometry
    if seen != expected:
        raise ValueError('missing contribution')


def _accumulate_gradients(gradients: dict, owner: str, name: str, gradient) -> None:
    if gradient is not None:
        old = gradients[owner][name]
        gradients[owner][name] = gradient.detach().clone() if old is None else old + gradient.detach()


def _scatter(parameter, rows, gradient):
    result = torch.zeros_like(parameter)
    result.index_add_(0, torch.tensor(rows, dtype=torch.int64), gradient)
    return result


def _geometry(rows: tuple, tensors: tuple) -> tuple:
    return rows, tuple((tuple(t.shape), str(t.dtype)) for t in tensors)


def _joint_gradients(cfg, parameters, plan, cursor, pools, generation, versions, clocks):
    _check_inputs(cfg)
    sources = tuple(sorted({s for pool in pools.values() for s in pool}))
    step = peek_round_drain_step(plan, cursor, requested_source_party_ids=sources)
    packed = {b.party_id: pack_party_group_batch(b, dtype=cfg['dtype']) for b in step.batches}
    request = _requests(cfg, {p: tuple(b.item_ids.tolist()) for p, b in packed.items()}, pools,
                        plan.selected_party_ids)
    geometry = asdict(request)
    request_id = _hash('thesisfedrec.joint_request', geometry)
    dependency_versions = tuple((o, versions[o], n, clocks[o][n])
                                for o in sorted(parameters) for n in sorted(parameters[o]))
    step_id = _hash('thesisfedrec.joint_step', dict(context=cfg['context_id'], plan=plan.plan_id,
        trace=plan.trace_id, round=plan.round_index, step=cursor.next_step_index,
        generation=generation, parameter_versions=dependency_versions,
        bindings=tuple(cfg['input_bindings'].items()), request=geometry))
    expected, records = {}, []
    center = parameters['center']
    dtype = str(cfg['dtype']); d = center['item_embeddings'].shape[1]
    h, K = cfg['h'], cfg['K']
    source_rows = dict(zip(request.source_party_ids, request.source_item_ids))

    def key(phase, sender, receiver, kind):
        return (step_id, phase, sender, receiver, kind, dependency_versions)

    def declare(phase, sender, receiver, kind, rows, shapes, dtypes=None):
        expected[key(phase, sender, receiver, kind)] = (
            rows, tuple((shape, dt) for shape, dt in zip(shapes, dtypes or (dtype,) * len(shapes))))

    for s, rows in source_rows.items():
        declare('publication', s, s, 'H', rows, ((len(rows), K, h), (len(rows),)), (dtype, 'torch.bool'))
        group = parameters[f'party:{s}']
        declare('source_vjp', s, s, 'local_encoder', rows, tuple(tuple(group[n].shape) for n in _SOURCE))
        declare('shared_query', s, 'center', 'queries', (), ((K, h),))
    for p, batch in packed.items():
        rows = tuple(batch.item_ids.tolist()); E = len(rows)
        declare('local_reply', p, p, 'g_z', rows, ((E, h),))
        declare('center_vjp', 'center', p, 'fusion', rows, tuple(tuple(center[n].shape) for n in _FUSION))
        declare('direct_item', p, 'center', 'item_rows', rows, ((E, d),))
        declare('query_item', 'center', p, 'item_rows', rows, ((E, d),))
        for s in pools[p]:
            rows_s = source_rows[s]
            declare('receiver_source', p, s, 'g_H', rows_s, ((len(rows_s), K, h),))
    # All keys and shapes above exist before any forward/backward contribution.
    pubs = _source_outputs(cfg, parameters, request)
    leaves = {}
    for s, (ids, H, avail) in pubs.items():
        _record_contribution(records, key('publication', s, s, 'H'), _geometry(ids, (H, avail)))
        leaves[s] = (ids, H.detach().clone().requires_grad_(True), avail)
    slots = pack_doi4_source_slots(request, source_publications=leaves, slot_count=K,
        compatible_dim=h, dtype=cfg['dtype'], expected_context_id=cfg['context_id'])
    gradients = {o: dict.fromkeys(group) for o, group in parameters.items()}
    incident = {s: [] for s in sources}
    for slot in slots:
        p = slot.receiver_party_id; batch = packed[p]; rows = slot.entity_ids
        user_rows = tuple(cfg['routing'][u][1] for u in batch.user_ids.tolist())
        users = parameters[f'party:{p}']['user_embeddings'][list(user_rows)].detach().clone().requires_grad_(True)
        direct = center['item_embeddings'][list(rows)].detach().clone().requires_grad_(True)
        query = center['item_embeddings'][list(rows)].detach().clone().requires_grad_(True)
        z, available, _ = _fuse(slot, query, center)
        reply = z.detach().clone().requires_grad_(True)
        loss, _ = group_loss_pri_none_from_fused(reply, available, batch.labels, batch.group_offsets,
            user_embeddings=users, item_embeddings=direct, **{n: center[n] for n in _HEAD})
        m, N = batch.loss_coefficient
        local = torch.autograd.grad(loss * (m / N), (reply, users, direct, *(center[n] for n in _HEAD)))
        gz, gu, gi, *head_grads = local
        _record_contribution(records, key('local_reply', p, p, 'g_z'), _geometry(rows, (gz,)))
        _record_contribution(records, key('direct_item', p, 'center', 'item_rows'), _geometry(rows, (gi,)))
        _accumulate_gradients(gradients, f'party:{p}', 'user_embeddings',
                              _scatter(parameters[f'party:{p}']['user_embeddings'], user_rows, gu))
        _accumulate_gradients(gradients, 'center', 'item_embeddings', _scatter(center['item_embeddings'], rows, gi))
        for name, grad in zip(_HEAD, head_grads):
            _accumulate_gradients(gradients, 'center', name, grad)
        source_ids = pools[p]
        fusion_grads = torch.autograd.grad(z, (query, *(center[n] for n in _FUSION),
            *(leaves[s][1] for s in source_ids)), grad_outputs=gz)
        gq, *rest = fusion_grads
        _record_contribution(records, key('query_item', 'center', p, 'item_rows'), _geometry(rows, (gq,)))
        _record_contribution(records, key('center_vjp', 'center', p, 'fusion'), _geometry(rows, tuple(rest[:4])))
        _accumulate_gradients(gradients, 'center', 'item_embeddings', _scatter(center['item_embeddings'], rows, gq))
        for name, grad in zip(_FUSION, rest[:4]):
            _accumulate_gradients(gradients, 'center', name, grad)
        for s, grad in zip(source_ids, rest[4:]):
            _record_contribution(records, key('receiver_source', p, s, 'g_H'), _geometry(source_rows[s], (grad,)))
            incident[s].append((p, grad))
    # Require all receiver incidents before any source backward; zero entries
    # remain present and are not confused with a missing message.
    for s in sources:
        expected_receivers = tuple(p for p in packed if s in pools[p])
        if tuple(p for p, _ in incident[s]) != expected_receivers:
            raise ValueError('source VJP requires the complete sorted receiver set')
    early = {k: v for k, v in expected.items() if k[1] not in ('source_vjp', 'shared_query')}
    _validate_contributions(early, records)
    for s in sources:
        summed = incident[s][0][1].clone()
        for _, grad in incident[s][1:]:
            summed = summed + grad
        p = parameters[f'party:{s}']
        local = torch.autograd.grad(pubs[s][1], (*(p[n] for n in _SOURCE), center['queries']), grad_outputs=summed)
        _record_contribution(records, key('source_vjp', s, s, 'local_encoder'), _geometry(source_rows[s], tuple(local[:4])))
        _record_contribution(records, key('shared_query', s, 'center', 'queries'), _geometry((), (local[4],)))
        for name, grad in zip(_SOURCE, local[:4]):
            _accumulate_gradients(gradients, f'party:{s}', name, grad)
        _accumulate_gradients(gradients, 'center', 'queries', local[4])
    _validate_contributions(expected, records)
    for o, group in parameters.items():
        if set(gradients[o]) != set(group):
            raise ValueError('incomplete gradient package')
        for name, value in gradients[o].items():
            if value is not None:
                _floating(value, cfg['dtype'])
                if value.shape != group[name].shape:
                    raise ValueError('gradient shape mismatch')
    # Canonical ordering follows the predeclared expected set, not arrival order.
    digest = _hash('thesisfedrec.joint_contributions', tuple(expected.items()))
    completed = _hash('thesisfedrec.joint_contributions', tuple((k, dict(records)[k]) for k in expected))
    return gradients, step_id, request_id, digest, completed, request, step


def _receipt_counters(receipt, parameters, gradients, generation, versions, clocks):
    if type(receipt) is not OptimizerStepReceipt:
        raise ValueError('apply did not return an OptimizerStepReceipt')
    active = tuple(o for o in sorted(parameters) if any(g is not None for g in gradients[o].values()))
    next_versions = {o: versions[o] + (o in active) for o in sorted(parameters)}
    next_clocks = {o: {n: clocks[o][n] + (gradients[o][n] is not None) for n in sorted(parameters[o])}
                   for o in sorted(parameters)}
    if (receipt.generation_before != generation or receipt.generation_after != generation + 1
            or receipt.stepped_owner_ids != active
            or receipt.owner_update_versions != tuple(next_versions.items())
            or receipt.parameter_steps != tuple((o, n, v) for o, group in next_clocks.items() for n, v in group.items())):
        raise ValueError('optimizer receipt counters do not match submitted gradient activity')
    return next_versions, next_clocks


def fit_doi4_pri_none(
    *, rounds: tuple[RoundDrainPlan, ...],
    initial_values: dict[str, dict[str, torch.Tensor]],
    user_ids_by_party: dict[int, tuple[int, ...]],
    adapters: dict[int, BinaryInputAdapter], feature_runtimes: dict[int, FeatureRuntime],
    input_bindings: dict[int, tuple[str, str, int, str]],
    allowed_sources_by_receiver: dict[int, tuple[int, ...]],
    step_source_pools: dict[tuple[int, int], dict[int, tuple[int, ...]]],
    inference_source_pools: dict[int, tuple[int, ...]], inference_online_party_ids: tuple[int, ...],
    context_id: str, initialization_id: str, profile_id: str,
    learning_rate: float, betas: tuple[float, float], epsilon: float,
    validation_candidates: EvaluationCandidateSet, k_values: tuple[int, ...],
) -> Doi4PriNoneFit:
    """Fit all supplied rounds once, fixed VALID only; return last-round state.

    Prepared trace provenance, map/table associations and permissions are trusted
    outer-coordinator obligations. Binding strings do not prove these claims.
    """
    for name, value in [('initialization_id', initialization_id), ('profile_id', profile_id)]:
        if type(value) is not str:
            raise TypeError(f'{name} must be exact str')
        if not value.strip():
            raise ValueError(f'{name} must be explicit and nonblank')
    cfg, pools, permission, candidates, ks = _prepare_config(
        rounds, initial_values, user_ids_by_party, adapters, feature_runtimes, input_bindings,
        allowed_sources_by_receiver, step_source_pools, inference_source_pools,
        inference_online_party_ids, context_id, validation_candidates, k_values)
    parameters = {o: {n: torch.nn.Parameter(t.detach().clone(), requires_grad=True)
                       for n, t in group.items()} for o, group in initial_values.items()}
    initial_digest = _values_digest(parameters)
    valid_digest = _candidate_digest(candidates)
    schedule = dict(context_id=context_id, initialization_id=initialization_id, profile_id=profile_id,
        initial_values_digest=initial_digest, plans=tuple((p.round_index, p.plan_id, p.trace_id) for p in rounds),
        step_source_pools=tuple((key, tuple(value.items())) for key, value in sorted(pools.items())),
        permissions=tuple(permission.items()), routing=tuple(cfg['user_ids_by_party'].items()),
        input_bindings=tuple(cfg['input_bindings'].items()), inference_pools=tuple(cfg['inference_source_pools'].items()),
        inference_online=cfg['inference_online_party_ids'], learning_rate=learning_rate, betas=betas,
        epsilon=epsilon, weight_decay=0.0, validation_digest=valid_digest, k_values=ks,
        warmup='none', pri='none', terminal_rule='last_supplied_round')
    owners = prepare_optimizer_owners(parameters, learning_rate=learning_rate, betas=betas,
                                     epsilon=epsilon, weight_decay=0.0)
    try:
        scheduled = _hash('thesisfedrec.joint_scheduled_end', schedule)
        versions = dict.fromkeys(sorted(parameters), 0)
        clocks = {o: dict.fromkeys(sorted(group), 0) for o, group in sorted(parameters.items())}
        steps, round_records = [], []
        generation = 0
        for plan in rounds:
            before = generation
            cursor = start_round_drain(plan)
            for step_index in range(plan.step_count):
                grads, sid, rid, expected, completed, request, step = _joint_gradients(
                    cfg, parameters, plan, cursor, pools[(plan.round_index, step_index)], generation, versions, clocks)
                receipt = apply_optimizer_gradients(owners, grads, expected_generation=generation)
                versions, clocks = _receipt_counters(receipt, parameters, grads, generation, versions, clocks)
                cursor = advance_round_drain_cursor(plan, cursor, completed_step_index=step_index)
                generation = receipt.generation_after
                steps.append(StepReceipt(step_id=sid, round_index=plan.round_index, step_index=step_index,
                    plan_id=plan.plan_id, trace_id=plan.trace_id, request_id=rid,
                    expected_contribution_digest=expected, completed_contribution_digest=completed,
                    receiver_ids=request.receiver_party_ids, source_ids=request.source_party_ids,
                    positive_group_counts=tuple((b.party_id, len(b.positive_ids)) for b in step.batches),
                    round_positive_count=plan.round_positive_count, optimizer_receipt=receipt, cursor_after=cursor))
            _check_inputs(cfg)
            metrics = evaluate_candidates(candidates, lambda u, i: _score_values(cfg, parameters, u, i), k_values=ks)
            round_records.append(RoundReceipt(round_index=plan.round_index, plan_id=plan.plan_id,
                trace_id=plan.trace_id, expected_step_count=plan.step_count,
                completed_step_count=cursor.next_step_index, round_positive_count=plan.round_positive_count,
                skipped=plan.skipped, generation_before=before, generation_after=generation,
                validation_metrics=tuple(sorted(metrics.items())), validation_candidate_digest=valid_digest))
        values = {o: {n: t.detach().clone() for n, t in group.items()} for o, group in parameters.items()}
        final_digest = _values_digest(values)
        binding = dict(scheduled_end_id=scheduled, final_parameter_digest=final_digest,
            routing=tuple(cfg['user_ids_by_party'].items()), input_bindings=tuple(cfg['input_bindings'].items()),
            steps=tuple(asdict(s) for s in steps), rounds=tuple(asdict(r) for r in round_records))
        terminal_id = _hash('thesisfedrec.joint_terminal', binding)
        fit = object.__new__(Doi4PriNoneFit)
        fit._values, fit._config = values, cfg
        fit._terminal_id, fit._scheduled_end_id = terminal_id, scheduled
        fit._initialization_id, fit._profile_id = initialization_id, profile_id
        fit._steps, fit._rounds = tuple(steps), tuple(round_records)
        fit._bindings = dict(initial_values_digest=initial_digest, final_parameter_digest=final_digest,
            plan_ids=tuple(p.plan_id for p in rounds), trace_ids=tuple(p.trace_id for p in rounds),
            user_ids_by_party=dict(cfg['user_ids_by_party']), input_bindings=dict(cfg['input_bindings']),
            inference_source_pools=dict(cfg['inference_source_pools']),
            inference_online_party_ids=cfg['inference_online_party_ids'],
            validation_candidate_digest=valid_digest, k_values=ks, terminal_content_binding=binding)
        fit._handle = _bind_terminal_evaluation(terminal_id=terminal_id,
            score_function=lambda u, i: _score_values(cfg, values, u, i),
            validate_candidates=lambda c: _candidate_preflight(cfg, c), k_values=ks)
    except BaseException:
        try:
            close_optimizer_owners(owners)
        except BaseException:
            pass  # Do not replace the primary failure with cleanup failure.
        raise
    close_optimizer_owners(owners)
    return fit
