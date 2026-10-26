"""Persistent, caller-owned dense Adam parameters; no training orchestration."""

from bisect import bisect_left
from dataclasses import dataclass
import math

import torch


_MAX_VERSION = 2**63 - 1
_MAX_STEP = 2**24 - 1
_OwnerVersions = tuple[tuple[str, int], ...]
_ParameterSteps = tuple[tuple[str, tuple[tuple[str, int, torch.dtype | None], ...]], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizerStepReceipt:
    generation_before: int
    generation_after: int
    stepped_owner_ids: tuple[str, ...]
    owner_update_versions: _OwnerVersions
    parameter_steps: _ParameterSteps


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizerOwnersSnapshot:
    """Mutable detached buffers in a frozen record, not a checkpoint format."""

    generation: int
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    owner_update_versions: _OwnerVersions
    parameters: dict[str, dict[str, torch.Tensor]]
    parameter_steps: dict[str, dict[str, int]]
    exp_avg: dict[str, dict[str, torch.Tensor | None]]
    exp_avg_sq: dict[str, dict[str, torch.Tensor | None]]


class OptimizerStepError(RuntimeError):
    """A step began but the handle broke; __cause__ retains the original error."""


class OptimizerOwners:
    """Prepare-only handle. Private attributes are not an adversarial boundary."""

    __slots__ = (
        '_parameters', '_optimizers', '_bindings', '_owner_versions',
        '_generation', '_status', '_last_failure', '_learning_rate', '_betas',
        '_epsilon', '_weight_decay',
    )

    def __init__(self) -> None:
        raise TypeError('use prepare_optimizer_owners to create this handle')

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_failure(self) -> tuple[tuple[str, ...], tuple[str, ...], str | None, str, str] | None:
        return self._last_failure


def _name(value: str) -> None:
    if type(value) is not str:
        raise TypeError('owner and parameter names must be exact builtin str')
    if not value:
        raise ValueError('owner and parameter names must be nonempty')


def _float(value: float, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f'{name} must be an exact builtin float')
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')


def _version(value: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f'{name} must be an exact builtin int')
    if not 0 <= value <= _MAX_VERSION:
        raise ValueError(f'{name} must be in 0..2**63-1')


def _tensor(value: torch.Tensor, name: str) -> None:
    if value.layout != torch.strided or value.device.type != 'cpu':
        raise TypeError(f'{name} must be dense strided CPU')
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f'{name} must use float32 or float64')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must contain only finite values')


def _parameter(value: torch.nn.Parameter) -> None:
    if type(value) is not torch.nn.Parameter:
        raise TypeError('parameters must be exact torch.nn.Parameter instances')
    _tensor(value, 'parameter')
    if not value.is_leaf or not value.requires_grad:
        raise ValueError('parameters must be leaves with requires_grad=True')
    if value.grad is not None:
        raise ValueError('parameter .grad must be None at entry')


def _storage(value: torch.Tensor) -> tuple[int, int]:
    storage = value.untyped_storage()
    return storage.data_ptr(), storage.data_ptr() + storage.nbytes()


def _binding(value: torch.nn.Parameter) -> tuple:
    # The version/storage/structure sentinel catches ordinary caller mutation,
    # not every .data bypass or native-memory write.
    return (id(value), value.shape, value.stride(), value.storage_offset(),
            value.dtype, value.device, _storage(value), value._version)


def _intervals(parameters: dict[str, dict[str, torch.nn.Parameter]]) -> list[tuple[int, int]]:
    seen: set[int] = set()
    intervals: list[tuple[int, int]] = []
    for group in parameters.values():
        for p in group.values():
            if id(p) in seen:
                raise ValueError('a Parameter may be registered only once')
            seen.add(id(p))
            lo, hi = _storage(p)
            if hi > lo:
                intervals.append((lo, hi))
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
        raise ValueError('registered parameters must not share nonempty storage')
    return intervals


def prepare_optimizer_owners(
    owner_parameters: dict[str, dict[str, torch.nn.Parameter]], *,
    learning_rate: float, betas: tuple[float, float], epsilon: float,
    weight_decay: float,
) -> OptimizerOwners:
    """Validate all inputs, retain authoritative Parameters, create one Adam/owner."""
    for value, name in ((learning_rate, 'learning_rate'), (epsilon, 'epsilon'),
                        (weight_decay, 'weight_decay')):
        _float(value, name)
    if learning_rate <= 0 or epsilon <= 0:
        raise ValueError('learning_rate and epsilon must be positive')
    if weight_decay != 0.0:
        raise ValueError('K0 supports only weight_decay=0.0')
    if type(betas) is not tuple:
        raise TypeError('betas must be an exact tuple')
    if len(betas) != 2:
        raise ValueError('betas must have two entries')
    for beta in betas:
        _float(beta, 'beta')
        if not 0 <= beta < 1:
            raise ValueError('betas must be in [0, 1)')
    if type(owner_parameters) is not dict:
        raise TypeError('owner_parameters must be an exact dict')
    if not owner_parameters:
        raise ValueError('at least one owner is required')
    for owner, group in owner_parameters.items():
        _name(owner)
        if type(group) is not dict:
            raise TypeError('each owner parameter group must be an exact dict')
        if not group:
            raise ValueError('each owner needs at least one parameter')
        for name, p in group.items():
            _name(name)
            _parameter(p)
    parameters = {owner: {name: owner_parameters[owner][name]
                          for name in sorted(owner_parameters[owner])}
                  for owner in sorted(owner_parameters)}
    _intervals(parameters)
    owners = object.__new__(OptimizerOwners)
    owners._parameters = parameters
    owners._bindings = {o: {n: _binding(p) for n, p in group.items()}
                        for o, group in parameters.items()}
    owners._optimizers = {
        o: torch.optim.Adam(list(group.values()), lr=learning_rate, betas=betas,
                            eps=epsilon, weight_decay=0.0, amsgrad=False,
                            maximize=False, foreach=False, fused=False,
                            capturable=False, differentiable=False)
        for o, group in parameters.items()
    }
    owners._learning_rate, owners._betas = learning_rate, betas
    owners._epsilon, owners._weight_decay = epsilon, weight_decay
    owners._generation = 0
    owners._owner_versions = dict.fromkeys(parameters, 0)
    owners._status, owners._last_failure = 'active', None
    return owners


def _active(owners: OptimizerOwners) -> None:
    if type(owners) is not OptimizerOwners:
        raise TypeError('owners must be an exact OptimizerOwners handle')
    if owners.status != 'active':
        raise RuntimeError(f'optimizer owners handle is {owners.status}')


def _state(optimizer: torch.optim.Adam, parameter: torch.nn.Parameter) -> tuple[int, torch.dtype | None]:
    state = optimizer.state.get(parameter)
    if not state:
        return 0, None
    if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
        raise ValueError('Adam state must contain step, exp_avg and exp_avg_sq')
    step = state['step']
    if type(step) is not torch.Tensor:
        raise TypeError('Adam step must be a plain Tensor')
    _tensor(step, 'Adam step')
    if step.shape != torch.Size([]):
        raise ValueError('Adam step must have scalar shape ()')
    counter = step.item()
    if not 0 <= counter <= _MAX_STEP or not counter.is_integer():
        raise ValueError('Adam step must be exact-integral in 0..2**24-1')
    for name in ('exp_avg', 'exp_avg_sq'):
        moment = state[name]
        if type(moment) is not torch.Tensor:
            raise TypeError(f'{name} must be a plain Tensor')
        _tensor(moment, name)
        if moment.dtype != parameter.dtype:
            raise TypeError(f'{name} dtype must match its parameter')
        if moment.shape != parameter.shape:
            raise ValueError(f'{name} shape must match its parameter')
    if bool((state['exp_avg_sq'] < 0).any()):
        raise ValueError('Adam exp_avg_sq must be nonnegative')
    return int(counter), step.dtype


def _validate(owners: OptimizerOwners) -> dict[str, dict[str, tuple[int, torch.dtype | None]]]:
    _active(owners)
    _version(owners._generation, 'generation')
    counters = {}
    for owner, group in owners._parameters.items():
        _version(owners._owner_versions[owner], 'owner_update_version')
        counters[owner] = {}
        for name, p in group.items():
            _parameter(p)
            if _binding(p) != owners._bindings[owner][name]:
                raise ValueError('registered parameter identity/structure/storage/version changed')
            counters[owner][name] = _state(owners._optimizers[owner], p)
    return counters


def _versions(owners: OptimizerOwners) -> _OwnerVersions:
    return tuple(owners._owner_versions.items())


def apply_optimizer_gradients(
    owners: OptimizerOwners,
    gradients: dict[str, dict[str, torch.Tensor | None]], *,
    expected_generation: int,
) -> OptimizerStepReceipt:
    """Prevalidate/copy the whole package, then step owners once in sorted order.

    Pre-step errors do not commit. Errors after entering any step poison the
    handle; physical partial updates remain and are not rolled back.
    """
    _active(owners)
    _version(expected_generation, 'expected_generation')
    if expected_generation != owners.generation:
        raise ValueError('stale expected_generation')
    counters = _validate(owners)
    if owners.generation == _MAX_VERSION:
        raise ValueError('generation would overflow')
    if type(gradients) is not dict:
        raise TypeError('gradients must be an exact dict')
    for owner in gradients:
        _name(owner)
    if gradients.keys() != owners._parameters.keys():
        raise ValueError('gradient owner keys must match the complete registration')
    intervals = _intervals(owners._parameters)
    starts = [lo for lo, _ in intervals]
    copied: dict[str, dict[str, torch.Tensor | None]] = {}
    for owner, group in owners._parameters.items():
        source = gradients[owner]
        if type(source) is not dict:
            raise TypeError('each gradient group must be an exact dict')
        for name in source:
            _name(name)
        if source.keys() != group.keys():
            raise ValueError('gradient parameter keys must match the complete registration')
        copied[owner] = {}
        for name, p in group.items():
            grad = source[name]
            if grad is None:
                copied[owner][name] = None
                continue
            if type(grad) is not torch.Tensor:
                raise TypeError('a gradient must be a plain Tensor or None')
            _tensor(grad, 'gradient')
            if grad.dtype != p.dtype:
                raise TypeError('gradient dtype must match its parameter')
            if grad.shape != p.shape:
                raise ValueError('gradient shape must match its parameter')
            lo, hi = _storage(grad)
            index = bisect_left(starts, hi) - 1
            if hi > lo and index >= 0 and intervals[index][1] > lo:
                raise ValueError('gradient must not alias registered parameter storage')
            if counters[owner][name][0] == _MAX_STEP:
                raise ValueError('active Adam step would exceed 2**24-1')
            copied[owner][name] = grad.detach().clone()
        if any(g is not None for g in copied[owner].values()):
            if owners._owner_versions[owner] == _MAX_VERSION:
                raise ValueError('owner_update_version would overflow')
    attempted: list[str] = []
    completed: list[str] = []
    current: str | None = None
    after = {o: dict(group) for o, group in counters.items()}
    try:
        try:
            for owner, group in owners._parameters.items():
                if all(g is None for g in copied[owner].values()):
                    continue
                current = owner
                for name, p in group.items():
                    p.grad = copied[owner][name]
                attempted.append(owner)
                owners._optimizers[owner].step()
                completed.append(owner)
                owners._owner_versions[owner] += 1
                for name, p in group.items():
                    _tensor(p, 'updated parameter')
                    # Update may change version, not structure or backing storage.
                    if _binding(p)[:-1] != owners._bindings[owner][name][:-1]:
                        raise ValueError('optimizer changed parameter identity/structure/storage')
                    after[owner][name] = _state(owners._optimizers[owner], p)
                    expected = counters[owner][name][0] + (copied[owner][name] is not None)
                    if after[owner][name][0] != expected:
                        raise ValueError('Adam counter did not advance exactly for active gradients')
        finally:
            for group in owners._parameters.values():
                for p in group.values():
                    p.grad = None
        current = None
        owners._bindings = {o: {n: _binding(p) for n, p in group.items()}
                            for o, group in owners._parameters.items()}
        receipt = OptimizerStepReceipt(
            generation_before=owners.generation, generation_after=owners.generation + 1,
            stepped_owner_ids=tuple(completed), owner_update_versions=_versions(owners),
            parameter_steps=tuple((o, tuple((n, step, dtype) for n, (step, dtype)
                                           in group.items())) for o, group in after.items()),
        )
    except BaseException as exc:
        if attempted:
            owners._status = 'broken'
            owners._last_failure = (tuple(attempted), tuple(completed), current,
                                    type(exc).__name__, str(exc))
            raise OptimizerStepError('optimizer step failed; handle is broken, no rollback') from exc
        raise
    owners._generation += 1
    return receipt


def snapshot_optimizer_owners(owners: OptimizerOwners) -> OptimizerOwnersSnapshot:
    """Validate active state and independently clone every exposed numeric buffer."""
    counters = _validate(owners)
    parameters, steps, first, second = {}, {}, {}, {}
    for owner, group in owners._parameters.items():
        parameters[owner], steps[owner], first[owner], second[owner] = {}, {}, {}, {}
        for name, p in group.items():
            parameters[owner][name] = p.detach().clone()
            steps[owner][name] = counters[owner][name][0]
            state = owners._optimizers[owner].state.get(p)
            first[owner][name] = state['exp_avg'].detach().clone() if state else None
            second[owner][name] = state['exp_avg_sq'].detach().clone() if state else None
    return OptimizerOwnersSnapshot(
        generation=owners.generation, learning_rate=owners._learning_rate,
        betas=owners._betas, epsilon=owners._epsilon, weight_decay=owners._weight_decay,
        owner_update_versions=_versions(owners), parameters=parameters, parameter_steps=steps,
        exp_avg=first, exp_avg_sq=second,
    )


def close_optimizer_owners(owners: OptimizerOwners) -> None:
    """Release handle references only; caller parameter values/storage/.grad stay intact."""
    if type(owners) is not OptimizerOwners:
        raise TypeError('owners must be an exact OptimizerOwners handle')
    if owners.status != 'closed':
        owners._parameters.clear()
        owners._optimizers.clear()
        owners._bindings.clear()
        owners._owner_versions.clear()
        owners._status = 'closed'
