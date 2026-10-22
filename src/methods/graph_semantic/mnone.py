"""Unprotected M-none source contributions and pure K1 arithmetic.

These values have no roster, replay, version, transport or privacy semantics.
Record buffers are owned but remain mutable; operations validate them again.
"""

from dataclasses import dataclass

import torch

from .path_operator import PathFactors
from .u4_kernel import SourceKernel, detached_operator_snapshot


_MAX_MASS = 2**53 - 1


def _p(p: int) -> None:
    if type(p) is not int:
        raise TypeError("p must be an exact built-in integer")
    if p < 1 or 2 * p * p + 1 > 2**63 - 1:
        raise ValueError("p must be positive with 2*p*p+1 in int64 index range")


def _tensor(value: torch.Tensor, dtype: torch.dtype, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cpu" or value.layout != torch.strided or value.dtype != dtype:
        raise TypeError(f"{name} must be dense strided CPU {dtype}")


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def _pair(values: torch.Tensor, mass: torch.Tensor, *, contribution: bool) -> int:
    dtype = torch.float32 if contribution else torch.float64
    for name, value in (("values", values), ("mass", mass)):
        _tensor(value, dtype, name)
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError(f"{name} must be detached")
        _finite(value, name)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[1] != values.shape[2]:
        raise ValueError("values must have shape [2,p,p]")
    p = values.shape[1]
    _p(p)
    if mass.shape != ():
        raise ValueError("mass must be a scalar")
    count = mass.item()
    if contribution:
        if count not in (0, 1):
            raise ValueError("contribution mass must be exactly 0 or 1")
    elif count < 0 or count > _MAX_MASS or count != int(count):
        raise ValueError("sum mass must be an exact integer in 0..2**53-1")
    if count == 0 and torch.count_nonzero(values).item():
        raise ValueError("zero mass requires zero values")
    return p


@dataclass(frozen=True, slots=True, kw_only=True)
class UnprotectedMNoneContribution:
    values: torch.Tensor
    mass: torch.Tensor

    def __post_init__(self) -> None:
        _pair(self.values, self.mass, contribution=True)
        object.__setattr__(self, "values", self.values.detach().clone())
        object.__setattr__(self, "mass", self.mass.detach().clone())


@dataclass(frozen=True, slots=True, kw_only=True)
class K1Sums:
    values: torch.Tensor
    mass: torch.Tensor

    def __post_init__(self) -> None:
        _pair(self.values, self.mass, contribution=False)
        object.__setattr__(self, "values", self.values.detach().clone())
        object.__setattr__(self, "mass", self.mass.detach().clone())


def _shape(value: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")


def _indices(value: torch.Tensor, length: int, upper: int, name: str) -> None:
    _tensor(value, torch.long, name)
    _shape(value, (length,), name)
    if torch.any(value < 0).item() or torch.any(value >= upper).item():
        raise ValueError(f"{name} is out of range")


def _source(source: SourceKernel, final_a: torch.Tensor) -> int:
    if type(source) is not SourceKernel or type(source.paths) is not PathFactors:
        raise TypeError("source must be a prepared SourceKernel with PathFactors")
    floats = (
        ("item_embeddings", source.item_embeddings), ("weight", source.weight),
        ("bias", source.bias), ("basis", source.basis), ("f2", source.f2),
        ("f4", source.f4), ("weights", source.paths.weights),
        ("h", source.paths.h), ("final_a", final_a),
    )
    for name, value in floats:
        _tensor(value, torch.float32, name)
        _finite(value, name)
    if source.item_embeddings.ndim != 2 or min(source.item_embeddings.shape) < 1:
        raise ValueError("item_embeddings must have positive shape [n,d]")
    n, d = source.item_embeddings.shape
    if source.basis.ndim != 2 or source.basis.shape[0] != n:
        raise ValueError("basis must have shape [n,p]")
    p = source.basis.shape[1]
    _p(p)
    if p > n:
        raise ValueError("source basis p must not exceed n")
    paths = source.paths
    if type(paths.num_items) is not int or type(paths.auxiliary_count) is not int:
        raise TypeError("path counts must be exact built-in integers")
    if paths.num_items != n or paths.auxiliary_count < 0:
        raise ValueError("path counts must match source dimensions")
    if type(paths.structurally_valid) is not bool:
        raise TypeError("structurally_valid must be a bool")
    _tensor(paths.local_item_ids, torch.long, "local_item_ids")
    if paths.local_item_ids.ndim != 1 or paths.weights.ndim != 1:
        raise ValueError("local_item_ids and weights must be vectors")
    m, e = paths.local_item_ids.numel(), paths.weights.numel()
    for name, value, length, upper in (
        ("local_item_ids", paths.local_item_ids, m, n),
        ("sorted_item_ids", paths.sorted_item_ids, m, n),
        ("sorted_local_rows", paths.sorted_local_rows, m, m),
        ("item_rows", paths.item_rows, e, m),
        ("auxiliary_columns", paths.auxiliary_columns, e, paths.auxiliary_count),
    ):
        _indices(value, length, upper, name)
    for name, value, shape in (
        ("weight", source.weight, (d,)), ("bias", source.bias, ()),
        ("f2", source.f2, (m, d)), ("f4", source.f4, (m, d)),
        ("h", paths.h, (m,)), ("final_a", final_a, (2,)),
    ):
        _shape(value, shape, name)
    return p


def prepare_mnone_contribution(
    source: SourceKernel, final_a: torch.Tensor,
) -> UnprotectedMNoneContribution:
    """Recompute one final detached T; mass uses only existing structural validity.

    The caller guarantees the prepared source's origin and stable dependencies.
    final_a may retain gradients; this call neither trains nor certifies finality.
    """
    p = _source(source, final_a)
    try:
        values = detached_operator_snapshot(source, final_a)
    except ValueError as error:
        # Inputs were checked above. The accepted second Apply uses this exact
        # error when the first Apply has produced a nonfinite intermediate.
        if str(error) == "values must be finite CPU torch.float32 with strided layout":
            raise ArithmeticError("computed path intermediate is nonfinite") from error
        raise
    _tensor(values, torch.float32, "computed T")
    _shape(values, (2, p, p), "computed T")
    if not torch.isfinite(values).all().item():
        raise ArithmeticError("computed T is nonfinite")
    valid = source.paths.structurally_valid
    return UnprotectedMNoneContribution(
        values=values if valid else torch.zeros_like(values),
        mass=torch.tensor(1.0 if valid else 0.0, dtype=torch.float32),
    )


def empty_k1_sums(*, p: int) -> K1Sums:
    _p(p)
    return K1Sums(
        values=torch.zeros((2, p, p), dtype=torch.float64),
        mass=torch.zeros((), dtype=torch.float64),
    )


def add_k1_contribution(
    sums: K1Sums, contribution: UnprotectedMNoneContribution,
) -> K1Sums:
    if type(sums) is not K1Sums or type(contribution) is not UnprotectedMNoneContribution:
        raise TypeError("expected K1Sums and UnprotectedMNoneContribution")
    p = _pair(sums.values, sums.mass, contribution=False)
    if _pair(contribution.values, contribution.mass, contribution=True) != p:
        raise ValueError("contribution p must match sums p")
    if int(sums.mass.item()) + int(contribution.mass.item()) > _MAX_MASS:
        raise ArithmeticError("sum mass exceeds 2**53-1")
    values = sums.values + contribution.values.to(torch.float64)
    mass = sums.mass + contribution.mass.to(torch.float64)
    if not torch.isfinite(values).all().item() or not torch.isfinite(mass).item():
        raise ArithmeticError("computed sum is nonfinite")
    return K1Sums(values=values, mass=mass)


def finalize_k1_bank(sums: K1Sums) -> torch.Tensor:
    if type(sums) is not K1Sums:
        raise TypeError("expected K1Sums")
    _pair(sums.values, sums.mass, contribution=False)
    if sums.mass.item() == 0:
        return torch.zeros_like(sums.values, dtype=torch.float32)
    mean = sums.values / sums.mass
    if not torch.isfinite(mean).all().item():
        raise ArithmeticError("computed mean is nonfinite")
    bank = mean.to(torch.float32)
    if not torch.isfinite(bank).all().item():
        raise ArithmeticError("float32 bank is nonfinite")
    return bank
