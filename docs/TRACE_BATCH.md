# TRACE-BATCH-K0: group-major tensor packing

This unit provides one public function, imported directly from
`src.federation.trace_batch`; it changes no package exports, existing planner,
sampler, model, method, runner, or evaluation path.

```python
pack_party_group_batch(
    batch: PartyGroupBatch, *, dtype: torch.dtype,
) -> PackedPartyGroupBatch
```

The input is exactly the existing `PartyGroupBatch` from
`src.federation.round_drain`. There is no default label dtype or sampling
policy. `dtype` must be `torch.float32` or `torch.float64`.

## Output and ownership

`PackedPartyGroupBatch` is a frozen, slotted, keyword-only record with exactly:

| Field | Meaning |
|---|---|
| `party_id: int` | Trusted party sidecar, never substituted for a user ID |
| `positive_ids: tuple[int, ...]` | Ordered interaction-ID sidecar, not model features |
| `loss_coefficient: tuple[int, int]` | Unchanged unreduced `(m, N)`; neither converted nor applied |
| `user_ids: torch.Tensor` | CPU contiguous int64 vector of length `L` |
| `item_ids: torch.Tensor` | CPU contiguous int64 vector of length `L` |
| `labels: torch.Tensor` | CPU contiguous float32/float64 vector of length `L` |
| `group_offsets: torch.Tensor` | CPU contiguous int64 vector of length `m+1` |

Every tensor owns fresh independent storage, has no input/output `.base` view,
and has `requires_grad=False`. No tensor aliases another output or another
call's output. The immutable builtin sidecar tuples may be safely shared.
Freezing the record does **not** freeze its tensors: callers may modify their
outputs, including tensor contents or shape, without affecting the descriptor
or other deliveries. This is an in-memory data adapter, not a cryptographic
privacy boundary or a guarantee against hostile Python-internals manipulation.

## Exact packing and validation

For `m` positive groups and uniform `q` supplied negatives per group,
`L=m*(q+1)`. For each group, in the descriptor's existing order:

- repeat that group's **user** ID `q+1` times;
- emit its positive item first, followed by its negative tuple without sorting;
- emit labels `(1, 0, ..., 0)`;
- emit offsets `(0, q+1, 2*(q+1), ..., m*(q+1))`.

Example: party 41, positive interactions `(203, 101)`, users `(3, 17)`,
positive items `(2, 0)`, negative rows `((8, 7), (7, 7))`, and coefficient
`(2, 7)` yield:

```text
user_ids     [3, 3, 3, 17, 17, 17]
item_ids     [2, 8, 7, 0, 7, 7]
labels       [1, 0, 0, 1, 0, 0]
group_offsets[0, 3, 6]
```

The example's duplicate negative 7 is preserved: the batch has no replacement
flag, so the packer cannot reject duplicates on an inferred sampling policy.
Users, positive items and negatives may repeat; positive **interaction IDs**
must be unique within a batch. A current positive item must never occur in its
own negative row, even with replacement.

All input scalars and nested tuple containers must have exact builtin types;
bool, NumPy integers, subclasses, lists and coercible substitutes are rejected.
IDs and `start`/`stop` lie in `0..2**63-1`. The four aligned group tuples have
equal positive length `m`; each negative row is an exact tuple of the same
length `q`. Require `start < stop`, `stop-start == m`, and the exact coefficient
`(m, N)` with `m <= N <= 2**63-1`. Python-integer arithmetic checks both
`L <= 2**63-1` and `m+1 <= 2**63-1` before any tensor/output allocation.

Unsupported dtype/type/representation raises `TypeError`; invalid values,
structure, lengths, capacities or current-target exclusion raise `ValueError`.
Dtype validation occurs first. Error prose is descriptive, not a wire format.
The packer has no bundle and does not prove ownership, access permission or
complete all-split known-positive exclusion; the validated trace producer and
planner remain responsible for their respective checks.

`q=0` is valid and produces one positive row per group, all labels one, and
offsets `0..m`. `m=0` is explicitly invalid; no q is guessed and no fake
positive, zero loss or optimizer update is manufactured. The existing planner
does not produce empty batches. Its empty/exhausted/source-only/idle roles
remain scheduling concepts outside this adapter. Tail batches keep every row
and retain their original denominator, e.g. `(1, 7)`, rather than renormalizing.

## Caller boundary, costs and deferred work

After the caller explicitly checks a `TrainingTrace` against its source
context, the existing round-drain planner consumes its permutations and
negative tuples. Pass an actual step's nonempty batch to this packer. The
caller supplies explicit label dtype; score code receives numeric user/item
indices, while an approved loss callback receives labels and offsets. Feature
encoders must not receive labels, offsets or the trusted sidecars as features.

The group-major payload is structurally compatible with the existing
`positive_group_bce` requirements: nonempty groups, increasing offsets, each
group's first label one and all remaining labels zero. Production packing
does not import or call that method. The caller applies `m/N` exactly once
outside the group loss, not once here and again later. This unit performs no
loss, model call, backward pass, optimizer update or cursor advancement.

Validation and packing use O(m*(1+q)) time and output storage, with temporary
Python lists/tuples and the positive-ID uniqueness set counted separately.
Repeated packing performs no whole-dataset scan, context hashing, sampling,
RNG consumption or hidden cache lookup. It does not change global Python,
NumPy or Torch RNG state. This is **not** a batch-only claim for the complete
pipeline: trace context preparation scans consumed data, a nonempty q-positive
trace calls the existing sampler once (including its full known-positive
scan), and existing planner preparation scans its consumed context again.

Synthetic tests use literal integer/tensor oracles, q-zero/tail/duplicate
controls, malformed nested representations, exact int64 scalar/capacity
boundaries without huge allocation, and mutation/RNG isolation. They do not
establish training quality, production throughput, privacy, GPU readiness or
a complete execution profile. Persistence, replay/recovery, executor and
optimizer semantics, model/method integration and real training remain
separately scoped. No old sampler/planner test or formal numerical profile is
replaced by this adapter.
