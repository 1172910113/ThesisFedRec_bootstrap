# Persistent Adam owners (K0)

## Boundary and public API

`src.federation.optimizer_owners` provides a synchronous, single-process,
in-memory optimizer owner runtime. Import it directly from that submodule;
package exports and the existing FedGMF/HFL runner are unchanged.

```python
prepare_optimizer_owners(
    owner_parameters, *, learning_rate, betas, epsilon, weight_decay,
) -> OptimizerOwners

apply_optimizer_gradients(
    owners, gradients, *, expected_generation,
) -> OptimizerStepReceipt

snapshot_optimizer_owners(owners) -> OptimizerOwnersSnapshot

close_optimizer_owners(owners) -> None
```

All arguments shown are required. `OptimizerOwners` is a prepare-only handle,
not a model, a method-specific training engine, a public mutable optimizer
container, or a checkpoint. The caller supplies already-created authoritative
parameters and the explicit optimizer profile. This module does not initialize
parameters or select a seed, learning rate, or scientific experiment profile.
It does not depend on the other PW8 implementation branches.

## Parameter ownership and explicit profile

`owner_parameters` must be a nonempty exact built-in dictionary mapping
nonempty exact built-in owner strings to nonempty exact dictionaries of
nonempty exact built-in parameter names and parameters. Owner and parameter
names are opaque: they are not interpreted as party/user identities, roles,
authorization, or proof of message completeness. Owner and parameter order is
normalized lexically, independently of dictionary insertion order.

Every parameter must have exact type `torch.nn.Parameter`, be a leaf with
`requires_grad=True`, have `.grad is None`, and contain finite CPU dense-strided
`float32` or `float64` values. Parameter subclasses and implicit conversions are
not supported. Valid noncontiguous parameters and zero-element shapes are
supported. Reusing a Parameter under multiple names/owners is rejected.
Distinct nonempty parameters may not share an underlying storage, even if
their views address disjoint portions. Zero-byte buffers are not treated as
aliases merely because their pointers are zero.

Preparation copies the registration dictionaries, but retains the caller's
actual Parameter objects: it does not clone, detach, replace, or initialize
them. Later input-dictionary edits do not change the registered collection.
One Adam optimizer is created per owner and persists with those Parameter
objects over successive applies. All owners in one handle share one explicit
profile:

| Argument | Required value |
|---|---|
| `learning_rate` | Finite exact built-in `float`, strictly positive |
| `betas` | Exact two-element tuple of finite exact built-in floats, each in `[0, 1)` |
| `epsilon` | Finite exact built-in `float`, strictly positive |
| `weight_decay` | Explicit finite exact built-in `float` equal to `0.0` |

Booleans, integers, NumPy scalars, and other substitutes are not coerced into
profile floats. The optimizer is ordinary `torch.optim.Adam`, with explicit
`amsgrad=False`, `maximize=False`, `foreach=False`, `fused=False`,
`capturable=False`, `differentiable=False`, and `weight_decay=0.0`. Its `lr`,
`betas`, and `eps` are the validated caller values. No clipping, scheduler,
sparse/lazy/touched-row Adam, norm-based skipping, automatic learning-rate
fallback, or alternate owner profiles are provided.

During a handle's lifetime, the caller must not rebind parameters, change
their shape/dtype/device/storage, externally update their values, or mutate
the private Adam state. Object, structure, storage, and ordinary tensor-version
checks diagnose supported misuse; they are not a hostile-Python boundary and
cannot promise to intercept every `.data` bypass or reflective mutation.
The caller must also prevent concurrent mutation and overlapping calls: this
runtime is not a lock or a concurrent optimizer.

## Complete gradient packages and update meaning

`gradients` is an exact dictionary with precisely all registered owner keys.
Each value is another exact dictionary with precisely that owner's registered
parameter keys. Every entry is an explicit exact `torch.Tensor` or `None`.
Missing and extra keys are errors, not requests for implicit zeros.

Tensor gradients must match the registered parameter's exact shape, dtype,
and CPU device, use dense strided storage, and be finite. Ordinary Tensor
nonleaf and noncontiguous gradients are supported. Parameters, Tensor
subclasses, sparse gradients, and automatic dtype/device conversion are not.
A nonempty incoming gradient may not share storage with any registered
nonempty authoritative parameter. Incoming gradients may share storage with
one another: every accepted non-None gradient is detached and cloned into an
independent private buffer before any optimizer step. Neither external
gradient values nor their autograd graph/storage are retained or modified.

Every registered parameter must enter an apply with `.grad is None`. This
interface consumes explicit gradients, not old implicit `.grad` accumulation.
A future executor can use `autograd.grad`, or finish its own complete merge
and clear `.grad` before calling. The runtime does not run forward or backward,
prove source-message completeness, infer roles from names/norms, or certify
production protocol success from a dictionary's shape.

| Entry or call | Parameters, moments, and counters |
|---|---|
| Parameter gradient `None` | Parameter and existing Adam state stay unchanged; no fake parameter step |
| Tensor gradient with numerical zeros | Ordinary Adam applies; an existing first moment can still move the parameter |
| Tensor gradient with zero elements | Still an active parameter, with its Adam counter advanced |
| Owner with all entries `None` | Its optimizer is not stepped; owner update version stays unchanged |
| Complete all-None call | Successful no-update call; handle generation advances once |

In particular, a source-only encoder can have a Tensor gradient while an
owner's private/user parameter has `None`. A shared item table is a complete
dense parameter: zero-gradient unrequested rows may move under existing
momentum. This is ordinary dense Adam, not touched-row optimization.

## Counters and public observation records

The handle's read-only properties are `generation`, `status`, and
`last_failure`. They remain readable after breakage or close. Status is exactly
`"active"`, `"broken"`, or `"closed"`.

Three counts are intentionally distinct:

1. Handle `generation` starts at zero and advances once after each successful
   complete apply, including all-None calls. `expected_generation` must be an
   exact nonnegative built-in integer equal to the current generation. This
   is not a content hash, replay-proof message identity, or durable commit.
2. Each owner update version starts at zero and advances only when that owner's
   optimizer step returns successfully. An inactive owner does not advance.
3. Each parameter has its actual Adam `state.step`, initially uninitialized
   (reported as zero, with no moments). Parameters need not share one counter.

Both generation and owner update versions lie in `[0, 2**63 - 1]`. Any apply
that would advance either applicable count beyond its bound is rejected
before any owner is stepped.

Actual initialized Adam counters are CPU floating scalar tensors, not
arbitrary-precision Python counters. They must have shape `()`, dtype
`float32` or `float64`, and finite, nonnegative, exactly integral values no
greater than `2**24 - 1`. The bound is deliberately the same for either dtype.
An active parameter at that upper bound rejects the entire package before
updates; `2**24 - 2` may advance to `2**24 - 1`. A `None` parameter may retain
the upper bound. Post-step validation requires active counters to advance
exactly once and inactive counters not to change.

`OptimizerStepReceipt` is frozen, slotted, and keyword-only, with exactly:

```python
generation_before
generation_after
stepped_owner_ids
owner_update_versions
parameter_steps
```

`stepped_owner_ids` is a tuple of owners whose optimizer steps actually
returned, in canonical order on success. `owner_update_versions` is a
lexically ordered tuple of `(owner_id, int_version)` pairs. `parameter_steps`
is an ordered tuple of
`(owner_id, tuple((parameter_name, int_step, actual_counter_dtype), ...))`
pairs. The dtype is the actual `torch.dtype`, or `None` for uninitialized
state; its accompanying step is an exact built-in integer. This reports
counter dtype without pretending the underlying state is an integer tensor.
Receipt containers are immutable and hold no parameter or moment tensors.
A receipt is not a persistent transaction journal.

`OptimizerOwnersSnapshot` is frozen, slotted, and keyword-only, with exactly:

```python
generation
learning_rate
betas
epsilon
weight_decay
owner_update_versions
parameters
parameter_steps
exp_avg
exp_avg_sq
```

The owner-version tuple uses the receipt representation. The final four fields
are new exact nested dictionaries, ordered by owner and parameter name:
`dict[owner_id][parameter_name]`. `parameters` contains owning detached copies;
`parameter_steps` contains exact built-in integers; `exp_avg` and `exp_avg_sq`
contain owning detached copies or `None` before initialization. Each numerical
buffer is independent of the runtime, authoritative parameters, incoming
gradients, all other snapshot buffers, and other snapshots. Snapshot tensor
values, headers, and dictionaries may be edited without affecting those other
objects. Frozen record fields do not make contained dictionaries/tensors
read-only. The explicit profile fields are immutable scalars/a tuple; there
is no extra generic profile record or serialization layer.

## Prevalidation, failure, and lifecycle

Before any optimizer step, apply checks handle liveness and generation, all
registered parameters and their identity/finite-value sentinels, every
initialized Adam moment and counter, the complete gradient package, and all
counter limits. It copies every accepted active gradient before entering the
update section. A bad last owner's gradient therefore causes zero calls to
the first owner's optimizer. Malformed/stale/nonfinite preflight input does
not advance generation, alter moments or parameters, or recreate optimizers.
This cannot undo damage the caller already made to authoritative state.

Owners are visited canonically. Each receives its copied `.grad` values or
`None`, is stepped at most once if active, and has its parameters/state checked
afterward. Registered `.grad` values are cleared to `None` on exit from the
update section, including failure. Only a fully successful call returns a
receipt and advances generation.

Once the first optimizer step is entered, an exception or nonfinite/invalid
post-state breaks the handle. Already performed Parameter and Adam changes
remain; there is no transactional rollback, deepcopy recovery, implicit
retry, or execution of remaining owners. The public
`OptimizerStepError(RuntimeError)` keeps the original exception in `__cause__`,
including exceptions such as `MemoryError` or `ValueError` rather than
silently swallowing them. Generation does not advance and no success receipt
is produced.

`last_failure` is `None` or exactly the immutable five-tuple:

```python
(
    attempted_owner_ids,
    completed_owner_ids,
    failure_owner_id,
    exception_type,
    message,
)
```

The first two fields are tuples of exact strings in actual call order.
Attempted is recorded immediately before entering a step; completed is
recorded immediately after a normal return, before post-validation. An owner
whose step returned but whose post-state is nonfinite therefore remains in
completed. `failure_owner_id` is the current owner string, or `None` for an
owner-external finalization failure. The last two fields are exact strings.
This diagnostic retains no Tensor, optimizer, or traceback object. A broken
handle rejects apply and snapshot; inspection of the diagnostic is not a
partial-success commit, recovery, or resume interface. Process death or
power loss need not produce any in-memory diagnostic.

Preflight rejection leaves an otherwise legal active handle usable, but does
not authorize automatic retries in a separately governed scientific run.
`close_optimizer_owners()` is explicit and idempotent. It releases the
handle's optimizer/registration references without changing caller parameter
values, storage, or `.grad`, and without restoring earlier state. Closed
handles reject apply and snapshot; diagnostic properties remain readable.
Previously returned snapshots remain independent and valid.

A new graph/fusion stage that requires fresh optimization must explicitly
prepare a new handle with its intended parameter collection. Warmup weights
may be copied by the caller before preparation. There is no implicit
round/stage reset, warmup/FedAvg optimizer migration, save/load, restart,
cross-version checkpoint contract, or persistent restoration.

## Costs, acceptance scope, and limitations

Finite-value and state checks scan registered parameter elements and existing
moments. Snapshot copies cost and allocate proportionally to all registered
parameters plus initialized moments. Active gradient buffers add storage
proportional to their full parameter shapes. Dense Adam visits complete active
parameters and retains their first/second moments; neither update nor
validation is promised to be O(requested rows) or O(batch). This is a reference
semantic implementation, not evidence of real-data scale, RSS limits,
training speed, recommendation benefit, or privacy.

The new synthetic acceptance module is `tests/test_optimizer_owners.py`. Its
short fixed float32/float64 examples use an independent scalar Adam recurrence
for parameter values, first/second moments, bias correction, epsilon placement,
and counters across successive calls. Other cases cover None versus zero,
source-only owners, dense momentum movement, noncontiguous/empty tensors,
ownership and snapshot isolation, exact profile/gradient validation, ordinary
RNG preservation, persistent object identities, bounded counters, and failures
before versus after the first physical step. Boundary-state fixtures are not
claims of executing millions of updates or a public state-loading API.
Actual executed counts, outcomes, warnings, and evidence belong in the
implementation handoff, not implied by this contract text.

K0 does not implement a loss, scheduler, model forward/backward, message/VJP
protocol, joint or staged executor, sampling/evaluation, round cursor,
checkpoint, network/filesystem operation, or RNG use. The existing runner,
FedGMF initialization/training semantics, data contracts, feature/graph
interfaces, and frozen experiment protocol are unchanged. Choosing production
optimizer profiles, combining components, a full-suite gate, real pilots, and
claims about complete thesis methods require separate authorization.
