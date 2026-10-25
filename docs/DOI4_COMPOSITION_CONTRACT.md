# DOI4 composition K0: pri-none

This unit adds three pure tensor compositions and immutable parameter shape
declarations. It fixes `private_branch=none` and
`partial_wiring_profile=true`. It does not implement the optional private
branch or the complete DOI4/Topic 1.1 method. Existing binary encoding,
compatible readout, center fusion, recommendation and loss kernels retain
their formulas and validation contracts.

Import directly from `src.methods.doi4.composition` and
`src.methods.doi4.parameters`; this unit adds no package exports. All actual
parameters, initialization, ownership, graph lifetime and updates remain
with the caller. No production function calls backward, `autograd.grad`,
an optimizer, sampling, logging, files, clocks or RNG.

## Tensor composition APIs

### Binary units to compatible slots

```python
encode_readout_binary_units(
    local_field_ids, binary_values, valid_unit_mask,
    *, field_embeddings, value_embeddings, queries, w_key, w_value,
) -> tuple[H, available, attention]
```

`local_field_ids`, `binary_values` and `valid_unit_mask` have shape `[E,P]`;
IDs/codes are CPU int64 and the mask is Boolean. `field_embeddings` is
`F[J,r]`, `value_embeddings` is `V[2,r]`, `queries` is `B[K,h]` and readout
key/value maps have shape `[h,r]`. The composition calls
`encode_binary_field_units` to obtain `X[E,P,r]`, then calls
`compatible_readout(X, valid_unit_mask, queries, w_key, w_value)`. For a valid
unit, `X[e,p] = F[id[e,p]] + V[code[e,p]]`.

The **same original mask** is passed to readout. A known code zero remains
observed; neither a code value nor an embedding norm determines availability.
The result is `H[E,K,h]`, Boolean `available[E]`, and
`attention[E,K,P]`. `P` comes from the input shape. Here `r` is the input
width called `d` by the standalone readout documentation; it is distinct
from final item width `d` and compatible width `h`. Both `h != r` and
`d != h` are supported.

IDs are in `[0,J)` or explicit `-1` at masked positions; padding cannot be
valid. Codes must be exact zero or one even when masked. With `J=0`, every
position must be masked and use ID `-1`. Both full tables must be finite,
including unused rows. The existing encoder gathers/adds only valid units
and returns exact zero at masked units. Readout attends only over valid
units, without residuals, bias, normalization or dropout.

### Compatible slots to local logits

```python
fuse_score_pri_none(
    compatible_tokens, valid_token_mask,
    *, user_embeddings, item_embeddings,
    w_query, w_key, w_value, w_output,
    w_fused, score_weight, score_bias,
) -> tuple[logits, fused_features, fused_available, fusion_attention]
```

The public caller supplies an already authorized, packed source-slot set
`compatible_tokens[E,L,h]` and Boolean `valid_token_mask[E,L]`. This function
does not choose sources, deduplicate requests, change order, remove zero
values or reconstruct the mask. It calls the existing `center_fusion` with
the caller's `item_embeddings[E,d]`, query map `[h,d]`, and key/value/output
maps `[h,h]`. Fusion returns `z[E,h]`, Boolean availability `[E]`, and
attention `[E,L]`. The same caller item tensor then enters the direct
recommendation path. `user_embeddings` is `[E,d]`, `w_fused` is `[d,h]`,
`score_weight` is `[d]` and `score_bias` is scalar `[]`.

The existing positive-width scoring interface is adapted using fixed
private width **g=1**: ordinary zero tensors `private_features[E,1]` and
`w_private[d,1]`, plus a false Boolean availability mask `[E]`. They are
constructed only after the corresponding scoring input type, layout,
rank, dtype, device and shape checks. They match validated inputs' dtype
and device, never require gradients, and are not Parameters, returned
outputs, persistent caches or declared owners. No RNG is consumed and no
private width argument or optional private API is added. This is a zero
interface placeholder, not a one-dimensional private model. Its logits
and existing trainable-input gradients equal those of the prior proposal's
ordinary all-zero g=r placeholder.

With the scorer's availability masking, the implemented formula is

```text
v_tilde = item_embeddings + z @ w_fused.T
logits  = sum(user_embeddings * v_tilde * score_weight, dim=1) + score_bias
```

Logits are raw `[E]` scores without sigmoid. Returned fusion values,
availability and attention are those produced by the accepted fusion
kernel. No hidden detach, clone, CPU transfer, dtype coercion or input
mutation is introduced.

### Unweighted positive-group loss

```python
fuse_score_group_loss_pri_none(
    compatible_tokens, valid_token_mask, labels, group_offsets,
    *, user_embeddings, item_embeddings,
    w_query, w_key, w_value, w_output,
    w_fused, score_weight, score_bias,
) -> tuple[loss, logits, fused_features, fused_available, fusion_attention]
```

This function composes `fuse_score_pri_none` with the existing
`positive_group_bce`. Labels are finite binary `[N]` floating tensors of
the logits' dtype. CPU int64 `group_offsets[M+1]` start at zero, end at N,
and strictly increase. Each nonempty group contains its positive first,
then zero or more negatives. The scalar loss averages stable
BCE-with-logits within each group, then averages the group means equally.
It rejects empty logits and zero groups; q=0 remains legal when each group
contains its one positive. It does not sample, reorder, divide again by
batch/example counts or create source-only losses. Any future `m/N`
weighting belongs exactly once outside this composition to the public
executor, with no active-party renormalization or extra VJP weighting here.
Labels and offsets are data; no label-gradient contract is introduced.

## Numerical and graph boundary

All floating inputs, including masked values, must be finite dense strided
CPU float32 or float64 tensors of one dtype. Integer tensors are CPU
int64 and masks are CPU Boolean. Caller-owned Parameters, non-leaf
expressions and non-contiguous tensors preserve their normal autograd
paths. Positive widths `d`, `r`, `h`, `K` are required. `E`, `P`, `J` and
`L` may be zero where the relevant accepted kernel permits them. No
clipping, epsilon repair or implicit promotion is added; arithmetic
nonfiniteness fails explicitly and accepted kernels locally disable CPU
autocast. GPU, other dtypes and Python 3.10 have no new qualification.

Calling these math functions executes their tensor paths. Source E=0,
P=0 or all-masked cases preserve connected zero derivatives to caller
tables, queries and maps. A singleton unit has connected zero query/key
derivatives. Fusion E=0, L=0 and all-masked cases return exact zero z and
attention with false availability while retaining connected zero paths;
a single valid slot has connected zero item-query/key derivatives.
Identical slots can also have degenerate query derivatives. None of
these properties warrants perturbing the accepted formula.

Scoring still executes the `w_fused` path when z is zero, so its derivative
is connected zero rather than structurally absent. User/item/head paths
follow the actual scorer. Empty inference is legal, but the group-loss
wrapper rejects empty loss inputs. There is no trainable private state.
A future public executor may choose not to call an entire structurally
absent path; that separate decision can yield `grad=None`. This module
does not turn an executed empty path into `None`.

Returned H, z, attentions and logits retain their graphs. A caller may
reuse a source H in several receiver calculations and standard autograd
sums those contributions, including shared-query contributions through
source readout. Repeated item gathers naturally accumulate gradients;
item direct-score and fusion-query contributions share the same caller
tensor. No source-VJP API, message completeness, graph handle, version,
receipt, deduplication, optimizer counter or production reuse mechanism
is implemented. The caller determines graph lifetime and update timing.

Existing kernel error types remain unchanged. For checks added by the
wrapper, wrong object/dtype raises `TypeError`; wrong rank, shape or
numeric content raises `ValueError`. Existing device error differences
are preserved. No common first-error order is promised for inputs with
several independent errors. Inputs' values, storage, versions and existing
`.grad` fields, and RNG state, are not modified by forward calls.

## Pure parameter declarations

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    owner_scope: str
    party_id: int | None
    trainable: bool


def doi4_parameter_specs(
    *, party_id, public_item_count, local_user_count, local_field_count,
    item_dim, unit_dim, compatible_dim, slot_count,
) -> tuple[ParameterSpec, ...]
```

There are no defaults. The declaration function accepts only exact builtin
integers: party ID and counts I/U/J are in `[0, 2**63-1]`, while dimensions
d/r/h/K are in `[1, 2**63-1]`. Booleans, NumPy scalars and integer subclasses
are rejected. Each shape extent and its Python-integer product must fit
signed int64 capacity; validation allocates no tensor or table. Allowing
I/U/J=0 is shape-description support, not a claim of a trainable empty loss.

The returned order is exactly the following 14 labels. All are trainable.

| Order | name | shape | owner_scope | party_id |
|---|---|---|---|---|
| 1 | item_embeddings | `[I,d]` | center | None |
| 2 | queries | `[K,h]` | center | None |
| 3 | fusion_w_query | `[h,d]` | center | None |
| 4 | fusion_w_key | `[h,h]` | center | None |
| 5 | fusion_w_value | `[h,h]` | center | None |
| 6 | fusion_w_output | `[h,h]` | center | None |
| 7 | w_fused | `[d,h]` | center | None |
| 8 | score_weight | `[d]` | center | None |
| 9 | score_bias | `[]` | center | None |
| 10 | user_embeddings | `[U,d]` | party | supplied party_id |
| 11 | field_embeddings | `[J,r]` | party | supplied party_id |
| 12 | value_embeddings | `[2,r]` | party | supplied party_id |
| 13 | readout_w_key | `[h,r]` | party | supplied party_id |
| 14 | readout_w_value | `[h,r]` | party | supplied party_id |

Direct record construction validates the same local record rules. `name`
is one of these exact builtin strings and must agree with its owner scope.
`owner_scope` is exactly the builtin string `center` or `party`.
`trainable` is exact bool True. Shape is an exact tuple of exact builtin
integer extents within int64 capacity, with the listed rank. Only the
first axis of item/user/field tables may be zero; all widths are positive.
Value embeddings have first axis exactly two. Fusion key/value/output
records must be square `[h,h]`, including when constructed directly.
Scalar bias uses `()`. Center party IDs are None; party IDs are exact
builtin integers in `[0, 2**63-1]`. Each record validates its own shape and
numel, while relationships between different declarations come from the
function's explicit common dimensions.

Records hold only immutable builtin values. `parameters.py` needs no torch
import and creates no Parameter, initializer, RNG, optimizer or allocation
state. Labels are software parameter names, not canonical feature names.
Across calls differing only in party ID, center declarations are equal;
this neither creates shared parameters nor certifies actual tensor aliases.
The public center owner creates one shared set. Local F/V and readout maps
belong to the supplied party. User count U and party ID occupy distinct
namespaces. Serving as a source does not create a user loss or gradient.
Public code must separately bind map identity/epoch to F and any optimizer
rows; the method neither reads maps nor changes field order.

No W_private is declared. Dtype, byte budgets, parameter binding,
initialization/restoration, seed, learning rate, Adam settings/state and
counters are outside this declaration API. There are no scientific default
dimensions. As an arithmetic example only, d16/r8/h8/K2/J6/U1 gives center
`16*I + 481` scalars and local `208`; this is not a pilot profile or a
resource measurement.

## Validation and scope of evidence

The authorized focused invocation comprises the new
`tests/test_doi4_composition.py` and unchanged binary-field, compatible
readout, center-fusion and recommendation test modules. The new tests are
required to cover an independent fixed NumPy/scalar forward and stable BCE
oracle; selected-entry central differences through the full chain;
multiple receiver uses of one source graph; repeated gathers and item
query/direct gradient decomposition; empty, masked, singleton and observed
zero cases; non-contiguous/non-leaf/Parameter paths and forward preservation;
14 declaration records, direct-construction checks and allocation/RNG
sentinels; and g=1 versus g=r ordinary-zero adapter equivalence. Test
fixture d=2/r=2/h=3/K=2 is a mathematical fixture, not a scientific default.

The frozen tolerances are float64 forward/loss `atol=rtol=1e-12`, float32
`atol=rtol=1e-5`, and float64 central differences with step `1e-6`,
`atol=2e-7`, `rtol=2e-6`. At least one frozen nondegenerate query/item-query
response and separate formula-degenerate controls are required. Actual
node outcomes, numerical errors, process/resource observations and warnings
belong in the external implementation evidence; this contract does not
predeclare counts or results. New tests prove bounded math composition,
not public message/version/update lifecycle integration. Historical kernel
and review evidence is reused within its original limits, not rerun or
extended by this document.

Only this contract, the two new source modules and the new composition
test file are in this unit. No old kernel, package export, README, shared
contract, runner or configuration changes are needed. NumPy-to-Torch
adaptation, entity routing/gather/scatter, source packing/permissions,
LocalFieldMap/GT/binding validation, DatasetBundle/FeatureView/runtime,
training/evaluation and persistence remain public-layer responsibilities.
Production math accepts explicit tensors, not those objects, their metadata,
hashes, records or callbacks capturing them. Dependencies stay within the
standard library, torch and the accepted same-method kernels; there is no
cross-method, public data/federation/evaluation, scheduler or training-engine
dependency.

This unit does not establish recommendation benefit, semantic alignment,
privacy, compatibility discovery, real-data/GPU/scale/profile readiness,
complete Topic 1.1 or U4 completion. It does not repair prior evidence:
PW4 A remains 43/44 with the 1016-byte item's role/value UNKNOWN; PW5 B's
historical process identities remain UNKNOWN, r1 UNRESOLVED and r2 bounded.
One local implementation commit stops for independent exact-commit review;
there is no merge, publication, next-unit or training authorization here.
