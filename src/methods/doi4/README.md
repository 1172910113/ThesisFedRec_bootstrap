# DOI4 compatible readout T: synthetic numerical slice

`compatible_readout(encoded_tokens, valid_unit_mask, queries, w_key, w_value)`
is the K3 readout from Library v13, `topic11.center_joint.v1.1`, subject to
L1's method/public execution ownership and L7's degeneracy rules. This
parallel L1-K slice implements only numerical T on already encoded tensors.
It does not complete U4 or any original U unit, bypass U1–U3 integration
prerequisites, or establish semantic alignment, recommendation benefit or
novelty. No scientific formula, experiment assumption or protocol changes.

Import from `src.methods.doi4` or `src.methods.doi4.encoders`. Inputs:

| Tensor | Shape | Meaning |
|---|---|---|
| `encoded_tokens` | `[E,P,d]` | Already encoded local information units |
| `valid_unit_mask` | `[E,P]` | Boolean; True means a valid unit |
| `queries` | `[K,h]` | Caller-owned readout queries |
| `w_key`, `w_value` | `[h,d]` | Caller-owned key/value maps |

The tuple result is `(H, available, A)`, with shapes `[E,K,h]`, `[E]`
(Boolean) and `[E,K,P]`. Here K means compatible slots, not a participation
count. With `key_ep = W_key X_ep` and `value_ep = W_value X_ep`,

```text
A_ekp = softmax over valid p of (B_k dot key_ep / sqrt(h))
H_ek  = sum_p A_ekp value_ep
```

Masked units never enter the denominator and have exactly zero weight and
input gradient. All-masked entities and P=0 return absence and zero H/A;
softmax is called only on entities with a valid unit. E=0 is also supported.
For exactly one valid unit, every slot returns the same mapped value and
query/key derivatives are zero, including when the unit is padded among
masked entries. The autograd connection is retained so those derivatives
can be tested. Empty/absent outputs also retain zero-derivative connections.

The supported boundary is dense strided CPU tensors: finite float32 or
float64 floating inputs of the same dtype and a Boolean CPU mask. E/P may
be zero; d/h/K must be positive. Non-contiguous inputs are supported.
Malformed types/shapes, incompatible dtype/device, and nonfinite input
values (including masked values) fail explicitly. Masked finite payloads
are zeroed before projection to isolate them even at large magnitudes.
Nonfinite projected keys/values, logits or output caused by arithmetic
overflow raise `ValueError`; there is no clipping, dtype promotion or
silent coercion. CPU autocast is locally disabled to preserve input dtype.
GPU and other dtypes are unsupported and have no validation claim.

The function neither mutates inputs nor creates, initializes, clones or
detaches parameters; it consumes no RNG. Gradients flow to the caller's
encoded tokens, queries and both maps, and shared queries accumulate
contributions through ordinary autograd. It adds no query residual, bias,
slot-specific value map, normalization or dropout. The caller retains all
parameter ownership and initialization decisions.

Tests use fixed CPU tensors, an independent Python scalar/list oracle,
the analytic L7 two-unit example, float64 gradcheck and finite differences,
singleton/empty/mixed-mask cases, joint token/mask permutations, masked
payload perturbations, input/RNG preservation and shared-query gradient
accumulation. Run `tests/test_doi4_compatible_readout.py` and unchanged
`tests/test_fedgmf.py` with external evidence/basetemp paths under the data
root. Focused validation is not a default full-suite result.

Raw FeatureViews, encoders/unitization, private pooling, F, scoring/loss,
source routing, parameter initialization policy, optimizers, execution,
messages/caches, persistence, evaluation, baselines and experiments remain
outside this slice. Future integration must use shared infrastructure.

## Center fusion F: T11-F-K0 synthetic numerical slice

`center_fusion(compatible_tokens, valid_token_mask, item_embeddings,
w_query, w_key, w_value, w_output)` is now exported through
`src.methods.doi4` and defined in `src.methods.doi4.fusion`. It implements
only K4's numerical query-to-set calculation from Topic 1.1 design v13,
`topic11.center_joint.v1.1`. The earlier T-only slice and its scope above
remain unchanged; this separate unit adds numerical F.

| Tensor | Shape | Meaning |
|---|---|---|
| `compatible_tokens` | `[E,L,h]` | Caller-packed authorized source-slot tokens |
| `valid_token_mask` | `[E,L]` | Boolean; True means valid |
| `item_embeddings` | `[E,d]` | Caller-owned shared item representations |
| `w_query` | `[h,d]` | Caller-owned query projection |
| `w_key`, `w_value`, `w_output` | `[h,h]` | Caller-owned fusion projections |

The return tuple is `(fused, available, attention)` with shapes `[E,h]`,
Boolean `[E]` and `[E,L]`. Using row-major tensors:

```text
q      = item_embeddings @ w_query.T
keys   = compatible_tokens @ w_key.T
values = compatible_tokens @ w_value.T
A_el   = softmax over valid l of (dot(q_e, keys_el) / sqrt(h))
fused  = (sum_l A_el * values_el) @ w_output.T
```

Masked token payloads are zeroed before projection. They never enter the
attention denominator and have exactly zero token gradients. Absent rows
and L=0 produce zero fused values/attention and false availability without
an empty/all-masked softmax. E=0 is supported. The fused result retains
zero-derivative connections to all floating inputs in these cases. A single
valid token has attention one and connected zero derivatives for item/query/
key paths, with the correct token/value/output derivatives retained.

F uses the accepted T boundary: dense strided CPU float32 or float64 tensors
of one dtype, finite values even at masked entries, and a Boolean CPU mask.
E/L may be zero; d/h must be positive. Non-contiguous and non-leaf caller
tensors are supported. Shapes, types and devices are validated explicitly;
no input is coerced, cloned, detached or mutated. CPU autocast is locally
disabled to preserve dtype. Nonfinite query/key/value projections, logits,
pooled values or final output raise `ValueError`; there is no clipping or
dtype promotion. GPU/other dtypes and Python 3.10 have no validation claim.

All parameters and their initialization remain with callers. There is no
RNG use, hidden state, source-to-source attention, positional/source embedding,
bias, residual, normalization or dropout. L already describes an authorized
packed set; F does not discover sources, route messages or infer permissions.
Local scoring/private residuals, losses, optimizers, communication, caches,
exact-reuse scheduling and training execution remain outside this unit.

`tests/test_doi4_center_fusion.py` checks an independent NumPy oracle, the
frozen two-token hand example, float64 central differences for every floating
input, empty/masked/singleton cases, permutations, mutation/RNG preservation,
non-leaf composition with accepted T, the item query plus direct-score gradient,
and shared-graph versus repeated-call gradient sums. These are synthetic
functional-composition tests; they add no production reuse mechanism.
Run it alongside unchanged `tests/test_doi4_compatible_readout.py` and
`tests/test_fedgmf.py` with external evidence paths and distinct fresh sibling
TMPDIR/pytest basetemp directories. The default full suite remains a separate
gate. Numerical F correctness does not establish learned compatibility,
recommendation improvement, a joint training engine or U4 completion.

## Local recommendation and positive-group loss: PW3-B

`local_recommendation_logits` and `positive_group_bce` are exported through
`src.methods.doi4` and defined in `recommendation.py`. This separate numerical
slice adds the local K4 recommendation path around accepted T/F. Earlier
slice descriptions remain historical boundaries; their implementations and
tests are unchanged. No scientific formula or experiment protocol changes.

The score API takes eight caller-owned floating tensors and two keyword-only
Boolean availability masks:

| Argument | Shape |
|---|---|
| `user_embeddings`, `item_embeddings` | `[E,d]` |
| `fused_features`, `private_features` | `[E,h]`, `[E,g]` |
| `w_fused`, `w_private` | `[d,h]`, `[d,g]` |
| `score_weight`, `score_bias` | `[d]`, scalar `[]` |
| `fused_available`, `private_available` | `[E]` |

Unavailable feature rows are zeroed **before** projection. With these masked
features z/q, the formula is:

```text
v_tilde = item_embeddings + z @ w_fused.T + q @ w_private.T
logits = sum(user_embeddings * v_tilde * score_weight, dim=1) + score_bias
```

The result is raw logits `[E]`, with no sigmoid. With both branches absent it
reduces mathematically to the public GMF score; tests compare matched public
state/IDs. Masked feature gradients are zero and residual maps keep connected
zero paths. E=0 is an empty connected result. Private pooling/encoding and
source routing are caller dependencies, not choices made by this API.

`positive_group_bce(logits, labels, group_offsets)` returns a scalar. Logits
and labels are `[N]` of one floating dtype, with exact binary labels. Dense
CPU int64 offsets `[M+1]`, M>=1, start at 0, end at N and strictly increase.
Each nonempty group explicitly packs one positive first and then zero or
more negatives. The function neither samples nor reorders. It computes
stable BCE-with-logits, averages within each group, then gives groups equal
weight. Unequal group lengths therefore differ from a labeled-example mean.
One group and groups with no negatives are valid; empty losses are errors.
Labels/offsets are data and have no label-gradient contract. The logits path
is differentiable. The public executor must apply `m_c,t/N_r` **once outside**
this kernel; no client/source weights or active-client renormalization are
invented here. Existing FedGMF mean-BCE training semantics are unchanged.

Both APIs accept only dense strided CPU float32 or float64 tensors with one
floating dtype and finite values, including masked entries. Score d/h/g must
be positive. Non-contiguous/non-leaf caller inputs are supported without
coercion, clone/detach, mutation, hidden parameters, initialization or RNG.
CPU autocast is locally disabled to preserve dtype. Types/layouts/ranks/
shapes/devices, masks and group structure are checked explicitly. Nonfinite
arithmetic at projections, residuals, products, score reductions or loss
reductions raises `ValueError`, with no clipping or promotion. Stable loss
does not imply that every extreme finite reduction fits the input dtype.
GPU/other dtypes and Python 3.10 have no new validation claim.

Tests freeze fixtures/tolerances before execution: float64 central differences
use step `1e-6`, atol `2e-7`, rtol `1e-6`; bounded float32 oracle comparisons
use `1e-5`. Independent NumPy scores/stable logistic loss, unequal positive
groups, public GMF equivalence and a test-owned unsplit T/F/score/loss graph
check forward values and all caller gradients. The same item leaf enters the
fusion query and direct score, including multiple receivers/private branches.
Tests also cover absence/empty inputs, large finite masked perturbations,
connected zeros, shared gradients, storage/versions/RNG and invalid inputs.

This adds a numerical local recommendation path, not a learned feature
encoder, real-data DOI4 model, scheduler, joint training protocol, U4
completion or recommendation gain. Parameter/optimizer/message lifecycle,
public scheduling and encoders remain separately scoped. No production reuse
mechanism or dependency on PW3-A's new runtime is introduced. Run only the
new recommendation tests plus unchanged readout, fusion and FedGMF regressions
with external logs/JUnit and distinct fresh sibling TMPDIR/basetemp paths;
full-suite and integration/review/publication gates remain later tasks.
