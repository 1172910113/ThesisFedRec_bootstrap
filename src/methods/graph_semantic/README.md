# Topic 1.2 U4-K0 synthetic numerical kernels

This package implements only the bounded synthetic kernel extraction from
Library v12, `1.2-M2-v1-u4-r1`, §§18.1–18.5 and 18.7–18.9. It is not completed
U4, G3, U4P/E-P0, full PKGRec, a privacy mechanism, or a runnable M2 experiment.
The parallel-wave task card narrows the larger design: no basis construction,
production GraphView adaptation, stage orchestration, or message services.

## Inputs and ownership

All numerical execution is CPU, using explicit float32 or float64 tensors.
The tested environment is Python 3.11; Python 3.10 compatibility is untested.
No data preprocessing, ID mapping, split, negative sampling, candidate, metric,
seed, existing FedGMF behavior, or experiment configuration changes here.

`prepare_path_factors(local_item_ids, incidence, *, num_items, dtype)` accepts:

- `local_item_ids`: nonnegative CPU long `[m]`, an injective local-row-to-public
  item vector in any order, with public IDs below `num_items`;
- `incidence`: nonnegative CPU long `[edges,3]`, columns `(local item row,
  canonical relation ID, local auxiliary ID)`; numerical aux IDs can be sparse;
- an explicit positive public catalogue size and float32/float64 model dtype.

The local auxiliary key is `(relation ID, local auxiliary ID)`. Binary triples
are deduplicated **before** degree calculation. Relation-separated columns are
preserved; four-hop paths may join different relations through an item.
Empty graphs/support and mapped isolated rows are valid. This boundary starts
**after** schema/direction canonicalization. It accepts no DatasetBundle,
GraphPartition, runtime, inverse declarations, raw identities, or real schema.
A future adapter must validate those semantics separately, never guess them.

`PathFactors` owns detached coordinate tensors and an item lookup sorted over
local support only. Tensor fields of the frozen dataclasses remain mutable;
these are numerical intermediates, not a delivery/security boundary. Keep them
fixed during use. Rebuild factors and F after any graph/input snapshot change.
Public E/w/b/B are detached shared references, not full per-party clones: the
caller must preserve their contents for the entire kernel/snapshot lifetime.
No automatic version checking or revocation is implemented.

## Sparse paths and source

`apply_path(paths, Y, *, steps=1)` accepts `[m,k]` and applies
`S(S.T Y) - h[:,None] Y`, where `S = D_I^-1/2 R D_A^-1/2` and
`h_i = sum_a S_ia²`. `steps=2` composes this same ungated operator twice.
Two-hop self-action is removed; four-hop return diagonals remain. There are
no extra self-loops, epsilon, row renormalization, or simple-path restrictions.
Zero-degree inverse square roots are zero. Degrees and normalization are built
in float64; S coefficients are then cast to model dtype, and h is derived from
that same cast S. Float32 cancellation follows ordinary floating-point error.

Only coordinate scatter/reductions implement propagation. There is no dense
R, M, M², item-pair adjacency, or one-hot selection matrix L. Validity is the
local structural predicate “some deduplicated relation-aux column links at
least two distinct items”; zero projected T does not invalidate such a graph.

`prepare_source_kernel(paths, E, w, b, B)` requires finite same-dtype
`E[n,d]`, `w[d]`, scalar `b`, and explicit `B[n,p]`, `1 <= p <= n`.
It checks `B.T @ B` against identity using `atol=rtol=1e-10` in float64 and
`1e-5` in float32. It does not construct B or implement §18.6 policy, QR,
rank selection, or basis lifecycle. The synthetic rank-two example is not a
new profile. Orthonormality validation and sketches have rank-dependent p²
workspace; this is distinct from constructing item-pair graph operators.

The prepared source stores only `F2 = Apply(E[local_item_ids])` and
`F4 = Apply(F2)` on local rows. Given private user table `P[local_users,d]`,
caller-owned `a[2]`, and explicit aligned `user_rows[batch]`/`item_ids[batch]`:

- `source_logits(source, P, a, user_rows, item_ids)` uses
  `g=sigmoid(a)`, `D=g2 F2+g4 F4`, and `w dot (P_u * (E_i+D_i)) + b`;
- `source_loss(..., labels)` uses stable mean BCE-with-logits for a nonempty
  binary labeled batch. An empty labeled batch is explicitly rejected.

Batch `user_rows` index the supplied private table; no assumption equates
user identity and party identity. Unmapped public items use E alone. Scoring
gathers only batch E rows and matching local residuals; no full X is stored.
Only P/a retain source gradients. E/w/b/B/topology are frozen without changing
caller tensors or their `requires_grad` flags.

`make_operator_sketch(source, a)` returns differentiable `[2,p,p]` in fixed
channel order `[2,4]`: `g2 B_local.T Apply(B_local)` and
`g4 B_local.T Apply(Apply(B_local))`. g4 is independent of g2². This consumes
the **same caller-owned a** as source scoring; P is not a sketch input.
`detached_operator_snapshot(source, final_a)` freshly recomputes T, then
detaches. There is no initialization-time T cache or release service.

## Frozen K1 receiver

`freeze_source_snapshot(source, final_a)` recomputes and copies detached D
only on local rows, retaining shared public references. It stores no private
a or P. `prepare_receiver(snapshot, P, bank)` requires an explicit finite,
detached `[2,p,p]` K1 bank (attached/autograd banks are rejected). It copies
small bank/private P/local D buffers and freezes
`Z = B.T E + B_local.T D_local`; it never builds full X. The bank is supplied
by the caller; no donor, collection, aggregation, routing, or provenance
lookup is implicit.

`receiver_tokens(receiver, item_ids)` returns `[batch,2,d]`, with each token
`B_i bank[channel] Z`. `receiver_logits(receiver, beta, user_rows, item_ids)`
uses `X_i + beta2 token2 + beta4 token4`; `receiver_loss(..., labels)` is
mean labeled BCE. beta is an unconstrained caller-owned `[2]`; the coordinator
or test initializes it to zero. Only beta has a gradient. P/a/D/Z/E/w/b/B and
bank remain frozen, without mutating source tensors or their gradient flags.
No graph is required to consume a nonzero external bank. A zero bank gives
exact local fallback for any finite beta and zero beta data gradients.

## Numerical evidence and remaining dependencies

`tests/test_graph_semantic_u4_kernel.py` uses independent tiny dense NumPy
oracles, analytical gradients, autograd, and float64 central differences
(step 1e-6, absolute tolerance 1e-7), plus float32 comparisons at 1e-5.
It checks §18.7 source/receiver values, independent channels, final-a refresh,
fixed-a control, graph/mapping invariances, relation separation, local degree
changes, projection loss, input rejection, stable batch BCE and freeze rules.
Test-owned Adam uses betas=(0.9,0.999), eps=1e-8, weight_decay=0 and
amsgrad=False; lr=0.01 is a test fixture, not a development profile.

The bounded allocation probe exercises source/sketch/receiver forward and
backward for n=8192, m=2048, q=512, edges=4096, d=8, p=4, batch=32 in float32.
It reports actual retained numeric storage, observed tensor shapes/sizes and
process peak RSS including imports/other tests. Shape checks exclude dense
incidence and item-pair outputs on the exercised path; source inspection also
checks production allocation/product sites. The trace is not a live allocator
peak or a proof of unexercised paths. Buffer totals exclude Python overhead,
allocator workspace, setup inputs, optimizer state and other parties. No
performance thresholds or formal learning/resource qualification are claimed.

Validate this unit with the new focused tests and `tests/test_fedgmf.py`, using
external task-specific basetemp/cache/log paths under THESISFEDREC_DATA_ROOT.
The parallel wave defers the default full suite to the combined candidate
gate; scoped results do not claim a full-suite pass.

Production schema/direction adaptation, basis policy/versioning, trusted
snapshot identity, public graph delivery, optimizers and stage/resume/sampling
orchestration, collection/bank aggregation, serialization, evaluation
integration and privacy mechanisms remain separately scoped dependencies.
No registry, cross-method import, private trainer, or frozen Phase 3 change is
introduced. Learning effect, cross-party utility, privacy and real-data
scalability are unqualified by this synthetic numerical unit.
