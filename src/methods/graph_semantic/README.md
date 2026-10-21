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

## T12-BASIS-K0: separately scoped public basis construction

This subsequent unit implements only Library v12 §18.6,
`1.2-M2-v1-u4-r1`. The earlier U4-K0 scope statements above describe that
accepted numerical kernel. Its signatures and implementation remain unchanged.
Basis construction is now available separately; distribution and T/bank version
enforcement are still not integrated. This unit does not complete U4 or
establish recommendation benefit.

The exact new public functions are:

```python
build_public_basis(
    item_embeddings: torch.Tensor,
    *,
    p: int,
    basis_seed: int,
    policy: str,
    public_embedding_id: str,
    catalogue_id: str,
) -> PublicBasis

validate_public_basis(
    result: PublicBasis,
    *,
    item_embeddings: torch.Tensor,
    p: int,
    basis_seed: int,
    policy: str,
    public_embedding_id: str,
    catalogue_id: str,
    expected_binding_id: str,
) -> None
```

`PublicBasis` contains `basis: torch.Tensor` and
`record: PublicBasisRecord`. Both dataclasses are frozen and slotted; B remains
mutable. Construction returns independent, contiguous CPU float32 B `[n,p]`
without retaining E or any mutable caller alias. Inputs must remain stable
throughout construction/validation; there is no concurrent-writer guarantee.

E must be finite, dense strided CPU float32/float64 `[n,d]`, with positive
n/d and `requires_grad=False`. Construction is not differentiable. Strided
views are supported and fingerprinted in logical row order. p and basis_seed
are exact built-in integers, excluding Boolean/NumPy scalar substitutes,
with `1 <= p <= n` and seed >= 0. The two caller identity strings are exact
nonblank strings, with no outer whitespace, control characters or invalid
UTF-8 surrogates. They are assertions, not independently verified provenance.
No IDs are normalized. Policies are exactly `public_E_pivot_mgs_v1` and
`public_random_mgs_v1`; neither has an implicit substitute.

### Construction policy

All construction arithmetic is float64, without centering E or consulting
client coverage, graph, training or held-out feedback. The pivot policy uses
`tau_rank = 1e-12 * max(1, ||E||_F)` and starts with empty Q. For every
unselected original feature column it recomputes the residual using two
complete modified Gram–Schmidt passes, each in current Q-column order.
If the maximum residual norm is <= tau_rank, the E phase stops. Otherwise,
among norms > tau_rank and within tau_rank of the maximum, the smallest
original feature-column index wins. Its residual is normalized, then signed
so the first largest-absolute coordinate in public item-row order is positive.
No original column is selected twice; selection stops immediately at p.

Rank completion uses a private `numpy.random.Generator(PCG64(basis_seed))`.
It draws one float64 standard-normal vector of length n at a time, performs
the same two passes, and rejects residual norm <=
`1e-12 * max(1, original_candidate_norm)`. Accepted vectors use the same
normalization/sign convention. At most `2*p` candidates are attempted in total,
including rejected candidates; drawing stops as soon as Q reaches p. If
needed, standard coordinate vectors are tried in ascending item-row order,
with the same residual/sign/rejection rules, until p columns are obtained.
Failure is an error, never an implicit rank reduction.

The random policy starts directly at completion with empty Q. E is checked
and fingerprinted, but its values/norms do not choose directions. With the same
n/p/seed its B is unchanged by finite E values or d; the binding still reflects
E content, dtype/shape, identity and catalogue. Both policies preserve global
Python, NumPy and PyTorch RNG state. QR/SVD, alternative pivots and silent
arithmetic repairs are absent. Nonfinite arithmetic is rejected.

B is converted to float32 and checked using its delivered coordinates:
finite, exact shape and `||B.T @ B - I||_F <= 1e-5 * max(1,p)`.
This check accumulates in float64; failure raises without reorthogonalization.
An unmodified delivered float32 B can be passed to the existing float32 U4-K0
source kernel, whose own existing basis checks remain in force.

### Identity format and authoritative coordinates

`PublicBasisRecord` contains exactly the fields defined by its typed dataclass:
format/version (`thesisfedrec.public_basis`, 1), policy/version (1), n/d/p,
basis_seed, construction/delivery dtype, RNG implementation, numerical
environment, rank threshold (None for the random policy), selected original
pivot indices, random attempt/acceptance counts, coordinate-fallback rows,
caller E/catalogue IDs, E dtype/fingerprint, B checksum, numerical payload bytes
and binding_id. Index choices and environment entries are immutable tuples.
Environment entries, in fixed order, record Python, NumPy, PyTorch, machine
and byte order; the task evidence additionally records NumPy build/BLAS details.

Every canonical JSON object below is encoded as UTF-8 using Python
`json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
allow_nan=False)` with no trailing newline. Tuples become JSON arrays; ordering
of pivots, fallback choices and environment entries is preserved.

1. Canonical numerical bytes are contiguous C-order little-endian coordinates.
   E uses its original float32/float64 dtype; B always uses float32. Signed zero
   bits are preserved. SHA-256 is applied to these exact bytes.
2. E's content fingerprint is SHA-256 of canonical JSON with exactly
   `format="thesisfedrec.public_embedding"`, `version=1`, `shape=[n,d]`,
   `dtype="float32"|"float64"` and `sha256=<canonical E bytes checksum>`.
3. `basis_sha256` is SHA-256 of canonical B bytes. Shape/dtype and all other
   coordinates are bound through the record. `numerical_payload_bytes=4*n*p`.
4. The binding is SHA-256 of the canonical JSON representation of **all**
   `PublicBasisRecord` dataclass fields except `binding_id` itself. This is
   precisely `dataclasses.asdict(record)` with that one key removed. The format
   is fixed/versioned and contains no arbitrary metadata, timestamps, repr,
   Python hashes or pickle.

The validator checks record types/versions/choices, explicit expected E and
configuration identities, recomputed E fingerprint, record binding, expected
binding, delivered shape/dtype/orthogonality and actual B checksum. The expected
binding must come from trusted caller bookkeeping, rather than being copied
from a received record. Validation never regenerates B. Changed B bytes,
including a sign flip, column permutation or signed-zero change, fail under
an existing record even when orthogonality is preserved.

Saved B bytes/checksum are the coordinate authority. The record captures the
construction environment but does not promise bitwise reconstruction across
numerical libraries or machines. Validation can accept those saved coordinates
with their expected binding in another environment without reconstruction.
The module has no production serializer: the payload/record round-trip test
uses ordinary test-owned files only.

### Validation, resources and remaining work

`tests/test_graph_semantic_public_basis.py` includes a separately written scalar
MGS reference, hand fixtures, actual PCG64 and controlled test-only streams,
rank/pivot/sign ties, rejection and fallback, dtype/strides, RNG/ownership,
identity/tampering/orthogonality rejection, record/payload round-trip and an
independent dense U4-K0 sketch check. Compatibility validation also runs the
unchanged U4-K0 and FedGMF tests; no default full suite is claimed.

Construction keeps float64 Q `[n,p]`, local residual/candidate/product vectors
of length n, and scalar norm/pivot bookkeeping over d columns. Float32 E may
need a float64 `[n,d]` reading copy; strided input hashing may need a contiguous
original-dtype `[n,d]` copy. Norm evaluation may also need reading workspace.
Hashing uses a contiguous buffer without a separate full Python bytes copy.
Delivery adds float32 B `[n,p]`; orthogonality checking reads it into float64
`[n,p]` and forms only `[p,p]` Gram work. No catalogue-square covariance,
projector, graph operator or per-party E copies are constructed. The required
output itself is n-by-n when p=n; this is a supported case.

The bounded observation uses strided float32 E `[4096,12]`, p=8, and reports
actual Q/E64/residual/output sizes, selected NumPy shapes, tracemalloc peak,
process peak RSS and an explicit workspace inventory. Some inventory components
have disjoint lifetimes. Tracemalloc misses untracked native/BLAS/PyTorch work;
process RSS includes imports/other tests. These observations are not performance
thresholds or formal resource qualification.

The output's numerical payload is exactly `4*n*p` bytes. Canonical serialized
record bytes are counted separately in tests/evidence. These sizes are not
measured network traffic or a claim that distribution/cache integration exists.
A future public coordinator must construct once, distribute authoritative B and
record, and enforce catalogue/E/basis/T/bank identity and lifecycle. Coordinator,
transport, cache, persistence, graph adaptation, training-stage integration,
collection and privacy mechanisms remain outside this unit.

## PW3 C: explicit authorized local GraphView adapter

This unit adds a pure adapter from one caller-supplied, already-authorized
local `GraphView` to the existing `PathFactors`. It implements the selected
Topic 1.2 v12 §§15.3 and 18.2 direction/schema boundary. It does not alter the
published path operator, U4-K0 kernels or corrected public-basis policy. The
previous sections' deferred graph-adaptation statement is resolved only for
this explicitly declared local schema; graph builders/alignment and public
execution/message lifecycle remain separate work.

Exactly three new package exports are added:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class GraphRelationRule:
    relation_id: int
    canonical_relation_id: int
    direction: Literal["item_to_aux", "aux_to_item"]

@dataclass(frozen=True, slots=True, kw_only=True)
class GraphOperatorSchema:
    node_type_names: tuple[str, ...]
    relation_names: tuple[str, ...]
    item_node_type_id: int
    aux_node_type_ids: tuple[int, ...]
    relation_rules: tuple[GraphRelationRule, ...]
    ignored_relation_ids: tuple[int, ...]

prepare_path_factors_from_graph(
    view: GraphView | None,
    schema: GraphOperatorSchema,
    *,
    num_items: int,
    dtype: torch.dtype,
) -> PathFactors
```

### Explicit declarations and local validation

All declaration IDs/counts are exact built-in integers excluding Boolean,
NumPy and integer-subclass substitutes. num_items is positive; model dtype is
exactly torch.float32 or torch.float64. Vocabulary declarations are tuples of
unique nonblank plain strings. Item and nonempty unique auxiliary type IDs
must be in the node vocabulary and disjoint. Rule IDs and ignored IDs must be
unique, in range, disjoint and together cover **every** declared relation,
including unused entries. A rule's canonical ID is nonnegative; several rules
may deliberately share one canonical relation. Its direction is explicit.
Records validate at construction, and the factory revalidates all declarations
even for None or zero-edge input. No defaults are inferred from names.

A real input must be an exact local GraphView with ordinary plain NumPy int64
arrays, without array subclasses, masks or dtype metadata. Supported ordinary
strided/read-only arrays are not coerced. The adapter runs local
`GraphView.validate()` once per preparation, and matches both vocabularies to
the schema exactly in order/content. Local structure, including raw-ID rules,
all endpoints/relations and explicit stored duplicates, remains subject to the
existing GraphView contract. No structural defect is repaired or laundered.

`item_to_node` is mandatory, with exact length num_items. -1 means unmapped;
nonnegative targets must be injective and have the declared item node type.
Every local node of that item type must have one public anchor, including
isolated nodes. There is no implicit catalogue extension, metadata-only item
support, raw-ID join or fallback to user_to_node. User mappings are validated
as local GraphView structure but never become a second item anchor map.

For each selected edge, item_to_aux requires item -> declared auxiliary type;
aux_to_item requires the reverse storage orientation. Wrong selected endpoint
types fail. Other node types/edge roles enter only through explicitly ignored
relations (and may remain isolated). Ignored edges are excluded before degree
calculation, while their underlying GraphView structure must still be legal.
Metadata is ignored, never used to infer an inverse, exclusion or permission.

The caller must supply only its authorized local projection and keep it
stable throughout the call. Structural/schema validation does not authenticate
provenance or prevent access by code holding an original global graph. The
adapter accepts no DatasetBundle, GraphPartition, roster or runtime and makes
no global graph fingerprint/validation/index-building calls.

### Compact canonicalization and unchanged numerical semantics

Mapped public IDs become sorted local rows and include mapped isolates. Each
selected edge becomes `(local item row, canonical relation ID, local auxiliary
node index)`. Canonical relation IDs are passed with their explicit values to
the existing int64 incidence factory. Exact canonical duplicates are collapsed
before invoking `prepare_path_factors`, so declared forward/inverse or alias
storage copies count once. Different canonical relations keep separate columns
even at the same auxiliary node. One call adapts one view; raw auxiliary names
never align nodes between parties/views.

None is explicit caller-supplied absence and produces zero local rows/triples
with the requested public catalogue size/dtype. A schema-carrying zero-node
view, an all-unmapped non-item graph, mapped isolates, singleton auxiliary
columns and all-ignored relations preserve the accepted zero-action behavior.
A valid structural graph can yield zero T under a chosen B; the existing
structural validity predicate is retained, with no B/T norm test.

Only the existing factory computes degrees/normalization/S/h and validity.
It uses local deduplicated binary incidence, float64 coefficient construction
followed by model-dtype computation. Existing Apply, two-hop diagonal removal,
two ungated applications for four-hop with retained returns, source/sketch
parameter gradients and receiver freeze rules remain unchanged. No dense R,
adjacency, item-pair operator or selection matrix, extra inverse/self-loop,
epsilon or additional normalization is constructed by this adapter.

Preparation reads the delivered length-n mapping: this is explicitly O(n),
not O(batch) or constant memory. Local GraphView validation also scans local
structure. Per-rule edge scans, mapped-node sorting and canonical triple
sorting/deduplication need local workspace. Numeric coordinate arrays scale
with selected edges/support rather than catalogue pairs. Existing full-length
input mapping storage belongs to the delivered GraphView, not a retained
per-party adapter cache. Returned factors own detached numeric buffers; they
retain no GraphView, input arrays, raw-ID tuple, schema record or metadata.
Caller mutation after preparation cannot alter those factors. Tensor fields
of PathFactors themselves remain mutable under its existing contract.

### Evidence and limits

`tests/test_graph_semantic_graph_adapter.py` declares public semantic triples
independently of adapter outputs and builds tiny dense NumPy R/S/M/M² oracles.
The fixed fixture includes explicit inverse/alias copies, two canonical
relations sharing an auxiliary node, sparse public IDs and an isolated row.
It compares Apply, source logits/BCE and sketch at float64 atol=1e-7 and
float32 atol=1e-5. Permutation/raw-ID/forward-only controls, local deletion
and relation-separation changes, strict declaration/mapping/type failures,
absence/empty/zero-projection cases and input/RNG/ownership checks are covered.
Dense graph matrices occur only in tiny test oracles.

A test-owned composition prepares an existing GraphRuntime, projects two
arbitrary party IDs with disjoint private edges/auxiliaries and shared public
anchors, adapts each local view once, then uses accepted source/sketch/receiver
functions. Global hash/validation/index helpers are guarded after runtime
preparation; local GraphView.validate is counted only during adaptation and
never per batch. The test owns runtime close; local factors remain usable.
This is numerical composition, not a production stage/training executor.

A bounded n=4096, mapped m=256, local auxiliary q=64 probe contains 1024 stored
forward/inverse edges and 512 unique canonical incidences. It reports actual
mapping/triple/factor buffer sizes, retained tensor shapes, Torch operation
shapes, preparation/factory/global-call counts, tracemalloc and process RSS.
Torch shape traces miss NumPy/native workspace; tracemalloc misses untracked
native/PyTorch allocations; RSS includes imports/prior tests. Source inspection
and the exercised shape trace cover absence of graph-square allocations, with
no timing threshold or formal resource/learning qualification.

Required focused regressions are the unchanged U4-K0, public-basis,
graph-resolution, graph-runtime, graph-segment and FedGMF modules. The default
full gate remains a later combined-candidate task. No ML1M/Books graph is built,
no graph-wide semantic alignment or basis distribution is added, and collection,
optimizer/stage/message versions/accounting, privacy and full U4/M2 remain
separately authorized work.

### PW3-C-P2-int64-r1: explicit canonical-label representation bound

The corrective contract adds an upper bound that the original task's
"nonnegative" wording did not state: `canonical_relation_id` must be an exact
built-in integer satisfying `0 <= canonical_relation_id <= 2**63 - 1`. Boolean, NumPy scalar,
integer-subclass and floating substitutes remain invalid. This bound matches
the existing CPU `torch.long` incidence representation; a canonical ID is an
exact label, never an array dimension or an invitation to remap/wrap/clip it.

Rule construction, schema construction and preparation revalidation apply the
same range check before inspecting any graph, converting coordinates or calling
the existing factory, including None, empty/isolate inputs and rules with no
local edge. Supported labels keep their exact values and all accepted path,
source and sketch computations remain unchanged. Boundary regressions include
zero, 2**40, both largest int64 labels, and overflowing exact integers; external
corrective evidence compares the original and corrected valid-input outputs.
