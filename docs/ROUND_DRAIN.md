# Round-drain K0: supplied-trace planning only

Protocol: `round_drain_once_interaction_mass`, version **1**. Import directly
from `src.federation.round_drain`; package exports and existing FedGMF/HFL
behavior are unchanged. This unit does not implement a trainer or seed scheme.

## Public API and ownership

```python
prepare_round_drain_plan(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    *,
    selected_party_ids: Sequence[int],
    ordered_positive_ids: Mapping[int, tuple[int, ...]],
    negative_items: Mapping[int, tuple[int, ...]],
    round_index: int,
    batch_unit: str,
    positive_groups_per_batch: int,
    negatives_per_positive: int,
    negative_replacement: bool,
    trace_id: str,
) -> RoundDrainPlan
start_round_drain(plan: RoundDrainPlan) -> RoundDrainCursor
peek_round_drain_step(
    plan: RoundDrainPlan, cursor: RoundDrainCursor, *,
    requested_source_party_ids: Sequence[int],
) -> RoundDrainStep | None
advance_round_drain_cursor(
    plan: RoundDrainPlan, cursor: RoundDrainCursor, *,
    completed_step_index: int,
) -> RoundDrainCursor
```

All four records are frozen, slotted and keyword-only. Direct construction of
`RoundDrainPlan` is rejected: preparation alone publishes a usable plan.
Supported public interfaces expose immutable builtin scalars/tuples and
read-only mappings over owned dictionaries. No caller-owned dictionary,
ndarray header, bundle, party or mutable metadata is retained. Immutable trace
tuples can be shared safely; mutating input arrays/maps after preparation cannot
change the plan. Inputs must remain stable **during** preparation: this is not
an atomic concurrent snapshot or a security boundary against Python internals.

| Record | Fields |
|---|---|
| `RoundDrainPlan` | `protocol`, `protocol_version`, `plan_id`, `trace_id`; explicit `round_index`, `batch_unit`, `positive_groups_per_batch`, `negatives_per_positive`, `negative_replacement`; sorted `selected_party_ids` (including empty parties), `loss_party_ids` (nonempty only); mappings `positive_ids_by_party`, `user_ids_by_party`, `positive_item_ids_by_party`, `negative_items`, `positive_counts`, `batch_counts`; `round_positive_count`, `step_count`, `skipped`, `utilization_fraction` |
| `RoundDrainCursor` | `plan_id`, `next_step_index` |
| `PartyGroupBatch` | `party_id`, half-open group positions `start`, `stop`, aligned `positive_ids`, `user_ids`, `positive_item_ids`, `negative_item_ids`, exact unreduced `loss_coefficient=(stop-start,N_r)` |
| `RoundDrainStep` | `plan_id`, `step_index`, `batches` (nonempty, sorted by party), sorted `requested_source_party_ids`, `source_only_party_ids`, `idle_party_ids`, `bce_row_count` |

Per-party user tuples align with **interactions**, not the unique ownership-user
list. Users, items and parties remain distinct. A grouped party is supported;
party IDs need not be contiguous or fit int64. This profile validates HFL
sample ownership but does not claim VFL/HyFL execution. The explicit roster is
independent of `bundle.parties` and of feature/graph visibility.

## Validation and errors

Wrong object/type/representation raises `TypeError`; wrong value, shape, range,
ordering, domain, identity or membership raises `ValueError`. Failures do not
mutate inputs or consume RNG. Error prose identifies the relevant field/party/
interaction but is not a versioned wire format.

Checks run in this order: top-level/scalars, roster, consumed domains/splits/
ownership, trace keys and positive permutations, negative payload, snapshot/hash.

- Actual `DatasetBundle`/`PartyData` instances are required. Roster, selection
  and source requests must be repeatable `collections.abc.Sequence` objects,
  not strings/bytes/bytearray or generators. Roster IDs, selected IDs and
  requested IDs are unique exact nonnegative builtin ints; bool, NumPy scalar
  and int-subclass aliases are rejected before normalization.
- `round_index` is in `[0,2**63)`, B in `[1,2**63)`, q in `[0,2**63)`.
  `batch_unit` is the exact builtin string `positive_group`; replacement is
  an exact bool. No defaults for scientific B/q/replacement are introduced.
  q=0 is structurally legal, not a recommendation for a formal profile.
- Before expanded payload allocation, Python integer arithmetic rejects
  `B*(q+1)`, total groups or total expanded rows exceeding `2**63-1`, even
  for an empty trace. Total-row capacity is checked before negative lengths.
- Consumed arrays are exact plain, one-dimensional, metadata-free NumPy int64
  arrays, including empty arrays. Noncontiguous/read-only arrays are legal.
  Interaction IDs equal positions; user/item columns align and lie in the
  processed namespace. Existing dict ID maps must have exact builtin int
  values bijectively covering `0..n-1`; raw keys receive no new semantics.
- All three split arrays are sorted, unique, in range, disjoint and cover all
  interactions. No complete bundle reconstruction or feature/graph validation
  is called. Timestamps, ratings, held-out labels and metadata are not read.
- Selected party train/user vectors are sorted unique and in domain. Train
  IDs belong to global train, each row's user belongs to the party's user set,
  and no positive belongs to two selected parties. `owns_labels` is an exact
  bool, and true for nonempty train. Empty selected histories are valid.
  Only object, ID and duplicate-ID checks apply to unselected parties.
  Roster train need not cover every global train row.
- Positive mapping keys exactly equal selected party IDs, including empty
  entries. Values are exact tuples of exact nonnegative ints, each a complete
  once-only permutation of its party's train history. Supplied order stays.
- Negative keys exactly equal the selected positive union, sorted in the
  stored mapping. Values are exact tuples of q exact item IDs in range. The
  current target is **always** excluded. Within-group duplicates require
  explicit replacement; across-group duplicates are legal. No sorting,
  deduplication, imputation, resampling or all-split known-positive scan occurs.
- Mapping keys are individually type-checked before set conversion/copying/
  sorting. The planner does not coerce invalid trace scalars through `int()`.
  Conversion of already validated NumPy ID-array entries to builtin integers
  is only snapshot serialization.

Range/target validation is not full leakage certification. An approved producer
must enforce the configured known-positive exclusion and bind it to the trace
manifest. This planner cannot prove that a supplied negative came from that
producer or seed.

## Drain, roles and exact mass

For party c, `n_c=len(train_c)`, `T_c=ceil(n_c/B)`; `N_r=sum(n_c)` and
`T_r=max(T_c, default=0)`. Step t slices `[t*B:min((t+1)*B,n_c)]` only when
nonempty. Every positive occurs once, tails stay, and there are no epochs,
cycling, drop-last, extra positive replacement or active-client reweighting.
`utilization_fraction=(sum(T_c), len(loss_party_ids)*T_r)` is an unreduced pair,
or None when skipped. Empty selections/histories give N=T=0 and no batches.

Let L be the parties with current loss batches, S the explicit requested
sources and P the selected roster. Source-only = S minus L; idle = P minus
(L union S). A party can have both loss and source roles. Exhausted/empty
parties can remain sources. Source requests never change loss slices or mass;
unselected recruitment is rejected. Requests assert prior authorization but do
not prove row/field access, token availability or message completeness.

Future tensor packing is group-major: positive first, then its ordered
negative tuple; each group uses its interaction's user, never the party ID.
For m groups, offsets would be `(0,q+1,...,m*(q+1))`, with `(1,0,...,0)` labels.
No labels/tensors/logits are built here. `bce_row_count=sum(m)*(q+1)`.

The existing `positive_group_bce` is a mean of group means. Its caller must
multiply by **m/N_r exactly once**, in the approved loss dtype. Do not multiply
again by n_c/N_r, normalize by active parties or by step groups, divide again
by m, or multiply by T_r/T_c. Each group mass is 1/N_r; each row mass is
1/[N_r*(q+1)]. Source-only participation adds no loss/receiver-count factor.

The fixed n=(2,5), B=2, q=1 reference yields batches (2,2), (0,2), (0,1),
coefficients (2/7,2/7), (0,2/7), (0,1/7) and BCE rows 8,4,2. With fixed
same-state group gradients A=(1,3), B=(2,4,6,8,10), step gradients are
10/7, 2, 10/7 and total 34/7. The pre-implementation Fraction report is
reused, not rerun; new synthetic implementation tests check those expectations.
Role counts are not optimizer counters. Sequential parameter updates generally
do not equal one update on the complete-round gradient; no FedAvg equivalence
or unbiased full-round step estimator is claimed.

## Cursor, identity and replay

Start returns index zero. Peek is pure; equal inputs give equal descriptors.
At index T_r, empty source requests return None; nonempty requests fail.
Advance requires matching plan identity, exact valid index, and
`completed_step_index == cursor.next_step_index < T_r`. It returns a new
cursor, never mutating the old one. Replaying an old cursor is possible.
There is no implicit rollover or hidden iterator.

`completed_step_index` is a caller assertion, **not** proof of successful
optimizer commit or complete messages. The future executor must advance only
after success and provide exactly-once/deduplication/recovery semantics.
The scheduling key `(plan_id,step_index)` excludes source requests: different
request sets yield different roles under the same key. A future full request
identity must bind those execution inputs separately.

`trace_id` must be a lowercase 64-hex SHA-256 string supplied by the caller.
Its external manifest should bind dataset/namespace, interactions/splits,
sample ownership, roster, round, positive order/seed, negative settings/seed
or artifact, and environment. No file is opened or authenticated here.

`plan_id` is SHA-256 of UTF-8 JSON with `sort_keys=True`, `ensure_ascii=True`,
`separators=(",",":")`, `allow_nan=False`, no trailing newline. Exact envelope:

```text
format="thesisfedrec.round_drain_plan", version=1,
protocol="round_drain_once_interaction_mass", protocol_version=1,
trace_id, round_index, batch_unit, positive_groups_per_batch,
negatives_per_positive, negative_replacement,
parties=[{party_id, positive_ids:[...], user_ids:[...],
          positive_item_ids:[...], negative_item_ids:[[...], ...]}, ...]
```

Party entries sort numerically and include selected empty parties. Aligned
payloads retain supplied order. Counts are derived, not independent hash inputs.
Map insertion order, input layout, descriptions and unselected payloads do not
matter; selected IDs/payload/order/settings/round/trace do. Same selected
integers in another source require an appropriately different trace_id.
This is not a global source fingerprint, privacy/access guarantee, signature,
revocation mechanism or leakage proof. Plans are trusted coordinator records,
not a justification to give all parties one another's private payloads.

## Costs and deferred work

Preparation scans global ID columns, processed mapping domains and split arrays
once per plan, plus selected histories/users. It may hold global membership
sets; it is not O(batch). Retained plan storage is O(N_r*(1+q)+|P|) with Python
tuple/int overhead. Input buffers, transient validation sets and canonical JSON
serialization can coexist; this is not a production-scale memory qualification.
No features/graphs or known-positive indexes are scanned.

Peek visits P and copies only active batch descriptors; no global rescan or
rehash occurs. Explicit request normalization sorts S (O(|S| log |S|)); slicing
cost is proportional to active IDs, and immutable negative tuples can be shared.
Advance checks scalar identity/ranges only. Drop references to release plans;
there is no singleton, cache, close registry or persistent reusable index.

Separate future units must approve/implement:

1. Named seeded trace production, provenance manifests and group tensor adapter.
   The current sampler still performs a known-positive scan per call; a reusable
   sampler index needs separate ownership/binding/lifetime/compatibility review.
2. Request identities, parameter/snapshot versions, completeness checks and
   shared/local VJP accumulation. Missing/duplicate/nonfinite/wrong-version
   messages must fail before updates, not drop clients or renormalize loss.
3. Explicit approved optimizer/numeric profile and persistent owner state:
   own-loss/source-only/idle, connected zero versus grad=None, summed late
   contributions once, cross-round persistence and dense Adam momentum semantics.
   No LR/betas/epsilon/dimension/seed/private-pooling defaults are chosen here.
4. Fully committed checkpoint boundaries, exact replay and actual-message ledger.
   Commit exceptions require stop/recovery, not an atomicity claim.
5. Training/model/public-input integration and separately authorized real pilots.
   No warmup/intermediate test evaluation is authorized; existing evaluation,
   FIT-1, FedGMF/HFL and seed behavior remain unchanged.

The implementation calls no sampler, RNG/seed utility, evaluation, model,
method, optimizer, feature/graph delivery or filesystem I/O, and directly imports
no torch. Historical package transitive imports do not imply new execution.
No production throughput, recommendation benefit, privacy or complete U4/M2
claim follows from these integer planning tests.
