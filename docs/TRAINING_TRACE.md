# Training-trace K0: in-memory, single-round/epoch production

Import directly from `src.data.training_trace`. This unit adds no package
exports, persistent trace format, reusable sampler index, optimizer or training
loop. It leaves the existing sampler, round-drain planner, shared seed utility,
FedGMF/HFL path, evaluation protocol and all processed namespaces unchanged.

## Public API and owned record

```python
build_training_trace(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    *,
    selected_party_ids: Sequence[int],
    round_index: int,
    local_epoch: int,
    run_seed: int,
    negatives_per_positive: int,
    excluded_splits: tuple[str, str, str],
    replace: bool,
) -> TrainingTrace

check_training_trace_context(
    trace: TrainingTrace, bundle: DatasetBundle, parties: Sequence[PartyData],
) -> None
```

There are no default seeds, q, epoch or replacement policies.
`excluded_splits` must be the exact tuple `('train', 'valid', 'test')` in
that order. Subsets, reordered tuples, duplicates and bare strings fail.
This is the existing all-split known-positive exclusion, not use of held-out
feedback as training labels or an approval of another sampling policy.

`TrainingTrace` is frozen, slotted, keyword-only and factory-only; ordinary
construction raises `TypeError`. Its fields are exactly:

| Field | Representation |
|---|---|
| `protocol_version` | builtin int, 1 |
| `trace_id`, `context_id` | 64-character lowercase SHA256 hex strings |
| `run_seed`, `round_index`, `local_epoch`, `negatives_per_positive` | explicit builtin int in `0..2**63-1` |
| `replace` | exact builtin bool |
| `excluded_splits` | exact tuple `('train', 'valid', 'test')` |
| `selected_party_ids` | sorted tuple of actual party IDs, including selected empty parties |
| `positive_order_seeds` | tuple of `(party_id, derived_seed)`, sorted by party |
| `negative_seed` | builtin int in `0..2**64-1` |
| `ordered_positive_ids` | tuple of `(party_id, tuple_of_permuted_train_ids)`, sorted by party |
| `negative_items` | tuple of `(interaction_id, tuple_of_negative_items)`, sorted by interaction |

Only owned builtin scalars and recursive tuples are retained. No bundle,
PartyData, ndarray/header, mapping, metadata, RNG or closure remains in the
record. Later source edits cannot change it. Immutability is through supported
APIs, not a defense against deliberate reflection or `object.__setattr__`.
Drop the record to release it; there is no registry, cache or close lifecycle.

## Consumed-domain validation

Inputs must remain stable during a call: this is not a concurrent transaction.
The bundle is an exact `DatasetBundle`; the finite repeatable roster contains
exact `PartyData` records. Strings, bytes and generators are not rosters or
selected-ID sequences. All roster/selected IDs are exact builtin ints in
`0..2**63-1`; bool, NumPy scalars and int subclasses are rejected **before**
deduplication, sorting or key normalization. Duplicates and unknown selected
parties fail. Arbitrary non-contiguous party IDs and grouped users are valid;
party IDs never substitute for user IDs. Embedded `bundle.parties` rules are
unchanged and do not determine the explicit selection.

Only these current source fields are revalidated/consumed:

- Interaction ID, user and item columns: exact ordinary native metadata-free
  int64 rank-one arrays, aligned in length, with positional interaction IDs
  `0..T-1` and nonnegative user/item IDs in the processed domains. Read-only and
  non-contiguous ordinary arrays are supported without mutation or coercion.
- Both dict mappings: exact str/int raw keys (no bool/subclasses) and exact
  builtin-int values bijectively covering `0..size-1`. Raw int `1` and str
  `'1'` remain distinct; no whitespace or Unicode normalization is invented.
- All global split arrays: the same ordinary int64 representation, ascending
  unique in-range IDs, disjoint, with complete interaction coverage.
- Selected parties only: ascending unique train and user arrays; train is a
  subset of global train and disjoint across selected parties; every train
  interaction's user is owned by its party. `owns_labels` is exact bool and
  true for nonempty train histories. Repeated user-item events remain legal.

Unselected party **IDs/types** still validate; their other payloads are not
consumed. No full bundle/party reconstruction occurs. Timestamps, ratings,
party item/held-out vectors, features, graphs and metadata are not read.
Python-integer capacity checks on N and `N*(q+1)` precede RNG/sampling or
expanded payload allocation. q=0, no selected parties and selected empty
histories remain legal, but do not bypass any consumed-domain validation.
Tests construct real valid bundles: unmapped catalogue gaps are not smuggled
past the existing constructor to manufacture negative candidates.

Wrong types/representations raise `TypeError`; invalid values, ranges,
dimensions, coverage or identities raise `ValueError`. Sampler errors,
including insufficient eligible pools, propagate unchanged. There is no
fallback seed, q, replacement setting or client selection.

## Frozen local named-seed protocol v1

This narrowly approved unit uses private `numpy.random.default_rng` instances,
not the global shared-seed helper. It does not read or advance global Python,
NumPy or Torch RNG state. It is not a new global seed framework or thesis
numerical profile.

For all identity envelopes below, canonical encoding is exactly:

```python
json.dumps(obj, sort_keys=True, separators=(',', ':'),
           ensure_ascii=True, allow_nan=False).encode('utf-8')
```

There is **no** trailing LF. The seed envelope is exactly:

```text
{'format': 'thesisfedrec.training_trace.seed', 'version': 1,
 'run_seed': run_seed, 'round_index': round_index,
 'local_epoch': local_epoch, 'stream': stream}
```

For `stream='positive-order'`, append the sole extra key `party_id=p`.
For `stream='negative-items'`, add **no** party, cohort or source field.
The seed is `int.from_bytes(SHA256(canonical).digest()[:8], 'big', signed=False)`.

For each party in numeric order, pass its ascending train tuple to
`np.random.default_rng(positive_seed).permutation(...)`, storing the resulting
order. Every selected party, including an empty one, gets its own seed/order
entry. Reordering selection or roster insertion order has no effect; adding
an empty selected party does not change the negative seed or another party's
order. Source changes still change the context/trace identity as appropriate.

For N>0 and q>0, pass the ascending union of selected positives to the unchanged
`sample_negative_items` **once**, with uniform strategy, q, all three excluded
splits, replace and the derived negative seed. Preserve its ascending negative
tuple values and replacement duplicates verbatim. The sampler itself orders
positive IDs; party permutation does not alter sampling order. N=0 or q=0
calls it zero times, keeping every positive's empty negative tuple and the
ordinary negative seed identity without constructing a negative RNG.

Once means once per build, not persistent exactly-once execution. Reuse the
trace rather than rebuilding every minibatch. Rebuilding explicitly repeats
work; no registry prevents it. Seed fixtures and compatibility are exercised
under NumPy 1.26.4; untested cross-version bitstream equality is not promised.
Python 3.11 testing is not Python 3.10 or GPU qualification.

## Exact context and realized-trace identities

The context envelope has exactly these keys:

```text
format='thesisfedrec.training_trace.context', version=1,
user_map=[...], item_map=[...],
interaction_ids=[...], user_ids=[...], item_ids=[...],
splits={'train': [...], 'valid': [...], 'test': [...]},
parties=[...]
```

Mapping entries sort by processed ID and each is exactly
`{'processed_id': int, 'raw_type': 'str' or 'int', 'raw_id': original_scalar}`.
Arrays become lists of builtin ints in their actual logical order. Party
entries sort by selected party ID and each is exactly
`{'party_id': p, 'user_ids': [...], 'train_interaction_ids': [...], 'owns_labels': bool}`.
No unselected payload, timestamp, rating, feature, graph or metadata is hashed.
`context_id` is SHA256 of this canonical object. Temporary context lists/JSON
are not stored in the trace.

The realized-trace envelope is exactly:

```text
format='thesisfedrec.training_trace', version=1, context_id,
run_seed, round_index, local_epoch, negatives_per_positive, replace,
excluded_splits=['train','valid','test'], selected_party_ids=[...],
positive_order_seeds=[{'party_id': p, 'seed': s}, ...],
negative_seed=s,
ordered_positive_ids=[{'party_id': p, 'interaction_ids': [...]}, ...],
negative_items=[{'interaction_id': i, 'item_ids': [...]}, ...]
```

Tuple order corresponds one-for-one to list order. `trace_id` is SHA256 of
that canonical object; the hash itself is not an input to its own digest.

`check_training_trace_context` first checks exact trace type, complete fields,
scalar/tuple domains, named seed identities, sorted keys, unique positive
coverage, negative q/order/replacement constraints and the realized hash. It
then revalidates consumed current context, compares its hash, checks complete
per-party permutation membership and negative domain/known-positive exclusion.
This latter integrity check can scan the consumed interaction columns again.
It never calls sampler, permutation or any RNG. It does **not** prove the
realized order/sample was generated by replaying NumPy: no replay is performed.
It does not authenticate files, authorize access or detect simultaneous writes.
These are trusted local bindings, not cryptographic signatures or safe public
telemetry. Check once at an explicit context boundary, not in the batch hotpath.

## Existing planner integration (caller-owned)

```python
from src.data.training_trace import check_training_trace_context
from src.federation.round_drain import prepare_round_drain_plan

check_training_trace_context(trace, bundle, parties)
plan = prepare_round_drain_plan(
    bundle, parties,
    selected_party_ids=trace.selected_party_ids,
    ordered_positive_ids=dict(trace.ordered_positive_ids),
    negative_items=dict(trace.negative_items),
    round_index=trace.round_index,
    batch_unit='positive_group',
    positive_groups_per_batch=caller_explicit_B,
    negatives_per_positive=trace.negatives_per_positive,
    negative_replacement=trace.replace,
    trace_id=trace.trace_id,
)
```

The planner retains its own independent validation; this data module does not
import federation or a method. See `TRACE_BATCH.md` for packing a returned
nonempty `PartyGroupBatch`. Cursor advancement remains a caller assertion,
not evidence of optimizer/message commit.

## Costs, acceptance boundary and deferred work

Trace preparation scans/validates/hashes all consumed source ID columns,
mapping entries and split membership plus selected histories. Nonempty q>0
sampling additionally rebuilds the existing sampler's whole-bundle known
positive sets once. Subsequent planner preparation performs its own scan.
No stage is silently promoted to an O(batch) whole-pipeline claim.

Retained trace storage is O(N*(1+q)+|P|) in Python scalars/tuples. Temporary
lists, JSON bytes, sets, arrays, sorting/permutation workspace and hashing
overhead coexist; the sampler can still have NumPy workspace costs. Packing
alone costs O(m*(1+q)) time/output storage. Tiny synthetic correctness checks
are not production memory/throughput or recommendation-quality acceptance.

New tests exercise independent literal JSON/hash/seed fixtures, existing sampler
compatibility, call counts, consumed-field sentinels, arbitrary/grouped parties,
capacity/type errors, mutation/RNG isolation and complete trace→planner→packer
tails, q=0 and replacement. No finite differences are relevant to integer
trace/packing logic. No real data, GPU, learning curves or default full gate
are part of this unit. Reusable sampler indexing, persistence/recovery, access
messaging, model/optimizer execution and formal numerical profiles are deferred.
