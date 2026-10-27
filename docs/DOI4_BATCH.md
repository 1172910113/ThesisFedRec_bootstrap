# DOI4 batch delivery and source slots (DOI4-BATCH-K0)

`src.federation.doi4_batch` implements three bounded public federation
operations: copying an already authorized binary delivery to Torch, planning
source/item row geometry for one explicit step, and packing already computed
source representations for all receivers. Import it directly from the
submodule. Production depends only on the standard library, NumPy and Torch;
it does not import method code, data runtimes, or the parallel PW8 components.

This is an in-memory numeric boundary. Public execution remains responsible
for permissions, map/table ownership, source computation and versions, loss,
complete gradient contributions, optimizer updates, scheduling and evaluation.
Party IDs and public item IDs are separate namespaces; neither is a user ID.
No model parameters, initialization, optimizer, backbone embedding gather,
training loop, transport protocol or checkpoint is created here. Existing
FedGMF, Phase 3 behavior, data mappings, splits and sampling are unchanged.

## APIs and records

```python
tensorize_binary_delivery(
    payload, *, expected_entity_ids, public_item_count, local_field_count,
) -> BinaryTensorBatch

prepare_doi4_batch_requests(
    receiver_item_ids, *, source_party_ids_by_receiver,
    online_party_ids, public_item_count, context_id,
) -> Doi4BatchRequests

pack_doi4_source_slots(
    requests, *, source_publications, slot_count,
    compatible_dim, dtype, expected_context_id,
) -> tuple[ReceiverSlotBatch, ...]
```

The three result records are frozen, slotted, keyword-only and factory-only;
use these functions rather than constructing unchecked records. Freezing
prevents ordinary field assignment, not mutation of a Tensor's values or
header. Returned values are not authenticated or permanently immutable.
Input containers, arrays, tensors and plan fields must remain stable throughout
each call. Reflection is not a hostile-process security boundary; packing
revalidates the current plan's content even though normal construction is
restricted.

| Record | Fields |
|---|---|
| `BinaryTensorBatch` | `entity_ids`, `local_field_ids`, `binary_values`, `valid_unit_mask` |
| `Doi4BatchRequests` | `context_id`, `public_item_count`, `online_party_ids`, `receiver_party_ids`, `receiver_item_ids`, `source_party_ids_by_receiver`, `source_party_ids`, `source_item_ids`, `receiver_source_row_indices` |
| `ReceiverSlotBatch` | `receiver_party_id`, `entity_ids`, `slot_source_party_ids`, `slot_indices`, `compatible_tokens`, `valid_token_mask` |

Records retain only independently owned output tensors or recursively immutable
built-in tuples/integers and the built-in context string. They retain no input
array, dictionary, runtime, feature view, map, metadata, closure or bundle.
Binary `entity_ids` and all receiver identifiers/slot labels are trusted row
sidecars. Pass only the three binary unit tensors explicitly into the encoder;
pass only tokens and the validity mask into the compatible-space kernel.
Passing an entire record or its sidecars into model features is unsupported.

## Authorized NumPy delivery to Torch

The caller first uses the accepted runtime/adapter path and its expected
binding to obtain an authorized delivery, then calls
`tensorize_binary_delivery`. The payload has exactly the current
`deliver_binary_field_units` five-key schema:

| Key | Accepted value |
|---|---|
| `entity_kind` | Exact built-in string `"item"` |
| `entity_ids` | Plain native `int64` ndarray, shape `[E]` |
| `local_field_ids` | Plain native `int64` ndarray, shape `[E,J]` |
| `binary_values` | Plain native `int64` ndarray, shape `[E,J]` |
| `valid_unit_mask` | Plain native `bool` ndarray, shape `[E,J]` |

The outer payload must be an exact built-in dictionary. Extra keys, including
metadata, semantic field/view names, `unit_type`, party data or GT sidecars,
are rejected without traversing their contents. Arrays must be exact
`numpy.ndarray` objects with no dtype metadata, fields or subdtype. Subclasses,
non-native byte order and implicit dtype conversion are rejected; supported
read-only and strided plain arrays are accepted.

`expected_entity_ids` is an exact tuple of exact built-in integers, with
original request order and duplicates preserved. Boolean, NumPy-integer and
integer-subclass IDs are rejected. `public_item_count=I` and
`local_field_count=J` are exact built-in integers in `[0, 2**63-1]`. Every item
must lie in `[0,I)`, and the packet's `entity_ids` must equal the expected
request element by element. Shape alone does not establish row identity.

The dense columns are positions in the already established local field map.
For every valid cell `[e,j]`, its field ID must be exactly `j` and its binary
value exactly `0` or `1`. Every invalid cell must have the exact placeholder
`(-1, 0, False)`. Observed zero is valid data; neither code zero nor a later
representation's norm determines absence. This module does not create or
infer local maps from these positions.

The full schema, types, ranks, shapes, item identities, every cell's content
and integer capacity are checked before copying outputs. The result has four
new CPU contiguous tensors: IDs/codes are `torch.int64`, validity is
`torch.bool`, and all have `requires_grad=False`. Storage is independent of the
packet, other outputs and other calls; no `from_numpy`/`as_tensor` alias to
caller storage survives. Editing an input, a returned tensor or its header
cannot modify another return or the original packet. No RNG is used.

`J=0` permits nonempty legal item requests and returns `[E,0]` unit tensors.
`E=0` permits any legal `J`; it still requires a complete, correctly typed
schema. `I=0` requires an empty item request. The width `J` and row equality
do not prove `runtime_binding_id`, `map_id`, schema epoch or ACL. The caller
must validate those before delivery and bind the actual local map to the
owning encoder's field table outside this module.

## Explicit source/item geometry

Both `receiver_item_ids` and `source_party_ids_by_receiver` are exact
dictionaries with identical keysets. Party IDs are exact built-in integers in
`[0, 2**63-1]`. `online_party_ids` is an exact sorted, unique tuple of those
IDs; every receiver and declared source must belong to that explicit roster.
Noncontiguous IDs are legal. The helper neither reads nor constructs a
`DatasetBundle`, so a noncontiguous fixture does not change bundle partition
rules or assert a party-to-user mapping.

Each receiver's item rows form an exact tuple of in-domain public item IDs;
order and repetitions are meaningful and preserved. Each receiver's source
pool is an exact sorted, unique tuple, possibly empty. Own source is present
only when explicitly listed, at most once. The caller supplies the approved
OWN/MULTI/inference pool. Planning does not select, replace, normalize or add
sources based on train counts, coverage, norms, metadata, graph validity,
candidates or benefit; it does not infer validation pools from the final
training roster.

The plan contains the following aligned built-in tuples:

- `receiver_party_ids`: sorted receiver IDs, independent of dictionary order.
- `receiver_item_ids` and `source_party_ids_by_receiver`: original row tuples
  and explicit sorted pools aligned with those receiver IDs.
- `source_party_ids`: sorted union of all declared source pools, including
  sources declared only by a receiver with zero item rows.
- `source_item_ids`: for each source, the sorted unique union of original
  item rows across all receivers naming that source.
- `receiver_source_row_indices`: for each receiver, one inverse tuple per
  declared source, mapping every original item row into that source's unique
  item tuple. Repeated items map to the same source row.

For example, receiver rows `(4,1,4)` and `(1,3)`, with pools `(7,41)` and
`(41,90)`, produce source rows `7 -> (1,4)`, `41 -> (1,3,4)` and
`90 -> (1,3)`. The first receiver's inverses are `(1,0,1)` for source 7
and `(2,0,2)` for source 41. The second receiver's inverses are `(0,1)`
for source 41 and `(0,1)` for source 90. Each source/item representation can
therefore be computed once for this plan and reused across receivers.

Plans own their geometry; later edits to input dictionaries do not change
existing plans. An empty receiver domain produces empty source geometry.
An empty receiver's nonempty explicit pool remains declared, even when its
source item union is empty. When `I=0`, all item rows must be empty.
Calling the planner again performs the geometry work again; there is no
registry, cache, cross-step deduplication or lifecycle promise.

`context_id` is an exact built-in lowercase 64-hex string supplied by trusted
public code for its established step/config/version intent. The module neither
derives a hash nor interprets that intent. Packing validates the same format
for `expected_context_id` and requires equality before output allocation.
This caller-tag comparison does not authenticate H's actual source, parameter
version, message completeness, permissions or exactly-once processing.

## Validate once, then pack all receiver slots

`source_publications` is an exact dictionary keyed by precisely the plan's
`source_party_ids`. Each value is an exact three-tuple:

```python
(source_entity_ids, tokens_or_None, available)
```

`source_entity_ids` is an exact tuple equal, element by element, to the
source's complete planned unique item tuple. It binds numeric rows to the
plan rather than trusting a shape or dictionary order; it does not prove the
representation's mathematical origin or permission to publish it.

`available` is a dense, strided CPU `torch.bool` tensor of shape `[U_s]`.
A present `tokens` tensor is dense, strided CPU, shape `[U_s,K,h]`, with dtype
exactly the explicit `torch.float32` or `torch.float64`. `K=slot_count` and
`h=compatible_dim` are required exact built-in integers in `[1,2**63-1]`.
Parameters, nonleaf and strided tensors are supported without detaching.
Every element must be finite, including masked or currently unreferenced
cells. No coercion, clipping, promotion or source dropping repairs a failure.

Before any receiver output is allocated, packing checks all scalar arguments,
context and the complete current plan, then all publication keys, triples,
row identities, tensor types/layout/device/dtype/shapes and finite content.
Plan checks include sorted/unique/domain constraints, complete source union,
exact per-source cross-receiver unique item unions, all alignment lengths,
and `source_item_ids[inverse[row]] == receiver_item_ids[row]`. An in-range
inverse pointing to the wrong item is rejected. Factory-only construction and
the caller tag cannot substitute for these content checks. A malformed later
source is rejected before an earlier receiver output is published.

For each receiver, sources appear in its explicitly sorted pool order; each
source contributes slots `0..K-1`. For `S` declared sources:

```text
compatible_tokens:     [E, S*K, h]
valid_token_mask:      [E, S*K]
slot_source_party_ids: tuple(s for s in pool for k in range(K))
slot_indices:         tuple(k for s in pool for k in range(K))
```

Both slot sidecars have exactly `S*K` members and align one-to-one with token
columns. Source rows are gathered using the validated inverses. Availability
is gathered and repeated across the source's K slots. A false mask produces
exact zero; a true mask retains the original H, including valid all-zero H.
There is no source normalization or `m/N` weighting.

`None` explicitly declares that the source tensor path did not execute and
requires availability to be entirely false. Its output segment is an ordinary
same-dtype CPU zero tensor, with no invented trainable leaf or dependency on
another source. A present H with all-false availability remains on its
executed gather/masking/cat graph and yields connected zeros, preserving the
caller's distinction between structural absence and a numerical zero path.
This routing is never inferred from a norm or availability alone.

All receiver buffers have new storage independent of publications and other
receiver outputs. New storage preserves autograd: packing never detaches,
uses `.data`, converts H through NumPy or reconstructs it with `torch.tensor`.
One caller objective over multiple outputs can backpropagate once through
shared H; autograd sums duplicate-item and cross-receiver contributions into
that original H and its upstream graph. Packing itself never runs backward,
`autograd.grad`, a loss or a VJP, and never changes caller `.grad` fields.
Later caller backward is an intended use, rather than a packing input mutation.

`S=0` yields `[E,0,h]`/`[E,0]`; `E=0` yields `[0,S*K,h]`/`[0,S*K]`.
An empty receiver domain returns `()`. Explicit present H keeps a natural
empty gather path even for `E=0`; `None` absence gains no connected dependency.
Inactive large but finite H is never summed and multiplied by zero to make
an artificial graph link, because that intermediate sum may overflow. Natural
empty gather/cat paths, or a valid empty slice sum that reads no large values,
provide any needed empty connectivity safely.

For a completely unavailable receiver, the caller decides whether to bypass
F or invoke subsequent user/head computations. Packing returns the actual
mask and makes no claim that later kernel/parameter gradients are `None`.
No fake source-only user sample or loss is manufactured. User/item backbone
row gather remains ordinary Torch indexing on the caller's authoritative
Tensor, outside this source-H geometry helper.

## Errors, capacity and costs

Wrong container, record, member, dtype, device or layout raises `TypeError`;
wrong rank/shape, out-of-domain IDs, mismatches and illegal content raise
`ValueError`. Multiple independently invalid inputs need not have a unique
first error, but validation precedes large receiver output allocation.
Allocation failures propagate without retries or recovery claims.

Dimensions, element counts and necessary stride products are checked with
Python integer arithmetic against `2**63-1` before allocation. The conservative
product of `max(1, extent)` prevents a zero axis from hiding an overflowing
nonzero extent or stride product. This includes `[E,J]`, source `[U_s,K,h]`
and receiver `[E,S*K,h]` geometry. Empty source/receiver domains do not bypass
scalar dtype/context/schema checks. Boundary validation does not require
allocating an actually enormous tensor.

Tensorization scans and copies `O(E*J)` cells, plus `O(E)` entity IDs; empty
width still entails row validation and ID copying. Planning scales with the
actual receiver/source/item occurrences, deduplication sorting and inverse
construction; its stored tuples scale with that explicit geometry. Packing
first revalidates the current plan's full geometry with that planning cost,
then validates each source publication once in `O(sum_s U_s*K*h)` and builds receiver outputs in
`O(sum_c E_c*S_c*K*h)` plus row/slot mask and sidecar work. It does not rescan
a source's full H for each receiver. Plan validation and sorting, finite-check
workspace, gathers, masks, concatenations, output storage and autograd index
retention all have real costs.

There is no full canonical projection, known-positive scan, catalogue-wide
embedding/parameter copy, n-by-n object, all-party parameter cache or claimed
constant-memory execution. Tiny synthetic interface acceptance provides no
RSS, throughput, real-data, privacy, recommendation-quality or scale
qualification. This component establishes neither a complete joint executor
nor G3/U4/M2 readiness.

## Validation evidence and scope

The new `tests/test_doi4_batch.py` freezes small literal row/slot/inverse
expectations and independently derived linear gather/scatter gradients before
execution. It covers strict schema/error boundaries, isolated storage,
noncontiguous parties, repeated item rows, source reuse, empty geometry,
caller-tag checks, plan relationship validation and `None` versus connected
zero behavior. One bounded adapter-to-existing-math case may import accepted
method kernels in tests; production retains its public numeric-only dependency
direction. No new helper serves as its own expected-value oracle.

Actual node counts, warnings, source origins, process identities, focused
results and resource samples belong to the external PW8/B implementation
evidence and handoff; this document does not predeclare a pass count or imply
an independent approval. The final focused scope is exactly
`tests/test_doi4_batch.py`, `tests/test_model_inputs.py` and
`tests/test_doi4_composition.py`. Accepted adapter/math regression evidence is
REUSED baseline evidence, distinct from NEW boundary acceptance. No default
full-suite run, old scientific probe, real-data pilot or hyperparameter freeze
is part of this unit. Completion stops for independent exact-commit review.

The new component does not reopen historical evidence or enlarge its claims:
A's early clone/initial-shell gaps, PW4 A's 43/44 with a 1016-byte missing item
of unknown role/value, PW5 B r1 UNRESOLVED and r2 bounded process UNKNOWN, and
PW6's limited treatment of two early hashes retain their prior dispositions.
They neither qualify this component nor excuse a missing or wrong new hash.
