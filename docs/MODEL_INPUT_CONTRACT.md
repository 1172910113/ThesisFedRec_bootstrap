# Party-local binary input contract (INPUT-MAP-ADAPTER-K0)

This unit provides an explicitly ordered local field map and a prepared,
NumPy-only adapter for already-authorized item feature projections. It does
not alter the broader `FeatureView`, `FeatureAccess`, resolver, runtime,
sample-ownership, or schema-1.0 cache contracts. It neither constructs
authorization nor proves its origin. Party IDs remain distinct from user IDs.

Import these APIs directly from their submodules; package exports are unchanged.

## 1. Public interface

```python
# src.data.local_field_maps
prepare_local_field_map(
    *, party_id: int, schema_epoch: int,
    field_assignments: tuple[tuple[str, str], ...],
) -> LocalFieldMap
save_local_field_map(local_map: LocalFieldMap, path: str | pathlib.Path) -> None
load_local_field_map(
    path: str | pathlib.Path, *, expected_party_id: int,
    expected_epoch: int, expected_map_id: str,
) -> LocalFieldMap

# src.data.model_inputs
prepare_binary_input_adapter(
    authorized_views: Mapping[str, FeatureView], *, party_id: int,
    public_item_count: int, runtime_binding_id: str,
    local_map: LocalFieldMap,
    authorized_rows_by_view: Mapping[str, tuple[int, ...] | None],
) -> BinaryInputAdapter
deliver_binary_field_units(
    adapter: BinaryInputAdapter, *, entity_ids: list[int] | tuple[int, ...],
    expected_binding_id: str,
) -> dict[str, object]
close_binary_input_adapter(adapter: BinaryInputAdapter) -> None
```

`LocalFieldMap` is a frozen, slotted, keyword-only record owning only immutable
built-in scalars and tuples. Preparation validates it; save and adapter
preparation also reject inconsistent or corrupted maps, including a fabricated
`map_id`. `BinaryInputAdapter` is created only by its preparation function,
not by a supported public constructor. Its `binding_id` is a read-only scalar
for trusted caller bookkeeping. Neither type is a registry or a model state.

## 2. Stable field identity and map persistence

`party_id`, `schema_epoch`, and `public_item_count` must have `type(x) is int`
and be in `[0, 2**63 - 1]`. Booleans, NumPy integer scalars, and integer
subclasses are rejected. `field_assignments` must be an exact tuple of exact
two-tuples of exact built-in strings. Text is nonempty, has no leading or
trailing whitespace and no Unicode `Cc` character, including C0/C1 controls.
There is no trimming, Unicode normalization, or semantic inference.

Each `(view_name, field_name)` pair is unique; the same field name in different
views is valid. Position in the caller's tuple is the local ID `0..J-1`;
`J=0` is valid. No random seed, digest order, canonical-column order, or
per-delivery remapping chooses these IDs. Equal local integers across parties
do not imply shared meaning or permit sharing parameter rows.

The version-1 semantic object has exactly these keys:

```json
{
  "format": "thesisfedrec.local_field_map",
  "schema_version": 1,
  "entity_kind": "item",
  "map_algorithm": "explicit-order.v1",
  "party_id": 41,
  "schema_epoch": 3,
  "rows": [
    {"local_id": 0, "view_name": "vB", "field_name": "b"},
    {"local_id": 1, "view_name": "vA", "field_name": "x"},
    {"local_id": 2, "view_name": "vA", "field_name": "y"}
  ]
}
```

Every row has exactly the three shown keys; local IDs are ascending,
contiguous exact JSON integers. Define canonical encoding as
`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False).encode("utf-8")`. `map_id` is the lowercase 64-hex SHA256 of
the semantic object's canonical encoding. The file contains that object plus
exactly one `map_id` key, canonically encoded, followed by exactly one LF.
Neither the `map_id` field nor the physical LF participates in its digest.

Load rejects duplicate JSON keys, NaN/Infinity, missing or unknown keys,
incorrect JSON types (including boolean integers), versions or algorithms,
invalid text, duplicate pairs, unordered/non-contiguous IDs, invalid hashes,
and a mismatch with any expected identity. It recomputes the semantic hash;
checking only a declared hash is insufficient. Harmless physical JSON key
ordering or whitespace differences do not change semantic identity.

Save validates and encodes before creating an output, writes a same-directory
temporary file, flushes and fsyncs it, then publishes without overwriting any
existing target. An existing target, including a symlink, raises
`FileExistsError`; there is no check-then-overwrite race or automatic reload.
Only the temporary file created by that call is cleaned on failure. Ordinary
filesystem errors propagate. Callers create the parent directory and choose
an authorized data-root destination. This small map format is independent of
base-cache schema 1.0 and stores no feature payload or optimizer state.

Canonical column reordering preserves local-ID meaning through pair lookup.
Rename/drop/expansion/split/merge, permission/domain/content changes, or local-ID
relabeling require explicit new-run/map/snapshot handling, as applicable.
There is no rename certificate, semantic registry, row auto-append, migration,
or F/Adam recovery. Different content may reuse the same semantic map, but
must prepare a new adapter and binding.

## 3. Authorized projection and binary validation

Preparation revalidates current contents rather than trusting prior
`FeatureView` construction. Each value in `authorized_views` must be exactly
a `FeatureView`, with key equal to its name, `entity_kind="item"`, and
`modality="tabular"`. IDs must be plain, native `int64`, rank-one NumPy arrays,
sorted, unique, nonnegative, and below `public_item_count`. The row count must
match values. Names are an exact tuple of unique valid strings matching the
rank-two value array's column count. `metadata` must be a dictionary, but its
contents are not traversed, copied, or retained.

The map's pair set equals the entire field set of the supplied views, with no
omission or excess. View-name sets agree between the map, projections, and
authorization mapping. With `J=0`, both mappings must be empty: hidden views
versus visible zero-column diagnostics remain trusted sidecar information,
not phantom model tokens.

The adapter accepts only plain `np.ndarray` values with native-endian,
metadata-free `bool`, `int64`, `float32`, or `float64` dtype and rank two.
Non-contiguous and read-only ordinary arrays are supported. Array subclasses,
masked arrays, object/structured/subarray/text/complex/float16/other integer
dtypes and non-native endian dtypes are rejected before conversion.
Every supplied cell, including masked and not-yet-requested cells, must be
finite and exactly zero or one. Negative floating zero is legal code zero.
There is no clipping, thresholding, casting away invalid input, or imputation.
This is a stricter input subset, not a change to general `FeatureView` rules.

`missing_mask` is `None` or a plain native boolean ndarray with exactly the
same shape as values; row-mask broadcasting is unsupported. `None` means that
stored cells are observed. Source `True` means missing, the inverse of model
`valid_unit_mask=True`. Observed zero remains valid evidence. All invalid
output cells, whether source-missing or missing a source row, are normalized
to `local_field_id=-1`, `binary_value=0`, `valid_unit_mask=False`; masked raw
values and otherwise valid local IDs are not exposed through those cells.

Each `authorized_rows_by_view` value is either `None` or an exact tuple of
sorted, unique, exact integer IDs in the public item range:

| Selection | Meaning in this adapter |
|---|---|
| `None` | Entire public catalogue `[0, public_item_count)`; never expanded |
| `()` | No authorized IDs |
| Explicit tuple | Exactly those public IDs, independently of source rows |

This new `None` does **not** mean the same thing as
`FeatureAccess.entity_ids=None`, which selects only existing canonical rows.
The caller must explicitly provide the new domain; it is not inferred from
that descriptor. Every stored source ID must already be authorized for its
view. Out-of-domain stored rows fail rather than being silently filtered.

Every requested ID must be authorized for **every** selected view. One denied
view rejects the whole delivery, without a partial result or disguised
missingness. An authorized in-catalogue ID with no row in a view yields absent
cells only for that view; other authorized views may still have valid cells.
Out-of-catalogue IDs fail. This intersection policy is not a union-of-access
recipe, roster validation, access signature, or source-authenticity check.

## 4. Owned snapshot, representation binding, and lifecycle

Prepare validates all inputs before publishing a handle. It copies numeric
state into owned buffers, builds ID-to-row and field indices once, and retains
no caller `FeatureView`, metadata, mapping, array header, or GT object for
later reads. A late-view validation failure publishes no partial handle.
Callers must keep inputs stable during preparation; this is not a linearizable
snapshot under concurrent writes.

Runtime and expected binding IDs must be exact built-in strings containing
lowercase 64-hex SHA256 text. The map's party ID must equal `party_id`.
Adapter binding is computed once using the same canonical JSON rule above.
Its semantic envelope contains exactly:

```text
format = "thesisfedrec.binary_input_adapter"
schema_version = 1
policy_revision = "strict-binary-intersection.v1"
party_id, public_item_count, schema_epoch, map_id, runtime_binding_id
views = records sorted by view_name
```

Each view record contains exactly:

```text
view_name
feature_names          # JSON list in original projection column order
entity_count
field_count
values_dtype           # "bool", "int64", "float32", or "float64"
entity_ids_sha256
values_sha256
missing_mask_sha256    # null or lowercase hash
authorized_rows        # null or sorted JSON list
```

Array hashes cover the full original validated logical C-order bytes, with
numeric integer/float bytes canonicalized to little endian without changing
dtype; boolean values/masks use their bytes. This is not a digest of cleaned
output codes. `None` mask and an explicit all-False mask have different
representation bindings. Metadata is excluded. The envelope contains no
`binding_id` key and is SHA256-hashed without a trailing LF.

Delivery checks the supplied expected binding against the prepared identity,
not a rehashed source. Later caller mutations cannot alter the snapshot, and
delivery performs no global validation, source rehash, map rebuild, resolver,
runtime call, or RNG operation. Binding is trusted-local identity bookkeeping,
not authorization, source/split authentication, or a privacy guarantee.

The adapter does not retain a `FeatureRuntime`: closing a parent runtime does
not close or revoke this independent snapshot. At revocation or replacement,
the trusted caller must explicitly close the adapter and manage the runtime
separately. Closing a correctly typed handle is idempotent and releases owned
payload/index references; subsequent delivery fails. Already-delivered arrays
remain valid. No GC/destructor behavior is needed for lifecycle correctness.

## 5. Delivery and numeric boundary

Requests are exact lists or tuples of exact built-in integer IDs in range;
booleans, NumPy scalars, subclasses, floats, and strings are rejected. Unsorted
order and duplicates are retained. Validate the complete request, binding,
and all selected-view permissions before allocating or retrieving payload.
Empty requests do not bypass those checks; empty maps still validate request
IDs and bindings.

The returned plain dictionary has exactly five keys:

| Key | Value |
|---|---|
| `entity_kind` | Exact string `"item"` |
| `entity_ids` | New owning C-order native `int64` ndarray `[E]` |
| `local_field_ids` | New owning C-order native `int64` ndarray `[E,J]` |
| `binary_values` | New owning C-order native `int64` ndarray `[E,J]` |
| `valid_unit_mask` | New owning C-order native `bool` ndarray `[E,J]` |

Columns always follow local IDs `0..J-1`; valid field IDs equal their column
indices. All four arrays have independent storage, including empty shapes,
and do not alias inputs, internal buffers, or earlier output. Mutation of any
output cannot affect later delivery. `P=J` is not the count of valid units.
There is no separately inferred availability flag.

These five keys constitute a trusted framework delivery record, not an
entire model call. A future authorized caller may pass only the three unit
arrays, converted to CPU int64/int64/bool tensors, with caller-owned tables
to a numeric encoder. Entity kind and item IDs are routing/scoring data.
This unit implements no Torch conversion, embedding/parameter creation,
encoder, method callback, training, scoring, or optimizer. Names, recipes,
GT, metadata, map IDs, source hashes, handles, and closures are not returned.
The software boundary does not defend against malicious same-process
reflection/memory access or guarantee that semantics cannot be inferred.

## 6. Fixed example and validation boundary

Use `I=5`, party 41, epoch 3 and map order `(("vB", "b"), ("vA", "x"),
("vA", "y"))`. `vA` has rows `[0,2]`, values `[[0,1],[1,0]]`, and mask
`[[False,True],[False,False]]`; `vB` has rows `[0,3]`, values `[[1],[0]]`,
and no mask. Both are item/tabular. Domains are `vA=None` and
`vB=(0,2,3,4)`. Request `[3,0,3,2,4]` produces:

```text
local_field_ids = [[0,-1,-1], [0,1,-1], [0,-1,-1], [-1,1,2], [-1,-1,-1]]
binary_values   = [[0, 0, 0], [1,0, 0], [0, 0, 0], [ 0,1,0], [ 0, 0, 0]]
valid_unit_mask = [[T, F, F], [T,T, F], [T, F, F], [ F,T,T], [ F, F, F]]
```

`[1]` is denied; `[4]` is authorized but has no source rows; `[5]` is outside
the public catalogue. Zero rows, zero fields, all-masked cells, and empty
requests do not relax input validation.

Wrong object/container/element/dtype types raise `TypeError`; invalid
rank/shape/domain/duplicates/content, malformed JSON/keys/version/hash,
permission, closed handles, and binding mismatch raise `ValueError`.
Filesystem errors propagate. Checks proceed from type/structure to
shape/domain and then content; callers must not depend on the incidental
first error in multiply-invalid content. Failures do not mutate inputs,
previous handles, or existing files.

Preparation scans, validates, hashes, and copies the complete supplied
projection once, including masked/unrequested cells. Its time and storage
therefore scale with that projection plus maps, row indices, and explicit
authorization domains, not just with the next batch. Delivery requires
`O(EJ)` payload work/storage plus request and authorization lookups. `None`
domains require range checks rather than catalogue-sized allocation. No
catalogue-wide source scan or digest is repeated per delivery.

Focused synthetic tests cover independent canonical-JSON/binding oracles,
the fixed example, strict raw/mask/type cases, ordering and duplicates,
permission/source-absence separation, atomic map persistence, mutation and
GT isolation, lifecycle, no repeated global work, and existing feature-runtime
regressions. This contract does not authorize real-data acceptance, model
integration, profile selection, source-provenance governance, schema migration,
method research, or full training/checkpoint persistence.
