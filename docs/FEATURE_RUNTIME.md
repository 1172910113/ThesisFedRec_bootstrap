# In-memory feature runtime (PW3 A)

This bounded public-data unit snapshots a validated feature assignment for a
trusted coordinator. It does not change the strict resolver, canonical contracts,
sample ownership, scenario recipes, cache schema, model inputs, or training.

## Public API

```python
prepare_feature_runtime(
    bundle: DatasetBundle, partition: FeaturePartition,
    parties: Sequence[PartyData], *,
    source_id: str, split_id: str, recipe_id: str,
) -> FeatureRuntime
check_feature_runtime_binding(runtime: FeatureRuntime, *, binding_id: str) -> None
project_party_features(
    runtime: FeatureRuntime, *, party_id: int, binding_id: str,
) -> dict[str, FeatureView]
close_feature_runtime(runtime: FeatureRuntime) -> None
```

These functions and `FeatureRuntime` are exported by `src.data`. Construct the
handle only through preparation; `FeatureRuntime()` fails. Its sole supported
public data property is `binding_id`. Close is idempotent for a correctly typed
handle, releases its references, and makes both property reads and delivery fail.
Wrong handle types, stale expected bindings, and unknown parties fail before
output. Party IDs are exact nonnegative built-in integers, never user identities.

## Preparation and ownership

Keep all inputs stable during preparation using external coordination if needed.
The runtime validates **all referenced** current source views and the complete
assignment/roster, not only one requested party. Unreferenced views are ignored.
Roster IDs are unique; tuple-of-plain-string view summaries must match assignments.
Sample arrays are not scanned and no new ownership relation is inferred.

At this new boundary, arrays must be plain `np.ndarray` objects with no object,
structured, subdtype, or dtype metadata. Text is a tuple of plain strings;
schema strings must also be plain built-ins. Unsupported original types fail
before copying, even when the corresponding selection is empty. Existing strict
resolver acceptance of other representations is unchanged; equivalence is only
claimed for runtime-supported inputs.

One content snapshot is owned per referenced view, shared across parties.
Array buffers use independent immutable bytes; shape/dtype/logical values are
preserved exactly. Text and field names use immutable tuples. Snapshot and local
metadata are empty: arbitrary source metadata is neither traversed nor retained.
Original structural validation still requires a metadata dictionary. Copied
content, copied access, domains, and references are checked again before a handle
is published; source/owned identities must agree. No bundle, interactions,
raw namespace maps, `PartyData`, or sample arrays are retained.

Each view has one prepared processed-ID-to-row lookup and field-to-column lookup.
All-access descriptors retain `None` and use slices, not party-sized copies of
full-row vectors. Explicit selectors use work proportional to their sizes.
Private implementation storage is not a public type or security boundary.

## Binding version 1

All private identity envelopes use UTF-8 canonical JSON with
`sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False`,
then SHA-256 lowercase hex. No `repr`, Python `hash`, or pickle is used.

The exact runtime envelope is:

```text
{
  "format": "thesisfedrec.feature_runtime", "version": 1,
  "source_id": string, "split_id": string, "recipe_id": string,
  "domain_fingerprint": graph_domain_fingerprint(bundle),
  "interaction_count": len(bundle.interactions),
  "feature_fingerprints": {view_name: content_digest, ...},
  "partition_checksum": feature_partition_checksum(partition),
  "roster": [{"party_id": integer, "view_names": [sorted names]}, ...]
}
```

Roster entries sort by party ID. Referenced view names sort lexically. Existing
`graph_domain_fingerprint` and `feature_partition_checksum` formats are reused
unchanged. The latter preserves `None` versus empty versus explicit-full
selections, and absent versus explicitly empty assignment entries. Partition
names, metadata, descriptor/roster insertion order, timestamps, and hosts are
excluded. Exact nonblank context strings are not trimmed.

The exact private content envelope is:

```text
{
  "format": "thesisfedrec.feature_content", "version": 1,
  "name": string, "entity_kind": string, "modality": string,
  "feature_names": [ordered strings],
  "entity_ids": array_identity,
  "values": {"kind": "array", "array": array_identity}
         OR {"kind": "text", "items": [ordered exact strings]},
  "missing_mask": null OR array_identity
}
array_identity = {
  "shape": [integer, ...], "dtype": array.dtype.str,
  "sha256": sha256(array.tobytes(order="C"))
}
```

Storage layout is not identity; logical C-order bytes are. Endianness, signed
zeros, and NaN payload bits are not normalized. `None` masks differ from arrays.
Zero and NaN are not inferred missingness and no values are imputed.

`source_id`, `split_id`, and `recipe_id` are **asserted provenance**, not verified
files or split content. The reused domain identity verifies dataset name/schema
and user/item namespaces; interaction count verifies only cardinality, not row
meaning. There is no interaction-content/split scan or proof against held-out
leakage. A binding is trusted-local identity, not authentication or permission
enforcement against hostile Python.

## Delivery and lifecycle

Delivery checks only the expected binding and requested party, then uses owned
views and prepared lookups. It does not hash, rescan sources/roster/assignments,
rebuild indexes, or invoke the strict resolver/global validator.

Views sort lexically; rows use canonical processed-ID order and named fields use
canonical column order, regardless of request order. `None` selects all;
explicit empty selections produce visible empty payloads. Missing descriptors
produce no view, and a known featureless party returns a fresh `{}`. Field
selection, including `()`, on an unnamed/text view is invalid as before.
Sparse IDs are identities, not array offsets.

Delivered arrays are plain, C-contiguous, owning writable copies of selected
data; returned metadata is a fresh `{}`. Strings/tuples may share immutable
content. Mutating original arrays, headers, schema or metadata cannot change an
existing snapshot. Mutating a returned view cannot affect later/other delivery.
Old handles do not detect changes or revoke themselves: prepare a new runtime
and explicitly close the old one. New verified identity changes change binding;
metadata-only changes do not. No automatic hot reload or persistent cache exists.

Preparation costs source validation/hashing/copying plus namespace/assignment
validation and row/field indexing. Original and snapshot buffers coexist.
Transient validation/hash buffers and Python lookup overhead also consume memory.
Persistent content is one copy per referenced view, not one per party. Delivery
allocates selectors for explicit selections, selected copies (possibly an
indexing temporary), and output-local validation temporaries. It is neither
zero-copy nor constant-memory. Closing releases runtime references, not already
delivered outputs.

## Validation and limits

Synthetic tests use hand-computed projections and the unchanged strict resolver,
independent canonical-JSON binding construction, mutation/lifecycle checks,
global-work counters, both accepted whole-view recipes, and a bounded
4096-row/12-field buffer inventory. Named unchanged regressions cover feature
contracts/resolution/scenarios, ML1M genre building, graph access/runtime, and
FedGMF. Component testing does not run the default full suite.

Clearing metadata does not remove canonical field identity or lineage from
model inputs. Model-side schema sanitization, feature-aware training, DOI4
integration, missingness learning, privacy guarantees, persistence, concurrency
isolation, GPU/Books-scale qualification, and a real ML1M pilot remain separate.
