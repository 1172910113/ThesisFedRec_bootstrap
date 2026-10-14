# Data Contract

## 1. Purpose and core rules

All thesis methods should consume a consistent processed-data representation whenever their research setting permits. The same processed base dataset must support interaction, feature, graph, continual, and federation-partition views without repeating raw-data preprocessing.

The data model supports HFL first and reserves clean interfaces for VFL and HyFL.

The framework distinguishes three independent identities:

```text
user_id
item_id
party_id
```

A party is a federated data holder. It may represent one user, multiple users, an organization, or a feature holder. Therefore:

- `party_id` must never be treated as a `user_id`;
- sample ownership and feature availability are represented separately;
- all derived views reference the same processed user, item, and interaction IDs;
- derived views must not remap IDs, redefine splits, or parse raw data again;
- missing features are represented explicitly rather than silently imputed;
- continual learning remains independent of federation type.

Conceptual type aliases used below:

```python
RawId = str | int
IntArray = one-dimensional NumPy int64 array
FloatArray = one-dimensional NumPy floating-point array
Metadata = JSON-serializable dict[str, object]
```

Arrays containing IDs are sorted unless their order has documented semantic meaning.

---

## 2. DatasetBundle

`DatasetBundle` is the in-memory assembly of one processed base dataset, zero or one selected federation partition, and its requested canonical data views. Canonical `FeatureView` data belongs to the bundle; party-specific feature visibility remains a separate partition/scenario concern.

```python
DatasetBundle:
    name: str
    schema_version: str

    interactions: InteractionTable
    splits: InteractionSplits

    user_id_map: dict[RawId, int]
    item_id_map: dict[RawId, int]

    partition_name: str | None
    partition_kind: Literal["none", "hfl", "vfl", "hyfl"]
    parties: tuple[PartyData, ...]

    feature_views: dict[str, FeatureView]
    graph_views: dict[str, GraphView]
    continual_views: dict[str, ContinualView]

    metadata: Metadata
```

Invariants:

- interaction, user, and item IDs belong to bundle-wide namespaces;
- `partition_kind == "none"` requires `partition_name is None` and an empty `parties` tuple;
- a partitioned bundle has a non-empty `partition_name`;
- view dictionaries are keyed by the corresponding view's `name`;
- empty view dictionaries are valid;
- counts are derived from mappings and interaction arrays rather than duplicated as mutable fields;
- a bundle selects one partition at load time, while multiple partition definitions may coexist in the processed cache.

---

## 3. Interaction representation

### 3.1 Logical record

`InteractionRecord` remains the logical definition of one interaction:

```python
InteractionRecord:
    interaction_id: int
    user_id: int
    item_id: int
    timestamp: int
    value: float
```

It may be used at loader boundaries or in documentation, but it is not the primary in-memory storage format.

### 3.2 Columnar storage

The main interaction representation is columnar:

```python
InteractionTable:
    interaction_ids: IntArray
    user_ids: IntArray
    item_ids: IntArray
    timestamps: IntArray
    values: FloatArray
```

Rules:

- all five arrays are mandatory and have the same length;
- `interaction_ids` are unique, contiguous, and start at zero;
- row `i` across all arrays describes one logical `InteractionRecord`;
- rows are stored in ascending `interaction_id` order;
- timestamps use a source-specific normalized integer time unit;
- timestamp unit, timezone handling, and parsing rules are recorded in metadata;
- `values` preserve normalized feedback values; a binarized implicit-positive record uses `1.0`;
- duplicate user-item interactions are handled before ID assignment using a configured and recorded policy;
- equal timestamps are ordered by `interaction_id`;
- context, modalities, and graph relations belong in named views rather than arbitrary table columns.

Splits, parties, negative samples, and derived views reference interaction IDs rather than copying interaction rows.

---

## 4. User and item ID mapping

The base preprocessing pipeline performs mapping exactly once:

1. Normalize and filter raw interactions.
2. Establish a canonical source-record order.
3. Assign IDs by first appearance in that canonical order.
4. Assign user and item IDs independently.
5. Use contiguous integers starting at zero.
6. Preserve every raw ID and its type in the mapping artifacts.
7. Never create a second user or item mapping for a feature, graph, continual, party, or research-method view.

Additional rules:

- boolean, null, and floating-point raw identifiers are rejected unless a loader explicitly normalizes them to strings;
- mapping collisions after normalization are errors;
- filtering occurs before final mapping so removed entities do not create gaps;
- mapping policy, filtering policy, canonical ordering policy, and mapping checksums are recorded;
- a legacy-compatible experiment requiring another mapping rule uses a separately named processed cache and documents the deviation.

Graph nodes use a graph-local namespace. A `GraphView` provides explicit user/item-to-node mappings where applicable.

---

## 5. Train, validation, and test representation

```python
InteractionSplits:
    train_interaction_ids: IntArray
    valid_interaction_ids: IntArray
    test_interaction_ids: IntArray
```

The arrays contain only bundle-wide interaction IDs.

Default split procedure:

1. Sort each user's interactions by `(timestamp, interaction_id)`.
2. Assign the latest interaction to test.
3. Assign the second latest interaction to validation.
4. Assign all remaining interactions to training.
5. Under the default protocol, remove users with fewer than three retained interactions before final ID assignment.

Invariants:

- every base interaction belongs to exactly one split;
- the three arrays are disjoint;
- their union equals all base interaction IDs;
- each array is sorted by interaction ID after assignment;
- views reference these IDs rather than copying interaction data;
- a documented legacy split receives a separate split name or cache fingerprint and never silently replaces the thesis default.

---

## 6. PartyData

```python
PartyData:
    party_id: int

    train_interaction_ids: IntArray
    valid_interaction_ids: IntArray
    test_interaction_ids: IntArray

    user_ids: IntArray
    item_ids: IntArray

    feature_view_names: tuple[str, ...]
    graph_view_names: tuple[str, ...]

    owns_labels: bool
    metadata: Metadata
```

The fields have separate meanings:

| Concern | Fields |
|---|---|
| Sample partition | `train_interaction_ids`, `valid_interaction_ids`, `test_interaction_ids` |
| Entity coverage | `user_ids`, `item_ids` |
| Coarse view summary | `feature_view_names`, `graph_view_names` |
| Label availability | `owns_labels` |

Rules:

- party IDs are contiguous integers starting at zero within a partition;
- user and item arrays describe entity visibility and do not imply one party per user;
- interaction arrays reference the bundle-wide split arrays;
- feature and graph data are referenced by name rather than copied into each party;
- empty interaction arrays are permitted for future feature-only VFL parties;
- metadata may describe party grouping or controlled heterogeneity but must not hide required sample, feature, or label ownership.

---

## 7. HFL partition representation

The HFL builder supports at least two explicit modes:

```text
per_user
grouped_round_robin
```

The planned interface is:

```python
build_hfl_partition(
    bundle,
    *,
    mode="per_user",
    num_parties=None,
    seed,
    party_feature_views=None,
    party_graph_views=None,
) -> tuple[PartyData, ...]
```

### 7.1 `per_user`

`per_user` is the default thesis recommendation setting.

- Create one party for each retained user.
- Each party references exactly one user and all of that user's train, validation, and test interactions.
- Derive the party's item coverage from those interactions.
- Allocate party IDs in a separate party namespace.
- Store the user-to-party assignment explicitly through `PartyData.user_ids` and partition metadata.
- Never use `party_id` as a lookup substitute for `user_id`.

Party and user numeric values may happen to coincide in a particular processed partition, but this is incidental and must not become a contract or implementation assumption.

### 7.2 `grouped_round_robin`

- Require a positive `num_parties` not exceeding the number of retained users.
- Sort user IDs, apply one seeded permutation, and assign users round-robin to party IDs.
- Move every interaction for an assigned user into the same party.
- Derive each party's item coverage from its interactions.

### 7.3 Common HFL invariants

- every retained user belongs to exactly one party;
- every interaction belongs to exactly one party;
- item sets may overlap between parties;
- a party may contain one or many users depending on mode;
- every HFL party normally has `owns_labels=True`;
- global train, validation, and test membership is preserved;
- per-party feature-view lists may differ, allowing DOI4-style modality heterogeneity without changing sample ownership or silently imputing unavailable views;
- mode, seed, number of parties, assignment map, and per-party counts are recorded in partition metadata.

Additional non-IID modes may be added only when required and must have explicit configuration, tests, and metadata.

---

## 8. Reserved VFL and HyFL partition representation

VFL and HyFL builders are future interfaces and are not part of the initial Phase 2 implementation.

### VFL representation

- parties may reference overlapping or identical interaction IDs;
- parties may reference overlapping user and item coverage;
- `feature_view_names` and `graph_view_names` describe differing feature holdings;
- at least one party owns labels, while feature-only parties may use `owns_labels=False`;
- alignment uses explicit interaction, user, and item IDs rather than row positions.

### HyFL representation

- interaction coverage may differ and overlap;
- user and item coverage may differ and overlap;
- feature-view availability may differ;
- label ownership remains explicit.

Reserved signatures:

```python
build_vfl_partition(...) -> tuple[PartyData, ...]
build_hybrid_partition(...) -> tuple[PartyData, ...]
```

When these builders are eventually added, they must not call the HFL builder or force VFL through HFL aggregation semantics.

---

## 9. Negative-sampling ownership and interface

Negative sampling belongs to shared data infrastructure. Research methods must not define private sampling protocols.

The shared sampler returns negative item IDs only:

```python
sample_negative_items(
    bundle,
    positive_interaction_ids,
    *,
    negatives_per_positive,
    strategy,
    seed,
    excluded_splits,
) -> dict[int, tuple[int, ...]]
```

The result is:

```text
positive interaction_id -> ordered negative item_ids
```

Rules:

- returned tuples contain only negative item IDs;
- the positive item is never inserted by the sampler;
- candidate-set construction belongs to the shared evaluator;
- for evaluation, the evaluator combines the held-out positive item with the sampled negative item IDs and owns candidate ordering, ranking direction, and tie handling;
- default negatives must not occur in the user's configured known-positive set;
- the default thesis protocol excludes positives from train, validation, and test;
- the eligible item pool is ordered before seeded sampling;
- insufficient eligible items cause an explicit error unless replacement sampling is configured;
- strategy, seed, excluded splits, replacement policy, sample count, and artifact checksum are recorded;
- future privacy-preserving VFL sampling is protocol-specific and must not assume all parties possess HFL-style user histories.

---

## 10. FeatureView interface

`FeatureView` is the concrete canonical in-memory feature contract as of Phase 4B1. Feature-view builders and detailed party visibility remain outside Phase 4B1.

```python
FeatureView:
    name: str
    entity_kind: Literal["user", "item", "interaction"]
    entity_ids: IntArray

    modality: str
    values: NumPy array | tuple[str, ...]
    feature_names: tuple[str, ...]

    missing_mask: NumPy boolean array | None
    metadata: Metadata
```

Rules:

- `name` and `modality` are non-empty strings;
- `entity_kind` is exactly `user`, `item`, or `interaction`;
- `entity_ids` is a sorted, unique, non-negative NumPy `int64` array;
- each row in `values` corresponds to the same row in `entity_ids`;
- entity IDs are unique within a view;
- an entity absent from `entity_ids` does not possess that view;
- element-level missingness uses `missing_mask`; zero is not implicitly missing;
- NumPy values have at least one dimension; string values use a tuple with one string per entity row;
- when `feature_names` is non-empty, values are a two-dimensional NumPy array and the names match its columns;
- `feature_names` is empty for tuple-based text or other unstructured string values;
- feature names, when present, are unique non-empty strings;
- `missing_mask`, when present, is a boolean NumPy array with exactly the same shape as `values`;
- no value, including zero or NaN, is implicitly treated as missing;
- categorical encoding maps and tokenization details belong in metadata;
- different modalities are separate named views;
- `DatasetBundle.feature_views` keys equal their `FeatureView.name` values, and bundle validation checks entity IDs against the corresponding processed user, item, or interaction namespace;
- `PartyData.feature_view_names` remains a coarse backward-compatible summary and does not determine sample ownership;
- detailed party row/field/view visibility belongs to the later `FeaturePartition` contract;
- DOI4-specific imputation, fusion, and collaboration remain method-owned.

### 10.1 Party-specific FeatureAccess and FeaturePartition

Phase 4B2 adds detailed feature visibility as a derived scenario contract
outside `DatasetBundle`:

```python
FeatureAccess:
    view_name: str
    entity_ids: IntArray | None
    feature_names: tuple[str, ...] | None

FeaturePartition:
    name: str
    party_access: Mapping[int, tuple[FeatureAccess, ...]]
    metadata: Metadata
```

`FeatureAccess` contains references only; it never owns or copies
`FeatureView.values`. Its rules are:

- `view_name` names exactly one canonical `FeatureView`;
- `entity_ids is None` grants access to every canonical row, while an explicit
  sorted, unique, non-negative `int64` array grants access only to those
  canonical entity IDs;
- `feature_names is None` grants access to every canonical field, while an
  explicit tuple grants access only to those named canonical fields;
- field selection is unsupported for a view with no named fields, including
  tuple-based text views; such a view uses `feature_names=None` for its whole
  payload;
- an empty field tuple is a valid explicit selection of no fields from a
  field-addressable view;
- absence of a descriptor for a party/view means that the entire view is
  unavailable to that party.

`FeaturePartition` contains a scenario name, party-to-access mapping, and
JSON-compatible reproducibility metadata. It contains no interaction IDs,
user ownership, item ownership, feature values, or training state. Party IDs
are non-negative but need not be contiguous and never imply user IDs.

Access assignments are read-only through the public configuration interface.
Construction defensively copies the party mapping and exposes a read-only
mapping over that copy; each party's descriptors and each field selection
remain immutable tuples. Constructor ID arrays are validated before their
contents are copied once into immutable bytes. The read-only `entity_ids`
property returns a fresh one-dimensional NumPy `int64` view on every access;
no ndarray object/header is stored or shared between reads. These views share
the immutable backing bytes without copying or reserializing the ID buffer.
Changing one returned view's shape or dtype cannot affect other views or the
stored selection; item assignment and `setflags(write=True)` are rejected.
Mutating constructor inputs cannot change stored assignments. `None` remains
the full-access sentinel, distinct from empty bytes for no selected rows.
`None`, empty selections, and explicit full selections retain their distinct
meanings and checksums.
Metadata remains a plain dictionary and is not part of this immutability or
access-checksum guarantee.

Validation against a `DatasetBundle` and an explicitly supplied `PartyData`
set requires:

- every referenced party and view to exist;
- every selected entity ID to be a row in the referenced canonical view;
- every selected field to exist in a field-addressable canonical view;
- at most one descriptor per party/view;
- `PartyData.feature_view_names`, as a coarse compatibility summary, to match
  the detailed view names. Detailed rows and fields come only from
  `FeaturePartition`.

Source missingness and scenario visibility are separate:

```text
FeatureView.missing_mask = canonical/source missingness
FeaturePartition         = experiment/party visibility
```

Partition validation never changes the canonical values or missing mask. The
deterministic feature-partition checksum hashes canonicalized party, view,
row, and field access definitions. It is independent of dictionary,
descriptor, and selected-field input order; scenario names and metadata are
not part of the access-assignment checksum.

### 10.2 In-memory whole-view scenarios (first Phase 4B3 unit)

`src/data/feature_scenarios.py` exports two explicit functions:

```python
build_complete_feature_scenario(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    *,
    view_names: Sequence[str],
    name: str,
) -> tuple[FeaturePartition, tuple[PartyData, ...], dict[str, object]]

build_single_view_absence_scenario(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    *,
    view_names: Sequence[str],
    missing_view: str,
    missing_party_count: int,
    seed: int,
    name: str,
) -> tuple[FeaturePartition, tuple[PartyData, ...], dict[str, object]]
```

Both accept valid existing canonical views, independently of dataset name.
They reject empty party/view sequences, invalid element types, duplicate
party IDs or view names, unknown views, bare string/bytes `view_names`, and
non-string or blank scenario names. Party IDs follow `FeaturePartition`'s
non-negative Python-integer rule, excluding booleans and NumPy integers.
They need not be contiguous and are never renumbered or inferred from users.

Complete access uses no randomness. Single-view absence requires a selected
`missing_view` and Python-integer count/seed (not booleans, NumPy integers,
floats, or strings), with `0 <= missing_party_count <= N` and
`0 <= seed < 2**32`. Seed validation also applies at counts zero and `N`.
The recipe uses `numpy.random.default_rng(seed).choice(N, size=K,
replace=False)` over positions in the numerically sorted party list, then
maps those positions back to the actual IDs. It neither reads nor changes
global RNG state. Party input order and view input order do not affect results.

Outputs use sorted party IDs and lexically sorted view names, including
mapping/statistics order. Every supplied party has an explicit mapping entry,
possibly an empty tuple. Visible descriptors use `FeatureAccess(view_name=...)`
with both selectors `None`; hiding a view removes its descriptor. Zero absence
has the same assignment, summaries, statistics, and checksum as complete
access for the same parties/views, despite different recipe metadata or names.
All-party absence and featureless parties are valid; no resampling guarantees
an alternative view. Assigned zero-row or source-missing views remain valid.

Input parties supply sample ownership, not preexisting access restrictions.
The recipe replaces their coarse `feature_view_names` from the explicit
selected-view assignment: it neither intersects with previous summaries nor
retains unselected views. It returns new `PartyData` records changing only
that field, then calls the existing `validate_feature_partition()` against
those records. Unchanged arrays and metadata are shared via a shallow copy;
the returned parties are not deeply immutable or fully independent copies.
Construction does not change input parties, canonical values/masks, base
interactions/splits, or `bundle.parties`. The returned tuple stays outside
the bundle; the immutable access assignment remains a separate object.

Partition metadata contains `recipe` (`complete` or `single_view_absence`),
`recipe_version: 1`, sorted `party_ids` and `view_names` lists, and, for absence,
`missing_view`, `missing_party_count`, and `seed`. The unchanged
`feature_partition_checksum()` continues to identify only access assignments;
it excludes names/metadata and does not bind canonical payloads or provenance.

The third return value is a derived plain dictionary:

| Key | Meaning |
|---|---|
| `party_count` | Number of supplied parties, `N` |
| `selected_views` | Sorted tuple of explicitly selected view names, size `M` |
| `visible_view_counts_by_party` | Actual party ID -> assigned-view count, including zero |
| `visible_party_counts_by_view` | Selected view name -> assigned-party count, including zero |
| `featureless_party_count` | Parties with zero assigned views in this selected-view universe |
| `available_party_view_pairs` | `N*M` for complete access; `N*M-K` for single-view absence |
| `possible_party_view_pairs` | `N*M` |
| `assignment_checksum` | Existing access-assignment checksum |

These counts do not measure nonmissing rows, semantic overlap, privacy, or
recommendation quality. These recipes add no canonical feature construction,
persistence, row/field recipe, configuration expansion, or training integration.

### 10.3 Standalone ML1M item genres (separate Phase 4B3 unit)

`src/data/ml1m_features.py` exports one explicit builder through `src.data`:

```python
def build_ml1m_item_genre_view(
    bundle: DatasetBundle,
    *,
    movies_relative_path: str | Path,
    data_root: str | Path | None = None,
) -> FeatureView:
    ...
```

The builder returns an unattached canonical `FeatureView` named `item_genres`,
with `entity_kind="item"` and `modality="tabular"`. Rows follow the existing
processed item IDs: `entity_ids` is one-dimensional `int64`, `0..N-1`;
`values` is `float32` with shape `(N, 18)` and multi-hot entries in `{0, 1}`.
`feature_names` is this fixed lexical tuple from the MovieLens-1M README's
MOVIES FILE DESCRIPTION, independent of source subsets or interaction splits:

```text
Action, Adventure, Animation, Children's, Comedy, Crime,
Documentary, Drama, Fantasy, Film-Noir, Horror, Musical,
Mystery, Romance, Sci-Fi, Thriller, War, Western
```

The boolean `(N, 18)` `missing_mask` is entirely false after successful
validation. An absent genre in valid metadata is an observed zero, not source
missingness. Missing metadata for any processed item raises with the missing
count and representative raw MovieIDs; no partial or imputed view is returned.
Scenario visibility still belongs only in `FeaturePartition`.

Source resolution reuses `resolve_data_path`: explicit `data_root`, otherwise
`THESISFEDREC_DATA_ROOT`, otherwise repository-local `data/raw`. The movies
path must be relative and must not escape that root, including via symlinks.
The builder reads the bytes once, hashes those same bytes with SHA-256, then
decodes Latin-1. It supports LF, CRLF, and a final record without a newline,
removing only record terminators rather than normalizing fields or tokens.

Every source row, including metadata-only movies, must have exactly three
`MovieID::Title::Genres` fields, a positive ASCII-decimal ID without signs or
whitespace, a nonblank title, and nonempty, unique, exact-case known genre
tokens separated by `|`. Empty sources, blank/malformed records, duplicate
parsed MovieIDs (even identical rows), and unknown/empty/repeated tokens fail;
record errors include line numbers. Leading zeros do not create a separate
MovieID. The current `item_id_map` is revalidated: raw and processed IDs must
be built-in integers (not bool, strings, floats, or NumPy scalars), raw IDs
must be positive, and processed IDs must be a bijection onto `range(N)`.
Metadata-only movies are validated and counted but never extend this namespace.
A valid empty namespace returns `(0, 18)` arrays only after validating a
nonempty source.

Deterministic metadata records builder/version, source-relative path,
Latin-1 encoding and field format, byte size/SHA-256, fixed vocabulary and
attribution, multi-hot encoding and row/column order, strict ID/token/duplicate/
missing-metadata policies, and coverage counts. `counts` includes source
records, unique metadata items, processed/matched/missing items, metadata-only
items, genre count, and positive entries. `item_mapping_sha256` hashes the
exact existing `base/items.jsonl` representation in memory: entries sorted
by processed ID, each serialized as
`json.dumps({"raw_id": raw_id, "processed_id": processed_id}, ensure_ascii=False, sort_keys=True) + "\n"`
with default separators, concatenated and encoded as UTF-8 including trailing
LF. An empty mapping hashes empty bytes. No cache writer is called. Source
record order or newline changes can change provenance without changing values.
This does not change the feature-partition access-assignment checksum.

Every call allocates fresh mutable arrays and metadata and leaves all input
objects unchanged on success and failure. `FeatureView` remains mutable.
Attachment is caller-controlled, using a collision-checked copied dictionary:

```python
from dataclasses import replace

view = build_ml1m_item_genre_view(
    base, movies_relative_path="ml-1m/ml-1m/movies.dat"
)
feature_views = dict(base.feature_views)
if view.name in feature_views:
    raise ValueError(f"feature view already exists: {view.name}")
feature_views[view.name] = view
derived = replace(base, feature_views=feature_views)
```

The base bundle's dictionary is unchanged. Existing views and all unchanged
base objects remain shallow-shared; this is not a deep-copy isolation boundary.
Default ratings preprocessing still creates no feature views. Neither this
builder nor attachment changes mappings, splits, sample ownership, models,
training, or evaluation. No cache or other artifact is written. Schema-`1.0`
cache saving rejects the derived bundle before creating its cache directory.

---

## 11. GraphView interface

`GraphView` is retained as a future interface. Graph builders are not part of the initial Phase 2 implementation.

```python
GraphView:
    name: str

    node_raw_ids: tuple[RawId, ...]
    node_type_ids: IntArray
    node_type_names: tuple[str, ...]

    edge_index: NumPy int64 array
    relation_ids: IntArray
    relation_names: tuple[str, ...]

    user_to_node: IntArray | None
    item_to_node: IntArray | None

    edge_weights: FloatArray | None
    node_features: NumPy array | None
    node_feature_names: tuple[str, ...]

    metadata: Metadata
```

Rules:

- graph node IDs are indices into `node_raw_ids`;
- `node_type_ids` contains one value per node;
- `edge_index` has shape `(2, number_of_edges)`;
- `relation_ids` contains one value per edge;
- relation and node-type IDs are contiguous from zero;
- user/item-to-node arrays use `-1` when no corresponding graph node exists;
- a PKGRec-compatible graph provides `item_to_node`;
- metadata records directedness, inverse edges, self-loops, deduplication, source splits, and construction rules;
- validation and test interactions do not enter a training graph unless an explicit experiment protocol permits and records this;
- PKGRec propagation, relation learning, and graph objectives remain method-owned.

---

## 12. ContinualView interface

`ContinualView` is retained as a future interface. Continual-view builders are not part of the initial Phase 2 implementation.

```python
ContinualStage:
    stage_id: int
    train_interaction_ids: IntArray
    valid_interaction_ids: IntArray
    test_interaction_ids: IntArray
    metadata: Metadata
```

```python
ContinualView:
    name: str
    criterion: str
    stages: tuple[ContinualStage, ...]
    metadata: Metadata
```

Rules:

- stage IDs are contiguous from zero;
- every stage references base interaction IDs;
- stages never create new user or item mappings;
- metadata records chronological boundaries, category/domain membership, or the controlled-shift definition;
- incremental versus cumulative stage training is explicit;
- repeated interaction IDs between cumulative stages are allowed only when declared;
- historical evaluation uses stored test IDs from earlier stages;
- party-specific stage data is computed by intersecting stage IDs with `PartyData` interaction IDs;
- no separate continual-HFL dataset is created;
- continual views remain orthogonal to `partition_kind`.

---

## 13. Serialization and cache layout

The complete target layout is retained as a future-compatible contract:

```text
data/processed/
└── <dataset_name>/
    └── <base_fingerprint>/
        ├── manifest.json
        ├── base/
        │   ├── interactions.npz
        │   ├── splits.npz
        │   ├── users.jsonl
        │   └── items.jsonl
        ├── features/                       # future
        │   └── <view_name>/
        │       ├── manifest.json
        │       ├── entity_ids.npy
        │       ├── values.npy or values.jsonl
        │       └── missing_mask.npy        # optional
        ├── graphs/                         # future
        │   └── <view_name>/
        │       ├── manifest.json
        │       ├── nodes.jsonl
        │       ├── edge_index.npy
        │       ├── relation_ids.npy
        │       ├── user_to_node.npy        # optional
        │       ├── item_to_node.npy        # optional
        │       ├── edge_weights.npy        # optional
        │       └── node_features.npy       # optional
        ├── continual/                      # future
        │   └── <view_name>/
        │       ├── manifest.json
        │       └── stages.npz
        ├── partitions/                     # later cache extension
        │   └── <partition_name>/
        │       ├── manifest.json
        │       └── parties/
        │           └── <party_id>.npz
        └── negatives/                      # later cache extension
            └── <sampling_name>/
                ├── manifest.json
                ├── train.npz
                ├── valid.npz
                └── test.npz
```

Rules:

- the base fingerprint is a SHA-256 digest of the schema version, raw-file checksums, and canonical preprocessing configuration;
- derived artifacts record the base fingerprint and their own construction configuration;
- paths contain dataset and view names rather than research-method names;
- arrays use non-pickle NumPy formats;
- metadata and mappings use UTF-8 JSON or JSONL;
- cache readers validate schema versions, base fingerprints, shapes, ID ranges, and checksums;
- rebuilding a derived view must not rewrite base artifacts.

The initial Phase 2 implementation only needs `manifest.json` and the `base/` artifacts. It must not implement feature, graph, continual, partition, or negative-sample cache trees until those caches are concretely needed.

Phase 4B1 does not change base-cache schema `1.0`. Loading a schema-`1.0` cache reconstructs a bundle with an empty `feature_views` collection. The schema-`1.0` writer accepts empty feature collections only and explicitly rejects a bundle containing canonical `FeatureView` payloads rather than silently dropping them. Persisting such payloads requires a future independently versioned derived-view artifact or a deliberate base-cache schema migration.

Phase 4B2 likewise adds no persistence. `FeaturePartition` instances and
their checksums are in-memory scenario objects; no base-cache or derived-cache
payload is introduced in this phase.
The first Phase 4B3 whole-view recipes are also in-memory only. They do not
change schema `1.0` or serialize non-empty feature collections.
The separately authorized ML1M item-genres builder in Section 10.3 is also
in-memory only; it does not change default preprocessing or cache behavior.

---

## 14. Reproducibility metadata

The base manifest records:

- data-contract schema version;
- dataset name and source version;
- raw file relative names, byte sizes, and SHA-256 checksums;
- canonical preprocessing configuration;
- preprocessing seed;
- parser or loader name;
- canonical source-record ordering policy;
- raw-ID normalization and mapping policy;
- filtering and minimum-interaction rules;
- duplicate-interaction policy;
- implicit-feedback conversion rule;
- timestamp parsing, unit, and timezone policy;
- split name, algorithm, tie-breaking rule, and exceptions;
- user, item, and interaction counts before and after filtering;
- train, validation, and test counts;
- checksums for mappings, interactions, and splits;
- code revision and dirty-worktree flag;
- relevant Python, NumPy, and PyYAML versions;
- every deviation from the thesis-wide protocol.

Partition and sampling metadata record, when applicable:

- base fingerprint;
- construction configuration;
- all random seeds;
- artifact counts and checksums;
- HFL mode and explicit user-to-party assignment;
- per-party interaction, user, and item counts;
- explicit feature-access definitions and their assignment checksum;
- negative-sampling exclusion and replacement policies.

Future derived-view manifests additionally record feature encoding and missingness rules, graph construction and source-split rules, or continual-stage criteria and boundaries.

Creation time and machine hostname may be recorded for auditing but do not affect fingerprints.

---

## 15. Mandatory and optional fields

| Record | Mandatory fields | Optional values |
|---|---|---|
| `InteractionRecord` | All fields | None |
| `InteractionTable` | All five arrays | None |
| `InteractionSplits` | All three arrays | Arrays may be empty only under an explicitly documented non-default protocol |
| `DatasetBundle` | All fields | `partition_name` may be `None`; party and view containers may be empty |
| `PartyData` | All fields | Interaction, entity, or view arrays may be empty when partition semantics permit |
| `FeatureView` | All fields except mask | `missing_mask` |
| `FeatureAccess` | `view_name` | `entity_ids` and `feature_names`; `None` means all canonical rows or fields |
| `FeaturePartition` | All fields | `party_access` may be empty |
| `GraphView` | Names, nodes, node types, edges, relations, metadata | User/item mappings, weights, and node features |
| `ContinualStage` | All fields | Arrays may be empty only when documented |
| `ContinualView` | All fields | None |

Required containers are empty rather than `None`. `None` is reserved for genuinely unavailable optional values except for the explicit all-rows/all-fields semantics of `FeatureAccess`.

---

## 16. Initial Phase 2 implementation boundary

The initial Phase 2 implementation is limited to:

- columnar `InteractionTable`;
- deterministic user and item ID mapping;
- deterministic leave-one-out split;
- basic `DatasetBundle` construction;
- base cache and manifest;
- HFL `per_user` partitioning;
- HFL `grouped_round_robin` partitioning;
- shared negative-item sampling.

The initial Phase 2 implementation must not include:

- feature-view builders;
- graph-view builders;
- continual-view builders;
- VFL partition builders;
- HyFL partition builders;
- complex derived-view, partition, or negative-sample cache infrastructure;
- method-specific preprocessing;
- recommendation models or training logic.

Phase 4B1 implements the canonical `FeatureView` contract and its in-memory
`DatasetBundle` integration. Phase 4B2 implements explicit `FeatureAccess`
and `FeaturePartition` contracts, validation, and deterministic assignment
checksums. The first Phase 4B3 unit adds only complete selected-view access
and seeded single-view absence (Section 10.2). A separate unit adds the
standalone ML1M item-genres builder (Section 10.3). Other feature builders,
recipes, severity presets, and feature persistence remain future work.
`GraphView`, `ContinualView`, VFL, HyFL, and the complete derived-cache layout
remain documented interfaces only.

---

## 17. Minimal planned files under `src/data/`

The initial implementation should add only:

```text
src/data/
├── __init__.py
├── contracts.py
├── features.py
├── feature_scenarios.py  # first Phase 4B3 unit
├── ml1m_features.py      # standalone item-genres builder
├── preprocess.py
├── split.py
├── negative_sampling.py
├── partition.py
└── cache.py
```

Responsibilities:

- `contracts.py`: `InteractionTable`, basic bundle/party contracts, the
  concrete feature contracts, other reserved view contracts, and schema
  validation;
- `features.py`: bundle-aware `FeaturePartition` validation and deterministic
  access-assignment checksums; no scenario generation or persistence;
- `feature_scenarios.py`: complete access and seeded single-view absence over
  existing canonical views, shallow party-summary copies, and assignment counts;
- `ml1m_features.py`: strict ML1M movies metadata parsing and standalone
  multi-hot item-genres construction in the existing item namespace; no writes;
- `preprocess.py`: deterministic ID mapping and basic bundle construction from normalized interaction inputs;
- `split.py`: deterministic leave-one-out splitting;
- `negative_sampling.py`: shared negative-item sampling only;
- `partition.py`: HFL per-user and grouped-round-robin partitioning; no VFL/HyFL implementation;
- `cache.py`: base manifest and base-artifact serialization only.

Dataset-specific loader modules and files such as `features.py`, `graph.py`, and `continual.py` are added only when their implementation is separately authorized. Empty placeholder modules are unnecessary.
