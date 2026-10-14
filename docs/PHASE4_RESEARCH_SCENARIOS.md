# Phase 4 Research-Scenario Infrastructure Design

## 1. Purpose and scope

Phase 4 must add only the common scenario capabilities that are needed by the
four thesis research blocks:

- 1.1: heterogeneous information availability;
- 1.2: distributed and incomplete relation structure;
- 2.1: restricted parameter update and communication scope;
- 2.2: continual-stage execution.

These are four independent experimental dimensions. They are not four names
for one generic scenario object, and they are not research methods. The common
platform exposes data and execution conditions; DOI4, PKGRec, FedPEFT, and
GCPN retain their own learning algorithms.

The design follows three boundaries:

1. `DatasetBundle` owns canonical processed information aligned to the stable
   user, item, and interaction namespaces: interactions plus canonical
   `FeatureView` and `GraphView` objects when available.
2. Derived scenario objects describe which parts of that information a party
   or stage may use. `FeaturePartition`, `GraphPartition`, and
   `ContinualView` remain outside `DatasetBundle`; they do not remap IDs or
   copy the base dataset.
3. Method code owns fusion, graph learning, specialized optimization,
   adaptation, and collaboration algorithms.

The continual boundary is specifically:

```text
DatasetBundle -> ContinualView -> StageManager
```

`DatasetBundle` owns canonical data, `ContinualView` owns derived stage
membership/reference definitions, and `StageManager` owns execution
sequencing.

Phase 4A is design-only. It does not alter source code, the Phase 2B cache, or
the validated Phase 3 execution path.

## 2. Current Phase 3 foundation

The validated Phase 3 baseline already provides:

- deterministic ML1M parsing, ID mapping, temporal leave-one-out splitting,
  and negative sampling;
- an `InteractionTable` and `DatasetBundle` with stable bundle-wide IDs;
- `PartyData` with separate party, user, item, and interaction identities;
- deterministic `per_user` and `grouped_round_robin` HFL partitions;
- party-local user state and shared FedGMF state;
- synchronous selection, local training, shared-only weighted FedAvg, and
  shared HR/NDCG evaluation;
- one-YAML-per-run validation, launch-time Git provenance, communication
  accounting, and plain result artifacts;
- a synthetic full-stack test and a successful one-round real ML1M/L40 smoke.

The relevant existing data seams are deliberately incomplete:

- Phase 4B1 makes `DatasetBundle.feature_views` a concrete canonical contract,
  while `graph_views` remains a `dict[str, object]` placeholder;
- the current empty `DatasetBundle.continual_views` placeholder must not be
  promoted into canonical dataset content; a future `ContinualView` is a
  separately constructed scenario object;
- `PartyData.feature_view_names` and `graph_view_names` remain coarse visible-
  view summaries; Phase 4B2 adds separate feature row/field access, while
  detailed graph node/edge visibility remains future work;
- the base cache serializes only interactions, splits, and mappings under
  schema `1.0`;
- the Phase 3 runner is explicitly FedGMF/HFL and initializes one run from
  scratch.

These limitations identify the required extension seams. They do not justify
a registry, plugin system, generic trainer, or protocol hierarchy.

## 3. Unified research capability matrix

| Thesis block | Independent experimental dimension | Common data/scenario responsibility | Shared execution responsibility | Method-specific responsibility |
|---|---|---|---|---|
| 1.1 heterogeneous information | Feature fields, modalities, view availability, and completeness | Canonical `FeatureView` values plus an explicit party feature partition | Deliver only the views/slices visible to each selected party and record the resolved assignment | Missing-view handling, representation learning, fusion, and adaptive collaboration |
| 1.2 distributed relations | Local relation subgraphs and overlap | Canonical `GraphView` plus explicit party node/edge/relation visibility | Deliver each party's allowed subgraph and account for graph-related payloads when a method sends them | GNNs, graph completion, relation propagation, graph objectives, and graph aggregation |
| 2.1 parameter update scope | Trainable and communicated subsets of model parameters | None | Apply a resolved `ParameterUpdatePolicy`, form the actual payload, and count actual tensors | PEFT modules and the scientific rule that selects a method-specific subset |
| 2.2 continual learning | Stage-indexed interaction views and state transitions | Derived `ContinualView` outside the bundle, referencing base interaction IDs | `StageManager` sequencing and explicit warm-start/reset handoff | Retention, replay, regularization, expansion, and other continual algorithms |

Statistical quantity skew, popularity skew, and label skew may later be
additional sample-partition scenarios. They are not the core abstraction for
block 1.1, whose subject is heterogeneous information.

## 4. Block 1.1: FeatureView and information scenarios

### 4.1 Canonical FeatureView

Phase 4B1 makes `FeatureView` a concrete public dataclass in
`src/data/contracts.py`:

```python
FeatureView:
    name: str
    entity_kind: Literal["user", "item", "interaction"]
    entity_ids: IntArray
    modality: str
    values: numpy.ndarray | tuple[str, ...]
    feature_names: tuple[str, ...]
    missing_mask: numpy.ndarray | None
    metadata: Metadata
```

The contract already reserved in `DATA_CONTRACT.md` is sufficient and should
not be generalized further. Its invariants are:

- `entity_ids` are unique IDs from exactly one bundle-wide namespace;
- row `i` of `values` and `missing_mask`, when present, corresponds to
  `entity_ids[i]`;
- absence from `entity_ids` means that the entity has no row in the view;
- `missing_mask` represents element-level source missingness; numeric zero is
  never implicitly missing;
- `feature_names` name tabular columns and are empty for unstructured values;
- encoding, normalization, tokenization, source fields, and missingness rules
  are recorded in metadata;
- one logical modality or field group is one named view. A single wide,
  weakly typed feature bag is not the target representation.

`DatasetBundle.feature_views` becomes `dict[str, FeatureView]`. The key must
equal `FeatureView.name`. These are canonical available views, not a party's
permission to observe them.

### 4.2 Separate feature partition

Whole-view names alone cover simple missing-view scenarios, but cannot
represent different parties seeing different rows or fields of the same
canonical view. Phase 4B2 implements the minimum additional scenario
contracts:

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

`None` means all rows or all fields of the referenced canonical view. Absence
of a `FeatureAccess` entry means that party cannot use that view. This gives
three independent levels of visibility without copying feature arrays:

- view visibility through `view_name`;
- entity-row visibility through `entity_ids`;
- tabular-field visibility through `feature_names`.

The partition is a derived experiment scenario object outside
`DatasetBundle`. It references both bundle view names and actual `party_id`
values. It does not own interactions, parties, or feature values.

The public access mapping is read-only over a defensive copy, with immutable
descriptor/field tuples. Selected entity IDs are stored as immutable bytes
after constructor validation. Each `entity_ids` read returns an independent
one-dimensional NumPy `int64` view over the same bytes, without copying the
buffer or exposing a stored ndarray header. Changing a returned view's shape
or dtype cannot affect other reads; item writes and re-enabling writes are
rejected. `None` remains the distinct full-access sentinel.
Constructor-input mutations cannot change the stored access assignments.
Names and metadata remain excluded from the unchanged assignment checksum;
metadata is not required to be immutable.

Bundle-aware validation receives the allowed `PartyData` set explicitly, so
party IDs may be non-contiguous and never imply user IDs. It rejects unknown
parties, views, rows, or named fields and duplicate party/view descriptors.
Unstructured views without named fields support whole-payload access only.

The assignment checksum canonicalizes party IDs, view names, entity IDs, and
selected field names. It is independent of mapping/descriptor input order and
does not include the scenario name or metadata. Phase 4B2 adds no persistence
or scenario generator.

`PartyData.feature_view_names` remains as the coarse, backward-compatible
summary of views visible to a party. When a `FeaturePartition` is supplied,
the names in `PartyData` must exactly match that party's `FeatureAccess`
names. This rule prevents two conflicting ownership descriptions. A Phase 3
party with an empty tuple continues to work unchanged.

### 4.3 Supported scenario shapes

This representation supports the intended scenarios directly:

| Scenario | Representation |
|---|---|
| Complete homogeneous information | Every party has full access to the same selected feature views |
| Single-view missing | A named view is absent from selected parties' access tuples |
| Multi-view missing | Multiple named views are absent, with the full assignment recorded |
| Future named heterogeneity levels | After researcher approval, a name may map to explicit quantitative parameters and a fully resolved per-party assignment |
| Unbalanced view availability | View assignment frequencies differ across parties |
| Cross-party view overlap | The same view, row, or field appears in multiple parties' access descriptors |
| Partial entity coverage | `entity_ids` selects the visible rows of a canonical view |
| Partial tabular fields | `feature_names` selects visible columns without changing sample ownership |

Intrinsic source missingness remains in `FeatureView.missing_mask`.
Scenario-induced hiding remains in `FeaturePartition`. Conflating the two
would make it impossible to distinguish a naturally absent value from a
controlled experimental intervention.

Randomized scenario builders must be deterministic under an explicit seed and
validate all references. The first unit returns resolved assignments and their
checksums in memory; saving them with experiments is a later integration. Phase
4B does not define or implement thesis-level labels such as `H0`-`H3`,
`mild`, `moderate`, or `severe`. If later approved, each label must map to
explicit quantitative scenario parameters; a label alone is never sufficient
provenance.

No 1.1 fusion, imputation, contrastive objective, collaboration rule, or
adaptive aggregation belongs in the data layer.

### 4.4 First Phase 4B3 implementation boundary

Only two in-memory whole-view recipes are implemented in
`src/data/feature_scenarios.py` and exported by `src.data`:

- `build_complete_feature_scenario(bundle, parties, *, view_names, name)`;
- `build_single_view_absence_scenario(bundle, parties, *, view_names,
  missing_view, missing_party_count, seed, name)`.

Each returns `(FeaturePartition, tuple[PartyData, ...], dict[str, object])`.
The full validation rules and statistics keys are in `DATA_CONTRACT.md`
Section 10.2. Inputs are valid canonical views plus explicit sample-owning
parties; no synthetic-only flag or hard-coded dataset is used. Tests use
synthetic in-memory bundles and non-contiguous, grouped-user parties.

The recipes sort actual party IDs numerically and selected view names
lexically. Complete access uses no randomness. Absence uses a local NumPy
generator to choose exactly `K` positions without replacement in the sorted
party list. Input ordering and global RNG state do not influence assignments,
and construction does not change global RNG state. Valid seeds are Python
integers in `[0, 2**32)` even for zero/all-party absence; counts range from zero
through the supplied party count. Booleans are not valid seeds/counts/IDs.

Visible access always has both selectors `None`; hidden views have no
descriptor. All parties retain mapping entries, including empty tuples.
Returned party records replace prior coarse feature summaries from the recipe,
not from an intersection with old summaries. Only `feature_view_names` changes;
arrays and metadata are shallow-shared rather than deeply copied or frozen.
The input parties and `bundle.parties` remain untouched, and the returned
parties remain outside the bundle. Existing validation checks these new
summaries against the immutable access assignment.

Metadata records recipe/version and canonical parameters, including the seed
for absence. The existing checksum remains access-configuration identity only,
excluding names/metadata. Zero absence matches complete access assignments,
summaries, statistics, and checksum. All-party absence and featureless parties
are legitimate; zero-row/source-missing views may still be assigned.
Statistics describe counts of assigned party/view pairs, not source coverage,
semantic overlap, privacy guarantees, or recommendation benefits.

No ML1M canonical feature builder, feature persistence, row/field slicing
recipe, graph work, configuration expansion, or training integration belongs
to this unit. Real ML1M preprocessing still produces empty canonical feature
collections; feature construction is a separate prerequisite for real-data
use. Schema-`1.0` base caches continue to reject non-empty feature collections.
Persisting reusable features later needs the versioned-artifact or migration
decision in Section 8.2. FedGMF still consumes IDs only, so scenario construction
alone is not evidence of a recommendation-quality effect. Multi-view absence,
unbalanced quotas, partial overlap, and thesis severity labels remain deferred.

## 5. Block 1.2: GraphView and partial-graph scenarios

### 5.1 Canonical GraphView

Raw relation structure belongs in the common data layer. `GraphView` should
become the concrete typed form already reserved by the data contract:

```python
GraphView:
    name: str
    node_raw_ids: tuple[RawId, ...]
    node_type_ids: IntArray
    node_type_names: tuple[str, ...]
    edge_index: numpy.ndarray
    relation_ids: IntArray
    relation_names: tuple[str, ...]
    user_to_node: IntArray | None
    item_to_node: IntArray | None
    edge_weights: FloatArray | None
    node_features: numpy.ndarray | None
    node_feature_names: tuple[str, ...]
    metadata: Metadata
```

Graph-local `node_id` is the row index in `node_raw_ids`, from zero through
`N-1`; a redundant `node_ids` array is unnecessary. Graph-local `edge_id` is
the column index in `edge_index`, from zero through `E-1`. Canonical node and
edge order must therefore be deterministic.

Required invariants include:

- `edge_index` has shape `(2, E)` and contains in-range graph node IDs;
- `node_type_ids` has length `N`, while `relation_ids` has length `E`;
- type and relation IDs are contiguous and index their name tuples;
- `user_to_node` and `item_to_node`, when present, align to the complete
  recommendation namespaces and use `-1` for no mapping;
- weights and node-feature rows align to edges and nodes respectively;
- metadata records directedness, inverse-edge policy, self-loops,
  deduplication, source files, construction seed, and source interaction
  splits;
- validation/test interactions never leak into a training graph unless an
  explicitly approved protocol records that deviation.

`DatasetBundle.graph_views` becomes `dict[str, GraphView]`, keyed by the
view's name. A bundle may carry multiple relation sources, but graph-learning
state never belongs in `GraphView`.

### 5.2 Party-local incomplete graphs

`PartyData.graph_view_names` can say which canonical graphs a party knows,
but it cannot express `G_c subset of G`. The minimum derived contracts are:

```python
GraphAccess:
    view_name: str
    node_ids: IntArray | None
    edge_ids: IntArray | None
    relation_type_ids: IntArray | None

GraphPartition:
    name: str
    party_access: dict[int, tuple[GraphAccess, ...]]
    metadata: Metadata
```

`None` means the complete corresponding canonical set. Explicit `node_ids`
preserve visible isolated nodes. Every selected edge must have visible
endpoints and an allowed relation type. The descriptor references canonical
indices and does not duplicate node or edge data.

This representation controls independently:

- edge visibility;
- relation-type visibility;
- visible isolated nodes;
- node and edge overlap between parties;
- complete, local-only, and structurally incomplete graphs.

As with features, `PartyData.graph_view_names` remains a coarse summary and
must match the names in the party's detailed access entries. The resolved
`GraphPartition`, seed, overlap statistics, visible counts, and checksum are
run metadata. Two parties may reference the same graph and overlapping edge
IDs without implying shared sample ownership.

Graph preprocessing may eventually live in `src/data/graph.py`; deterministic
partial-graph scenario construction may live beside it in
`src/data/graph_scenarios.py`. GNNs, completion, propagation, graph losses,
learned relation representations, and graph-aware aggregation stay under the
1.2 method directory.

## 6. Block 2.1: ParameterUpdatePolicy

### 6.1 Minimal resolved policy

The common object should be a concrete immutable value, not a base class or
plugin interface:

```python
ParameterUpdatePolicy:
    name: str
    parameter_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    communicated_parameter_names: tuple[str, ...]
    metadata: Metadata
```

For a resolved model instance:

- all names are unique and stable;
- trainable and communicated names are subsets of `parameter_names`;
- the communicated update subset is a subset of the trainable subset;
- frozen names are exactly `parameter_names - trainable_parameter_names`;
- a helper computes total, trainable, frozen, and communicated `numel` and
  bytes from an explicit `dict[str, Tensor]`;
- the policy contains names and metadata, not tensor copies.

Model-specific code constructs the resolved policy explicitly because only
that code knows its parameter namespace. There is no policy registry or
generic initializer factory. A method-specific subset is permitted, but the
method returns an ordinary resolved `ParameterUpdatePolicy` that shared
accounting can inspect.

### 6.2 Existing FedGMF compatibility

FedGMF can expose one explicit qualified tensor mapping:

```text
local.user_embeddings
shared.item_embeddings
shared.prediction_weight
shared.prediction_bias
```

Under `full`, all four are trainable, while only the three `shared.*` tensors
are communicated. “Full” therefore means every parameter allowed by the
existing ownership boundary; it does not upload party-local user embeddings.

An `embedding_only` baseline can train local user embeddings and shared item
embeddings, communicate the shared item embeddings, and freeze the prediction
layer once that baseline is explicitly approved. Future FedNCF code can
expose its normal `named_parameters()` with stable prefixes and resolve the
same policy structure. This avoids forcing the current state-based FedGMF
implementation into a new model hierarchy.

### 6.3 Training, aggregation, and communication

The policy determines scope only. It does not choose the loss, optimizer,
aggregation rule, initialization, or local-state ownership.

For a restricted update, local training optimizes only resolved trainable
tensors and `ClientUpdate.shared_state` carries only the resolved
communicated mapping. FedAvg operates on that communicated reference subset;
the server retains its other state unchanged and merges the aggregate back
explicitly. The existing full FedGMF path remains numerically unchanged.

Communication accounting must continue measuring actual payload tensors. The
policy's parameter statistics explain the scope but do not replace byte
counting. Downloads and uploads may differ and should be counted separately
when a later method actually sends different mappings. Any one-time initial
distribution of a frozen backbone must be reported separately if it is part
of the experimental protocol.

No LoRA, adapter, quantization, hashing, or FedPEFT algorithm belongs in the
policy implementation.

## 7. Block 2.2: ContinualView and StageManager

### 7.1 ContinualView

The existing reserved data contract is sufficient:

```python
ContinualStage:
    stage_id: int
    train_interaction_ids: IntArray
    valid_interaction_ids: IntArray
    test_interaction_ids: IntArray
    metadata: Metadata

ContinualView:
    name: str
    criterion: str
    stages: tuple[ContinualStage, ...]
    metadata: Metadata
```

Stages reference bundle-wide interaction IDs and never contain copied
interactions or new user/item mappings. Stage IDs are contiguous and ordered.
Metadata records temporal boundaries, incremental versus cumulative training,
coverage/overlap rules, source splits, and construction seed.

The first implemented builder should be temporal increment only. Client
evolution and feature/information evolution are future scenario builders, not
extra fields required now. Party-stage data is obtained by intersecting a
stage's IDs with each party's existing interaction IDs, so continual learning
remains independent of HFL/VFL/HyFL.

`ContinualView` is a derived experiment-scenario object outside
`DatasetBundle`. A selected view is immutable experiment input, not mutable
runner progress. It references the bundle's canonical interaction IDs and is
passed separately to `StageManager`.

This placement allows the same base bundle and base cache to support multiple
valid protocols, including 5-stage or 10-stage temporal views, equal-duration
or equal-interaction views, and future client-evolution or
information-evolution views. Constructing one of these protocols must not
require a different base `DatasetBundle`.

### 7.2 StageManager

`StageManager` belongs in shared execution infrastructure, likely
`src/federation/stages.py`. It is a small sequencer over one validated
`ContinualView`; it is not a continual-learning method.

Its responsibilities are limited to:

- yielding stages in increasing `stage_id` order;
- deriving per-party stage interaction IDs by explicit intersection;
- tracking stage and global-round indices for logs and seed derivation;
- requesting stage-boundary evaluation according to configuration;
- recording resolved stage counts, boundaries, and checksums.

It must not optimize parameters, store exemplars, compute a continual loss,
expand a model, or decide how knowledge is retained.

### 7.3 Runner state handoff

The current Phase 3 function initializes shared/local state and completes all
rounds in one call. Continual execution will eventually need a narrow split:

1. initialize the model and party-local state once;
2. build the selected `ContinualView` and `StageManager`;
3. run the existing HFL round procedure on stage-restricted party data;
4. return shared state, local states, and explicit method state;
5. pass those values into the next stage for `warm_start`, or reinitialize
   them for a configured `reset` baseline;
6. evaluate and record per-stage and historical-stage metrics.

The runner remains the owner of state handoff. `StageManager` only supplies
stage order and data. The existing optimizer-reset-per-party-round behavior is
unchanged unless a later method explicitly owns persistent optimizer state.

A monotonic global round index can preserve the existing selection and local
sampling seed schemes across stages. Stage-specific candidate construction
needs a documented stage component in its seed. These seed rules must be
frozen and tested in Phase 4F before any continual research method uses them.

The non-continual Phase 3 path should remain direct; it need not be wrapped in
a synthetic one-stage manager merely for conceptual uniformity.

## 8. Data-contract evolution plan

### 8.1 Ownership by layer

| Concept | Owner | Rationale |
|---|---|---|
| Interactions, splits, ID maps | Base `DatasetBundle` | Stable dataset identity used by every method |
| Canonical `FeatureView` values | `DatasetBundle.feature_views` | Dataset information aligned to base namespaces |
| Canonical raw `GraphView` structure | `DatasetBundle.graph_views` | Raw relations are shared data, not learned method state |
| Continual stage membership/reference definitions | Derived `ContinualView` outside the bundle | Multiple experiment protocols may reference the same canonical interaction IDs |
| Sample ownership | `PartyData` interaction/user/item fields | Already stable and independent of feature visibility |
| Coarse visible-view names | Existing `PartyData` name tuples | Backward-compatible summary and simple whole-view scenarios |
| Detailed feature visibility | Derived `FeaturePartition` outside the bundle | Experiment-specific party access, not canonical data |
| Detailed graph visibility | Derived `GraphPartition` outside the bundle | Experiment-specific `G_c`, not canonical graph data |
| Parameter update scope | Model/execution `ParameterUpdatePolicy` | It concerns model tensors, not processed data |
| Stage sequencing | `StageManager` | Runtime sequencing over immutable stage data |
| Fusion, GNN, PEFT, continual algorithm | Method directories | Scientific behavior must remain independent |

### 8.2 Backward-compatible steps

1. Replace the `feature_views` and `graph_views` `dict[str, object]`
   annotations with their concrete canonical view types and add validation.
   Empty dictionaries remain valid defaults.
2. Add the feature and graph view dataclasses without changing
   `InteractionTable`, mappings, splits, or PartyData's existing mandatory
   fields. Do not populate or serialize the current `continual_views`
   placeholder; Phase 4F must pass a derived `ContinualView` separately.
3. Keep detailed feature/graph partitions as separate derived objects. Do not
   insert them into the base cache or duplicate them into every party's
   metadata.
4. Load the existing Phase 2B cache exactly as today; the reconstructed bundle
   simply has empty view dictionaries.
5. When derived-artifact persistence is needed, use an independently
   versioned manifest that records the base fingerprint, construction config,
   checksums, and its own artifact-schema version. Continual protocols, when
   persisted, are scenario artifacts rather than canonical bundle content.
6. Save resolved scenario assignments and checksums with experiment results,
   not in the immutable base manifest.

The existing ML1M base-cache schema may remain `1.0` during the initial Phase
4B/4C synthetic contract work only if all of the following remain true:

- existing Phase 3 cache payload semantics are unchanged;
- new `FeatureView` and `GraphView` fields are optional and default-empty for
  existing bundles;
- `FeatureView` and `GraphView` payloads are not silently persisted into the
  old schema.

When canonical feature or graph data is first persisted as a reusable dataset
artifact, that work must either introduce an independently versioned
derived-view artifact referencing the base fingerprint or deliberately
migrate the base-cache schema. It must not add serialized fields while still
calling the payload schema `1.0`.

A base schema migration is also required if a later phase changes a mandatory
base artifact, base ID/split semantics, or the required base manifest
structure incompatibly. It must occur in the phase that implements the new
serialization, with an explicit migration or a clear rejection of `1.0`, and
must not be performed preemptively during synthetic contract work.

## 9. Configuration plan

One complete experiment YAML continues to define exactly one run. The new
dimensions should be independent small sections rather than one polymorphic
scenario schema:

```yaml
information:
  scope: D+X+G
  feature_views: [item_text, item_brand]
  graph_views: [item_relations]

feature_scenario:
  name: complete
  level: complete
  seed: 2026

graph_scenario:
  name: full_graph
  level: complete
  seed: 2026

parameter_update:
  policy: full

continual:
  enabled: false
  view_name: null
  state_transition: none
```

The sections have deliberately narrow meanings:

- `information` selects `D`, `D+X`, `D+G`, or `D+X+G` and names the canonical
  feature/graph views included in the run;
- `feature_scenario` selects a deterministic feature-partition recipe;
- `graph_scenario` selects a deterministic graph-partition recipe;
- `parameter_update` selects a model-supported resolved update policy;
- `continual` selects a separately resolved derived `ContinualView` and an
  explicit state transition; `view_name` is not a `DatasetBundle` key.

When continual execution is enabled, a minimal form is:

```yaml
continual:
  enabled: true
  view_name: temporal_5_stage
  state_transition: warm_start
```

The example `complete` values mean literal full visibility, not a thesis
severity category. Recipe-specific fields and named heterogeneity levels
should be added only when their quantitative definitions are scientifically
frozen. The saved result must include the fully resolved party access/stage
assignments and checksums; a future level label alone is not sufficient
provenance.

The current Phase 3 validator and its reference YAML remain unchanged. A
Phase 4 validator may require the new sections for Phase 4 scenario runs while
continuing to accept the frozen Phase 3 schema through its existing entry
point. There is no implicit conversion of an old run into a new scenario.

Configuration consistency checks include:

- `D` requires empty feature and graph view lists;
- `D+X`/`D+X+G` require named existing feature views;
- `D+G`/`D+X+G` require named existing graph views;
- scenario assignments may reference only selected views and existing party
  IDs;
- `parameter_update.policy` must be supported explicitly by the selected
  model/method;
- enabled continual execution requires a resolvable derived `ContinualView`
  and a supported transition; disabled execution requires `view_name: null`
  and `state_transition: none`.

## 10. HFL, VFL, and HyFL compatibility

Current implementation remains HFL-first. The proposed contracts avoid two
invalid equalities:

```text
party_id != user_id
sample partition != feature/graph partition
```

Feature and graph partitions key access by `party_id` but independently
reference bundle-wide entity IDs and graph-local IDs. They do not infer
access from a party's user list. Therefore:

- HFL can use disjoint sample ownership with homogeneous or heterogeneous
  views;
- future VFL can use overlapping/aligned samples with disjoint or overlapping
  view access and explicit `owns_labels`;
- future HyFL can vary both sample ownership and information access;
- graph overlap does not imply interaction overlap;
- continual stages intersect with whichever sample partition a protocol uses.

The minimum future VFL/HyFL seam is a protocol-specific executor that consumes
the same `PartyData`, `FeaturePartition`, and `GraphPartition` values but owns
its own alignment and representation/gradient exchange. No unused VFL or
HyFL classes are needed now, and VFL must never be routed through the HFL
FedAvg loop.

## 11. FedNCF placement

The future common FedNCF backbone belongs in `src/models/fedncf.py`, beside
rather than inside `src/models/fedgmf.py`. FedGMF remains the minimal reference
model; FedNCF becomes the initial common main backbone once its exact
architecture is approved.

FedNCF should reuse:

- bundle IDs, parties, negative sampling, and ranking evaluation;
- deterministic seed utilities and explicit initialization;
- the existing HFL selection/round boundary and `ClientUpdate` payload;
- party-local versus shared ownership rules where its approved architecture
  calls for them;
- result, runtime, and communication reporting;
- the later resolved `ParameterUpdatePolicy`.

It should not inherit from a new recommendation base class, use a model
registry, or absorb method-specific PEFT/graph/continual logic. The runner can
use explicit named FedGMF and FedNCF paths. Shared labeled-example or state
helpers should be extracted only after both concrete implementations need the
same behavior.

The exact FedNCF variant, layer sizes, user/item ownership, initialization,
and baseline hyperparameters remain researcher decisions and must be frozen
before Phase 4D implementation.

## 12. Experiment matrices and multi-seed aggregation

The invariant remains:

```text
one complete experiment YAML -> one independent run directory
```

A future matrix layer may take lists over dataset, information/scenario,
model, method, and seed, then materialize complete ordinary experiment YAMLs
and invoke the existing CLI once per entry. It must not inject hidden runtime
overrides or change the saved executed configuration.

A separate result aggregator may group completed runs only after validating
their configuration identities and report per-metric count, mean, sample
standard deviation, and contributing run paths/seeds. It must reject missing,
failed, duplicated, or scientifically non-comparable runs rather than silently
averaging them.

Scheduling, GPU allocation, retries, distributed launch, and experiment
tracking services are not part of this boundary. Phase 4G should begin with
matrix expansion and offline aggregation only.

## 13. Amazon Unified Pipeline path

Amazon remains blocked because the available directory does not contain the
required raw Industrial and Scientific review and metadata sources. Legacy
`.pth` objects must not be treated as raw thesis data.

Once the raw sources are available, one Amazon Unified Pipeline can populate
the same contracts without changing their semantics:

- reviews become the base `InteractionTable`, mappings, and temporal splits;
- title, description, category, brand, product features, price, and image
  references become separate item `FeatureView` values with explicit
  missingness;
- `also_buy`, `also_view`, `bought_together`, category hierarchy, and
  brand-item relations become one or more typed `GraphView` values;
- timestamps or category/domain slices can define `ContinualView` stages;
- feature and graph scenario builders operate on those canonical views in the
  same way as for any other dataset.

The Amazon pipeline must map interaction ASINs and metadata ASINs into the
single bundle item namespace once, record unmatched coverage, and then build
all views against processed item IDs. It must not create method-specific DOI4,
PKGRec, or GCPN preprocessing branches.

## 14. Baseline capability matrix

This matrix describes future coverage and dependencies only; none of these
baselines is implemented by Phase 4A.

| Block | Baseline family | Required common capability | Method-specific or later dependency |
|---|---|---|---|
| 1.1 | Local-only | FeatureView, resolved feature partition, per-party evaluation | No federation aggregation |
| 1.1 | FedAvg/common | Same views and feature partition plus existing HFL/FedAvg | A feature-capable common backbone |
| 1.1 | Simple available-feature | Explicit available views/fields with no invented missing values | Simple fixed combination rule to be approved |
| 1.1 | Simple missing-view handling | Missing masks and controlled view absence | Baseline imputation/fusion rule to be approved |
| 1.1 | Later FedProx/SCAFFOLD | All above plus reference aggregation/control-state implementations | Separate later baseline phase |
| 1.2 | No graph | `information.scope: D` or `D+X`; no GraphView access | Existing interaction backbone |
| 1.2 | Local graph only | Canonical GraphView and party GraphPartition | A basic graph recommender operating only on `G_c` |
| 1.2 | Full-graph/oracle | Full GraphAccess for each evaluated party | Clearly labeled non-private oracle baseline |
| 1.2 | Basic federated partial graph | Overlapping/incomplete GraphAccess and actual graph payload accounting | A separately approved simple collaboration rule |
| 2.1 | Full update | Resolved full policy and actual payload accounting | Existing FedGMF, later FedNCF |
| 2.1 | Embedding-only | Stable parameter names and restricted optimizer/payload | Exact embedding set frozen per backbone |
| 2.1 | Later LoRA/FedPEFT | Method-created modules represented through the same resolved policy | Phase 4+ FedPEFT migration; no plugin framework |
| 2.2 | Static | Continual disabled | Existing single-run behavior |
| 2.2 | Fine-tune | ContinualView, StageManager, warm-start state | No retention algorithm |
| 2.2 | Retrain | Same stages with explicit reset per stage | Fair initialization/seed protocol |
| 2.2 | Joint/oracle | Union of all stage training IDs available together | Clearly labeled upper-bound data access |

## 15. Proposed Phase 4 implementation sequence

The recommended B-H order is retained. In particular, `ParameterUpdatePolicy`
follows FedNCF so the abstraction is validated against two concrete backbones
rather than designed around FedGMF alone.

### Phase 4B — FeatureView and heterogeneous-information scenarios

- Likely files: `src/data/contracts.py`, `src/data/features.py`,
  `src/data/feature_scenarios.py`, `src/data/__init__.py`, focused config
  validation, tests, and `docs/DATA_CONTRACT.md`.
- Dependencies: stable base IDs and existing `PartyData`.
- Deliver: `FeatureView`, `FeatureAccess`, `FeaturePartition`, deterministic
  whole-view/row/field visibility, explicit missing views/rows/fields,
  validation, checksums, and deterministic synthetic scenario construction.
- Acceptance: aligned shapes/IDs and missing masks are enforced; same seed
  reproduces the resolved assignment; different parties can have overlapping
  features while sample ownership remains unchanged; Phase 3 tests pass.
- Out of scope: DOI4, learned fusion/imputation, VFL execution, Amazon data,
  and thesis-level `H0`-`H3` or mild/moderate/severe semantics unless their
  quantitative definitions are explicitly approved by the researcher.
- Cache impact: schema `1.0` remains only under the conditions in Section 8.2;
  persisted canonical features require a versioned derived artifact or a
  deliberate base-cache migration.

### Phase 4C — GraphView and partial-graph scenarios

- Likely files: `src/data/contracts.py`, `src/data/graph.py`,
  `src/data/graph_scenarios.py`, exports, tests, config validation, and the
  data-contract document.
- Dependencies: Phase 4B's established canonical-view/scenario separation,
  but not feature values themselves.
- Deliver: `GraphView`, `GraphAccess`, `GraphPartition`, canonical synthetic
  graph construction, explicit node/edge/relation visibility, deterministic
  partial-graph construction, party subgraph validation, overlap statistics,
  and checksums.
- Acceptance: node/relation mappings validate; every visible edge has visible
  endpoints; parties can overlap; withheld edges/relations are inaccessible;
  no validation/test leakage enters a training graph.
- Out of scope: PKGRec, GNNs, graph completion/propagation/aggregation, and
  thesis-level graph-severity categories until their quantitative definitions
  are explicitly approved by the researcher.
- Cache impact: schema `1.0` remains only under the conditions in Section 8.2;
  persisted canonical graphs require a versioned derived artifact or a
  deliberate base-cache migration.

### Phase 4D — public FedNCF backbone

- Likely files: new `src/models/fedncf.py`, model exports, explicit runner and
  config branches, and CPU model/local-training/integration tests.
- Dependencies: Phase 3 HFL/evaluation/result infrastructure; researcher-frozen
  FedNCF architecture and ownership decisions.
- Deliver: one explicit common FedNCF implementation alongside FedGMF.
- Acceptance: deterministic initialization/training, party/user separation,
  shared/local payload correctness, finite two-round CPU run, unchanged
  FedGMF path.
- Out of scope: PFedRec, PEFT, graph or heterogeneous fusion methods.
- Cache impact: none.

### Phase 4E — ParameterUpdatePolicy

- Likely files: new `src/models/parameter_update.py`, focused changes to
  FedGMF/FedNCF local training and HFL payload merging, result statistics,
  config validation, and tests.
- Dependencies: two concrete backbones from Phases 3 and 4D.
- Deliver: resolved full and embedding-only policies, frozen/trainable checks,
  communicated-subset payloads, parameter statistics, and actual byte counts.
- Acceptance: frozen tensors do not change; trainable tensors can change;
  uploads contain exactly allowed names; aggregation retains noncommunicated
  server state; counts match tensors; full FedGMF behavior is unchanged.
- Out of scope: LoRA, adapters, quantization, hashes, or FedPEFT algorithms.
- Cache impact: none.

### Phase 4F — ContinualView and StageManager

- Likely files: `src/data/contracts.py`, `src/data/continual.py`, new
  `src/federation/stages.py`, focused runner/result/config extensions, tests,
  and data/experiment protocol documentation.
- Dependencies: stable runner state handoff; it need not depend on feature or
  graph algorithms.
- Deliver: deterministic temporal-increment views, stage intersections,
  warm-start/reset execution, stage/global-round metadata, and per-stage
  evaluation records.
- Boundary: `ContinualView` is built from and references a `DatasetBundle`,
  but remains a separate scenario input to `StageManager`; it is never
  canonical bundle content.
- Acceptance: stages reference only base IDs; temporal boundaries and overlap
  rules validate; warm-start exactly preserves prior state; reset reproduces
  initialization; continual and federation dimensions remain separable.
- Out of scope: EWC, LwF, replay, GCPN, client evolution, feature evolution.
- Cache impact: no base schema bump; a persisted continual view, if needed,
  uses its own manifest referencing the base fingerprint.

### Phase 4G — experiment matrix and multi-seed aggregation

- Likely files: a small script or utility for matrix expansion, a separate
  result aggregation utility, schemas/examples, and tests.
- Dependencies: stable complete experiment YAML and result artifacts.
- Deliver: deterministic expansion into ordinary run configs and offline
  mean/std summaries with run provenance.
- Acceptance: expansion is reproducible; generated configs validate normally;
  aggregation rejects incomparable/missing/duplicate seeds and matches
  hand-calculated statistics.
- Out of scope: GPU scheduling, parallel execution, retries, services, or
  dashboards.
- Cache impact: none.

### Phase 4H — first cross-scenario baseline configs and smoke tests

- Likely files: explicit experiment YAMLs, synthetic scenario fixtures,
  end-to-end smoke tests, and validation documentation.
- Dependencies: the individually accepted capabilities above and researcher-
  approved baseline definitions.
- Deliver: small auditable configs crossing information scope, heterogeneity,
  update scope, and static/continual execution where scientifically valid.
- Acceptance: each config maps to one run; resolved scenario checksums and
  parameter counts are recorded; CPU synthetic tests pass; any real-data
  smoke is opt-in and labeled non-publication.
- Out of scope: publication-scale sweeps or thesis claims.
- Cache impact: none unless an earlier phase separately approved a derived
  view cache.

## 16. Researcher decisions still required

These decisions are scientific and are intentionally not made by this design:

1. The actual 1.1 feature/modalities used on each dataset and the precise
   recipes/coverage thresholds called mild, moderate, and severe.
2. The 1.2 relation set, directed/inverse-edge conventions, and the exact
   controlled definitions of partial-graph severity.
3. The FedNCF architecture variant, layer sizes, initialization, ownership
   boundary, and reference hyperparameters.
4. The exact parameter membership of `embedding_only` for each backbone and
   whether one-time frozen-backbone distribution is part of reported
   communication.
5. Temporal stage boundaries, incremental versus cumulative training,
   historical evaluation matrix, and the final forgetting definition.
6. Acquisition, version, licensing, and checksums of the raw Amazon review and
   metadata files.
7. Which later FedProx, SCAFFOLD, and FedPEFT references are required for the
   final thesis comparison.

Items 1-5 must be frozen in the corresponding implementation phase before
real experiment configurations are accepted. They do not block implementing
the structural validation contracts on synthetic fixtures.

## 17. Explicit non-goals

Phase 4A does not authorize:

- source-code or cache changes;
- any DOI4, PKGRec, FedPEFT, or GCPN algorithm;
- ordinary Non-IID quantity/popularity skew as the definition of block 1.1;
- GNNs, graph completion, relation propagation, or graph aggregation;
- LoRA, adapters, quantization, hashing, or a PEFT plugin framework;
- EWC, LwF, replay, model expansion, or continual-learning methods;
- VFL or HyFL execution, or simulated VFL through the HFL FedAvg loop;
- PFedRec;
- Amazon construction from legacy `.pth` artifacts;
- matrix scheduling, GPU allocation, distributed launch, or result services;
- publication-scale runs or changes to the frozen Phase 3 reference.

The next authorized implementation should therefore begin with Phase 4B only:
concrete `FeatureView` validation and deterministic heterogeneous-information
scenario primitives on synthetic data, parameterized explicitly and without
unapproved thesis-level severity labels.
