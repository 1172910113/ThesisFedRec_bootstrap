# Phase 3 Minimal HFL Baseline Design

## 1. Current reusable foundation

Phase 3 starts from the audited Phase 2B baseline at commit `3bea6fe`
(`v0.2.2-phase2b-audit`). The baseline has 21 passing tests and already
provides the following reusable pieces.

### Data layer

`src/data/` owns the processed-data contract and must remain the only owner
of dataset parsing, remapping, splitting, party construction, and negative-item
sampling.

- `contracts.py` defines `InteractionTable`, `InteractionSplits`,
  `DatasetBundle`, and `PartyData`.
- `loaders.py` resolves `THESISFEDREC_DATA_ROOT` and parses MovieLens-1M.
- `preprocess.py` performs the approved filtering, deduplication, implicit
  conversion, ID mapping, and base-bundle construction.
- `split.py` implements deterministic temporal leave-one-out splitting.
- `partition.py` builds `per_user` and `grouped_round_robin` HFL parties.
- `negative_sampling.py` returns target-safe negative item IDs and does not
  construct evaluation candidates.
- `cache.py` saves and validates the base processed cache.

Phase 3 must consume these public objects and functions. It must not create a
second mapping, split, partition representation, or negative sampler.

### Configuration, seeds, and logging

- `src/utils/config.py::load_yaml()` safely loads one YAML mapping. There is
  no composition system, override language, or schema framework.
- `src/utils/seed.py::set_random_seed()` seeds Python, NumPy, and PyTorch for
  best-effort reproducibility. It does not promise deterministic CUDA
  execution.
- `src/utils/logging.py::get_logger()` provides a minimal stdout logger.
- `configs/datasets/ml1m.yaml` contains the approved Phase 2B data choices.
- `configs/methods/fedavg.yaml` contains only sparse federation/evaluation
  defaults. Its `method: fedavg` label must not be interpreted as meaning
  that FedAvg is a research method; in Phase 3 it is an aggregation rule.

Phase 3 should keep `load_yaml()` and add only explicit validation of a
single, self-contained experiment mapping. It should not add Hydra, recursive
config merging, registries, or command-line overrides that make the saved YAML
differ from the executed configuration.

### Tests and entry points

The existing pytest suite covers package import, YAML loading, seed replay,
preprocessing, cache integrity, partition invariants, and negative sampling.
Fixtures in `tests/conftest.py` already provide a small processed bundle.

There is no training/evaluation CLI. `scripts/check_env.py` is only an
environment diagnostic. Phase 3 therefore needs one narrow experiment entry
point rather than extending an existing runner.

### Known Phase 2B limitations relevant to Phase 3

The audit found the data values and invariants ready for Phase 3. Two metadata
limitations affect experiment bookkeeping:

1. a loaded cache is internally validated, but the caller remains responsible
   for selecting the cache matching the requested raw checksum and
   preprocessing configuration;
2. `PartyData` makes assignments derivable, but no object owns
   partition-wide metadata.

Phase 3 should validate the selected bundle metadata against the experiment
configuration where values are supplied and save partition-wide metadata with
the run. It should not change `DATA_CONTRACT.md` or add a derived-cache
system.

## 2. Phase 3 scope

Phase 3 implements one end-to-end baseline:

```text
MovieLens-1M
  -> approved implicit-feedback base bundle
  -> HFL per_user partition
  -> FedNCF / FedGMF local training
  -> persistent party-local user embeddings
  -> synchronous weighted FedAvg over shared parameters
  -> shared HR@10 and NDCG@10 evaluation
```

The runner should also accept the already implemented
`grouped_round_robin` partition so its party semantics can be tested,
although `per_user` is the initial thesis setting. Full and partial party
participation are in scope. CPU execution and one explicitly selected CUDA
device are in scope.

The following are out of scope:

- Amazon integration;
- VFL and HyFL execution;
- multi-GPU or distributed training;
- asynchronous clients, simulated dropout, compression, privacy, or secure
  aggregation;
- personalization machinery beyond the required FedGMF party-local user
  embeddings;
- DOI4, PKGRec, FedPEFT, GCPN, or any new research method;
- early stopping, best-validation selection, or hyperparameter search;
- a registry, plugin system, dependency-injection container, trainer
  hierarchy, or generic model factory.

## 3. Execution flow

The CLI first reads and validates the input YAML because the first data action
depends on it. The following numbered steps are the observable experiment
flow requested for Phase 3. Step 3 creates the *resolved* run configuration by
adding only derived identities and counts; it does not choose new scientific
defaults.

| Step | Operation | Owner |
|---:|---|---|
| 1 | Load an explicitly selected base cache, or explicitly preprocess ML1M from its configured relative raw path. Never silently switch between the two modes. Validate dataset name, split policy, preprocessing settings, and an expected raw checksum when configured. | Existing `src.data`; called by the CLI/runner |
| 2 | Call `build_hfl_partition()` with `per_user` initially. Keep the returned `PartyData` objects separate from recommendation user IDs. | Existing `src.data.partition` |
| 3 | Form the resolved experiment mapping: exact input fields plus dataset/cache identity, actual party count, partition checksum, code revision, and runtime identity. Write it before training. | `src.federation.runner` and result writer |
| 4 | Initialize the FedNCF/FedGMF shared state (item embeddings and prediction layer) and one persistent FedGMF local state per party containing embeddings for exactly that party's `user_ids`. | `src.models.fedncf` |
| 5 | Select party IDs without replacement. Full participation returns every party; partial participation uses a deterministic round seed. | `src.federation.hfl` |
| 6 | In ascending selected-party order, combine a copy of the current shared state with that party's persistent local state, resample four negatives per positive for each local epoch, and optimize both shared and local FedGMF parameters. | FedGMF local-training function in `src.models.fedncf` |
| 7 | Retain the updated user embeddings in the local-state mapping keyed by `party_id`. Upload only the updated shared item/prediction state, positive-interaction count, party ID, and scalar training summary. | FedGMF local training and `src.federation.hfl` |
| 8 | Compute standard weighted FedAvg over the uploaded shared states, weighted by each party's positive training-interaction count. A failed selected party fails the round. | `src.federation.fedavg` |
| 9 | Replace the server's shared state with the aggregate. Local user states remain unchanged except for parties trained in step 6; optimizer state is neither aggregated nor retained. | `src.federation.hfl` |
| 10 | Every round, evaluate validation targets using the current shared state plus each target user's owning party-local embedding. On the final round, evaluate test targets once in the same way. Aggregate only scalar target outcomes. | `src.evaluation.ranking` |
| 11 | Append round metrics and write final metrics/runtime/partition records; if enabled, save a simulator checkpoint containing shared state, all local states, and restoration metadata. | `src.utils.results` and `src.federation.runner` |

The runner should be a readable loop over rounds. It should not dispatch by
registry or hide the flow behind callbacks. Phase 3 may explicitly reject any
`federation.type` other than `hfl` with `NotImplementedError`; this leaves
the future protocol boundary clear without creating unused VFL classes.

## 4. Proposed module boundaries

### Common infrastructure

- Existing `DatasetBundle` and `PartyData` remain the dataset and
  client-data representations. No new `Client` class is needed.
- `federation/hfl.py` owns deterministic party selection and the synchronous
  HFL round loop. It maintains the in-memory
  `dict[party_id, FedGMFLocalState]` but never interprets `party_id` as a
  recommendation user ID.
- `federation/fedavg.py` owns a small `ClientUpdate` dataclass and weighted
  aggregation of shared parameter state. It does not receive local user
  embeddings and does not know about users, datasets, or FedNCF.
- `federation/runner.py` owns run-level sequencing, validation/evaluation
  calls, and result-writer calls. It contains no local optimization math.
- `evaluation/ranking.py` owns candidate construction, scoring orchestration,
  HR/NDCG definitions, and aggregation across targets.
- `utils/results.py` writes plain YAML/JSON/CSV/text artifacts.
- Existing `utils/config.py`, `utils/seed.py`, and `utils/logging.py` are
  extended or reused rather than duplicated.

### Baseline-specific code

`models/fedncf.py` owns the concrete FedNCF/FedGMF shared module, the small
FedGMF-specific local-state value, labeled-example construction, scoring, and
local optimization. Keeping these together in one initial file avoids
introducing a generic personalization or trainer package. The local function
consumes shared data contracts, updates the supplied local state, and returns
a common `ClientUpdate` containing only the federated/shared payload.

Later research methods may provide their own model and local-update code under
`src/methods/<method_name>/`. If a second method demonstrates that a local
training helper is truly shared, it can then be moved to common
infrastructure. No abstract model base class or method registry is justified
in Phase 3.

The HFL loop makes two explicit direct calls:

```python
client_update, updated_local_state = train_local_fedgmf(
    shared_state,
    local_state,
    bundle,
    party,
    round_index,
    config,
    seed,
    device,
)
shared_state = aggregate_fedavg(shared_state, client_updates)
```

The runner imports these named functions directly; it does not accept injected
callbacks. There is no runtime discovery, registration, service container, or
base-class hierarchy. Later HFL methods can reuse the selection, round,
evaluation, and result functions and provide their own explicit method logic
when concretely needed. Methods with non-HFL execution are not forced through
this loop.

## 5. Party/user semantics

The three identities remain independent:

- `user_id` indexes a recommendation user in `DatasetBundle.user_id_map`
  and is an input to FedNCF;
- `item_id` indexes a recommendation item and is the other FedNCF input;
- `party_id` (also called client ID in round logs) identifies a federated
  data holder and selects one `PartyData` object.

Interaction ownership is defined only by a party's
`train_interaction_ids`, `valid_interaction_ids`, and
`test_interaction_ids`. For every interaction, the model obtains `user_id`
and `item_id` from the bundle's `InteractionTable`; it never substitutes
`party_id` for either value.

In the current `per_user` partition, each party owns exactly one user's
interactions. The numeric `party_id` and `user_id` happen to coincide
because of the current stable construction, but this is incidental. Training
code must still index the model using
`bundle.interactions.user_ids[interaction_ids]`, not `party.party_id`.

The party's local FedGMF state is indexed by the processed user IDs in
`PartyData.user_ids`. In `grouped_round_robin`, one party owns several
users and therefore holds one local embedding row for each assigned user.
`party_id` selects the owning local-state object; it is never an embedding
row index.

This rule supports grouped HFL without changing the model's identity
semantics. It also avoids embedding an HFL-only identity assumption into the
evaluator. Future VFL/HyFL protocols will need different execution logic, but
they can continue to use explicit party ownership and bundle-wide entity IDs.

## 6. Model and local-training interface

### FedNCF / FedGMF model boundary

The first concrete FedNCF baseline is explicitly named **FedNCF / FedGMF**.
Configuration and result metadata use `fedncf_fedgmf`, so its results cannot
later be confused with FedMLP or FedNeuMF.

The model has no hidden MLP or NeuMF branch:

1. look up the party-local 8-dimensional embedding for each processed
   `user_id`;
2. look up the shared 8-dimensional embedding for each processed `item_id`;
3. take their element-wise product;
4. map that 8-dimensional product to one unnormalized logit with a shared
   linear GMF prediction layer.

The prediction-layer weight and bias are shared. Larger logits rank ahead of
smaller logits. Training uses mean `BCEWithLogitsLoss` over positive labels
(`1`) and sampled negative labels (`0`), so scoring does not apply a
sigmoid.

The scoring boundary is explicit about local state:

```python
logits = score_fedgmf(
    shared_model,
    local_state,
    user_ids,
    item_ids,
)
```

`user_ids` and `item_ids` are one-dimensional `torch.long` tensors of
the same shape. `local_state` must contain every requested processed
`user_id`; lookup never uses `party_id` as a user or row index.

### Shared and local parameters

The server-owned/federated state contains only:

- the complete shared item-embedding table;
- the GMF prediction-layer weight;
- the GMF prediction-layer bias.

The frozen `ncf_gmf` initialization protocol is:

- party-local user embeddings use `Normal(mean=0, std=0.01)`;
- federated item embeddings use `Normal(mean=0, std=0.01)`;
- the prediction weight uses LeCun uniform, implemented for its
  `(1, embedding_dim)` shape as
  `Uniform(-sqrt(3 / embedding_dim), sqrt(3 / embedding_dim))`;
- the prediction bias is exactly zero.

Shared and party-local initialization use their existing explicit deterministic
seed streams. No initialization depends on global framework RNG state or party
iteration order. Persistent local user embeddings and all shared tensors use
the same floating dtype.

Each party owns a persistent FedGMF local state containing only the user
embeddings for the processed user IDs assigned to that party. The minimum
representation is a FedGMF-specific value such as:

```python
FedGMFLocalState:
    user_ids: one-dimensional tensor of processed user IDs
    user_embeddings: floating tensor shaped (len(user_ids), 8)
```

The simulator keeps `dict[int, FedGMFLocalState]` keyed by `party_id`.
States are initialized once under the run seed, survive communication rounds
and non-participating rounds, and are never sent to the server or passed to
FedAvg. This is not a generic personalization framework and does not add local
state to the Phase 2B data contract.

Complete-state FedAvg over a global table of all user embeddings is invalid for
this HFL setting. A per-user party updates only its own user vector; averaging
complete tables would mix that updated row with thousands of unchanged copies
from parties that do not own it, diluting the update. It would also make a
server-held all-user table part of the communication payload, contradict the
approved local ownership. Grouped parties have the same issue for the subset
of users they own. Local user rows therefore persist only at their owning
party and are never treated as globally shared model parameters.

### Local training behavior

For one selected party and round, local training receives:

- the current shared item/prediction `state_dict`;
- that party's persistent `FedGMFLocalState`;
- the unmodified `DatasetBundle`;
- one `PartyData` selected by `party_id`;
- the frozen optimization settings;
- the selected device.

It performs these explicit actions:

1. create a local working module from a copy of the current shared state and
   the party's local user embeddings;
2. read only `party.train_interaction_ids`;
3. for every local epoch, call the unchanged Phase 2B
   `sample_negative_items()` with four negatives per positive, uniform
   sampling, no replacement, and known positives excluded from train,
   validation, and test;
4. derive that sampler seed from exactly
   `(run_seed, round_index, party_id, local_epoch)` using a documented
   integer `SeedSequence`, never Python `hash()`;
5. expand each positive into `(user_id, item_id, 1)` and its sampled
   `(user_id, negative_item_id, 0)` records, then deterministically shuffle
   and use all records (`drop_last=False`);
6. optimize with a fresh Adam optimizer for this selected party/round using
   learning rate `0.001`, batch size `256`, weight decay `0`, and mean
   `BCEWithLogitsLoss`;
7. persist the resulting user-embedding rows back into that party's local
   state;
8. return a `ClientUpdate` containing `party_id`, a CPU copy of only the
   updated shared state, `num_positive_interactions`, and mean loss.

`num_positive_interactions` is exactly
`len(party.train_interaction_ids)`. Empty local training data is an explicit
error. Optimizer state is neither persisted nor communicated.

Returning a complete shared state is simpler to validate than returning a
delta. It is still only the federated subset of FedGMF, not the complete
shared-plus-local model. The HFL loop feeds each returned shared state into a
streaming FedAvg accumulator and can then release that working model/update.

Training negative sampling and evaluation candidate construction remain
different operations:

- local training calls the Phase 2B sampler and consumes only its negative
  item IDs;
- the evaluator independently calls the same sampler for held-out targets,
  then inserts the target positive and controls candidate ordering;
- neither caller changes the sampler's target exclusion, known-positive
  exclusion, sorted output, capacity checks, or replacement semantics.

The baseline has one local epoch, so each selected party samples once per
round. The epoch component remains in the seed derivation because the HFL
implementation supports more than one configured local epoch without changing
the resampling rule.

## 7. FedAvg and participation semantics

For selected client updates `i = 1..m`, uploaded shared-state tensor
`theta_i`, and positive local weight `n_i`, standard weighted FedAvg is:

```text
theta_global_next = sum_i(n_i * theta_i) / sum_i(n_i)
```

The approved weight `n_i` is exactly the number of positive training
interactions owned by that participating party. It is not multiplied by the
negative ratio, number of generated labels, number of batches, or local
epochs. FedAvg therefore remains independent of negative-sampling settings.

Aggregation rules are exact and intentionally narrow:

- every returned state must have exactly the server shared state's keys,
  shapes, and dtypes;
- the only Phase 3 FedGMF keys are the shared item embeddings and prediction
  layer; local user embeddings in an update are an error;
- floating tensors are accumulated on CPU in a fixed ascending-`party_id`
  order, using a higher-precision accumulator for float16/float32 and casting
  once to the global dtype;
- non-floating tensors are not averaged; every client value must be identical
  to the starting global value, otherwise aggregation raises an error;
- optimizer state, local loss, and counters are not model parameters and are
  not aggregated;
- after aggregation the server shared model is replaced by the resulting
  shared state;
- zero updates, zero/negative weights, missing keys, and non-finite tensors
  fail explicitly.

The non-floating rule is sufficient for an initial FedNCF without batch
normalization and avoids inventing a policy for incompatible integer buffers.

The actual total number of clients is derived from `len(parties)` rather
than duplicated in configuration. `partition.num_parties` is only supplied
for partition modes such as `grouped_round_robin` that require it.

For each round:

- require `0 < client_fraction <= 1`;
- full participation selects every party without consuming randomness;
- partial participation selects without replacement using NumPy's explicit
  generator and a seed derived from `(run_seed, round_index)`;
- choose `max(1, ceil(client_fraction * total_parties))` parties;
- sort selected party IDs before local execution and aggregation;
- record the selected IDs and count for the round.

There is no availability, dropout, retry, or stale-update model in Phase 3.
The initial Phase 3 reference run uses 20 communication rounds, full
participation (`client_fraction: 1.0`), and one local epoch. Partial
participation remains supported and deterministic infrastructure, but it is
not the reference configuration.

## 8. Evaluation design

Evaluation is centralized shared infrastructure and operates on the global
shared model in `eval()` mode under `torch.no_grad()`. Scoring also
requires the persistent local state belonging to each target user's owning
party.

### Candidate construction

For each validation or test target interaction:

1. obtain its processed `user_id` and held-out positive `item_id` from the
   bundle;
2. find the unique owning party through `PartyData.user_ids`, then retrieve
   that processed user's row from the party's `FedGMFLocalState`;
3. call the unchanged `sample_negative_items()` for exactly 99 negative
   items, with a fixed split-specific evaluation seed, no replacement, and
   known positives excluded from train, validation, and test;
4. combine the one target positive with those 99 negative item IDs;
5. score every `(user_id, candidate_item_id)` pair using the current shared
   item/prediction state and the owning party's local user embedding.

Validation and test candidate maps are generated once at run initialization
and remain fixed for the entire run. Their sampling configuration and checksum
are recorded. The current sampler's empty/insufficient-pool error is
propagated. Candidate construction remains evaluator-owned, and no evaluation
path modifies the Phase 2B negative sampler.

The evaluator needs access to local states inside the simulator, but it does
not upload those embeddings into the federated model. Only scalar
target-level hit and discounted-gain outcomes need to cross the evaluation
aggregation boundary.

### Metric definitions

The evaluation unit is one held-out positive interaction. Under the approved
leave-one-out split there is exactly one validation and one test target per
retained user.

- Candidates rank by descending logit.
- Let `r` be the one-based rank of the held-out positive.
- `HR@K = 1` when `r <= K`, otherwise `0`.
- With one relevant item, `NDCG@K = 1 / log2(r + 1)` when `r <= K`,
  otherwise `0`.
- Dataset metrics are the arithmetic mean across target interactions. Because
  ML1M currently has one target per user, this is also a macro-average across
  users.
- Parties do not receive equal secondary weight. For grouped HFL, pool all
  target outcomes and average by target, so a small party is not weighted the
  same as a large party.

Candidates rank by descending score with ascending processed `item_id` as
the deterministic tie break; input order never decides a tie.

Validation uses only validation target IDs and is evaluated after every
communication round for monitoring. It does not select a checkpoint. There is
no early stopping or best-validation model selection. Test uses only test
target IDs and is evaluated once after round 20 using the final shared state
and final persistent local states. Later thesis work may adopt a separately
approved model-selection protocol without changing this Phase 3 baseline.

## 9. Configuration

Phase 3 should use one self-contained YAML file per experiment. `load_yaml()`
loads it and a small explicit validator rejects missing keys, unknown enum
values, invalid ranges, and unknown top-level sections. No scientific field is
silently defaulted. The runner writes the resolved mapping back to the result
directory before training.

A minimal shape is:

```yaml
experiment:
  name: ml1m_fedncf_fedgmf_fedavg

dataset:
  name: ml1m
  input: cache                 # exactly cache or raw
  cache_path: data/processed/ml1m/<base_fingerprint>
  ratings_path: ml-1m/ml-1m/ratings.dat
  expected_raw_sha256: 506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f
  min_interactions_per_user: 3
  duplicate_policy: keep_latest
  implicit_feedback: all_observed
  split: leave_one_out_temporal

federation:
  type: hfl
  partition: per_user
  num_parties: null
  aggregation: fedavg
  rounds: 20
  client_fraction: 1.0
  local_epochs: 1

model:
  name: fedncf_fedgmf
  architecture: gmf
  embedding_dim: 8
  initialization: ncf_gmf
  user_embedding_scope: party_local
  item_embedding_scope: federated
  prediction_layer_scope: federated

optimization:
  optimizer: adam
  learning_rate: 0.001
  batch_size: 256
  weight_decay: 0
  train_negatives_per_positive: 4
  train_negative_replacement: false
  train_negative_resampling: per_local_epoch

evaluation:
  k: [10]
  negatives_per_positive: 99
  replacement: false
  excluded_splits: [train, valid, test]
  candidate_reuse: fixed_per_run
  ranking_direction: descending_score
  tie_breaker: item_id_ascending
  validation_interval: 1
  checkpoint_selection: final_round
  early_stopping: false
  test_timing: final_only

runtime:
  seed: 2026
  device: cuda:0
  results_root: results
  save_checkpoint: false
```

For `dataset.input: cache`, `cache_path` is required and
`ratings_path` is provenance only; for `dataset.input: raw`,
`ratings_path` is resolved relative to `THESISFEDREC_DATA_ROOT` and
`cache_path` is the explicit save destination. The absolute server data path
is never stored in source code. The runner must not silently rebuild an absent
cache in cache mode.

The values in this YAML are the frozen Phase 3 reference semantics, not
placeholders: 20 rounds, full participation, one local epoch, FedGMF with
8-dimensional embeddings and `ncf_gmf` initialization, Adam at `0.001`, batch
size 256, zero weight decay, four training negatives per positive, and 99
evaluation negatives.
Partial participation and other local-epoch counts remain supported
infrastructure but define a different run configuration.

One base seed is sufficient in the user-facing configuration. Deterministic
training-negative seeds use
`numpy.random.SeedSequence([run_seed, round_index, party_id, local_epoch])`
exactly. Deterministic client selection and fixed validation/test candidate
construction use separately documented integer seed derivations. No seed
derivation uses Python's process-randomized `hash()`. The resolved run
metadata records every scheme. `set_random_seed()` is still called once,
with its existing best-effort CUDA qualification.

CPU is required for the normal automated suite. A real run selects exactly one
device such as `cuda:0`; all tensors for one local job use that device. No
`DataParallel`, distributed process group, automatic use of all three L40s,
or multi-GPU configuration belongs in Phase 3. The target environment remains
Python 3.11.16, PyTorch 2.7.1+cu126, CUDA runtime 12.6, and three NVIDIA L40
GPUs; Phase 3 intentionally uses at most one of them.

## 10. Result and logging design

Use one non-overwriting run directory, for example:

```text
results/<experiment_name>/seed_<seed>_<UTC-run-id>/
├── config.yaml
├── metrics.json
├── round_metrics.csv
├── runtime.json
├── partition.json
├── run.log
└── checkpoint.pt              # optional
```

The existing protocol's core files keep their documented meanings:

- `config.yaml`: fully resolved executed configuration, including seed;
- `metrics.json`: final validation/test HR@10 and NDCG@10 plus common
  efficiency totals when available;
- `round_metrics.csv`: one row per round containing round index,
  selected-party count, selected-party IDs or a checksum plus a side record,
  positive training-interaction count, weighted mean training loss, validation
  HR@10/NDCG@10, elapsed seconds, and separate download/upload/total bytes;
- `runtime.json`: start/end times, duration, Python/PyTorch/CUDA versions,
  selected device/GPU, dataset/cache manifest checksum, git commit and dirty
  flag, and best-effort reproducibility qualification;
- `run.log`: the existing logger's human-readable progress;
- `checkpoint.pt`: when enabled, a versioned simulator snapshot containing
  the server shared FedGMF state, every `party_id -> FedGMFLocalState`,
  completed round, partition/cache identities, seed information, and the
  metadata needed to validate restoration.

`partition.json` is the minimal run-level owner for the missing Phase 2B
partition metadata. It records partition name/type/mode, configured partition
seed, actual party count, complete `party_id -> user_ids` assignment,
per-party train/validation/test/user/item counts, base-cache identity, and a
checksum of the canonical assignment. This avoids duplicating global metadata
into every `PartyData` and does not alter the data contract or cache layout.

Checkpointing local user states is solely for reproducible restoration of the
single-process simulator. It does not mean that those states were transmitted
to or owned by the federated server during the protocol.

Communication accounting is frozen as:

- download bytes: shared item-embedding and prediction-layer tensors logically
  sent to each selected training party;
- upload bytes: the updated shared tensors logically returned by each selected
  training party;
- total bytes: download bytes plus upload bytes.

Local user embeddings, optimizer state, scalar metrics, result artifacts, and
checkpoint serialization are not communication payloads. In particular,
saving all local states in `checkpoint.pt` contributes zero communication
bytes. Evaluation-only simulator access to local state does not turn that
state into an upload.

Writes should use standard-library JSON/CSV and PyYAML, with temporary files
renamed into place for final summaries where practical. No database,
dashboard, or external tracking dependency is needed.

## 11. Proposed file tree

Only the following additions or focused extensions are planned for Phase 3:

```text
configs/
└── experiments/
    └── ml1m_fedncf_fedgmf_fedavg.yaml
scripts/
└── run_experiment.py
src/
├── evaluation/
│   └── ranking.py
├── federation/
│   ├── fedavg.py
│   ├── hfl.py
│   └── runner.py
├── models/
│   └── fedncf.py
└── utils/
    ├── config.py              # focused extension
    └── results.py
tests/
├── test_ranking.py
├── test_fedavg.py
├── test_hfl.py
├── test_fedncf.py
├── test_results.py
├── test_phase3_integration.py
└── test_phase3_ml1m_smoke.py
```

File responsibilities:

- `configs/experiments/ml1m_fedncf_fedgmf_fedavg.yaml`: one auditable,
  complete and unambiguously named Phase 3 run configuration; no
  config-composition engine is required.
- `scripts/run_experiment.py`: parse only `--config`, call the runner, and
  return a nonzero exit code on failure. It contains no training logic.
- `src/evaluation/ranking.py`: shared candidate construction, model scoring,
  HR@K, NDCG@K, and target-level aggregation.
- `src/federation/fedavg.py`: `ClientUpdate` and standard weighted
  aggregation of shared/federated state only.
- `src/federation/hfl.py`: deterministic party selection and the synchronous
  HFL round loop, including the in-memory party-to-local-state mapping.
- `src/federation/runner.py`: the explicit run-level sequence and only the
  wiring between data, HFL, model, evaluation, and results.
- `src/models/fedncf.py`: the approved FedNCF/FedGMF shared module,
  `FedGMFLocalState`, labeled-example construction, local-state-aware
  scoring, and local training function.
- `src/utils/config.py`: retain `load_yaml`; add narrow Phase 3
  key/type/range validation rather than a new configuration framework.
- `src/utils/results.py`: create a non-overwriting run directory, write the
  plain result artifacts, and optionally serialize/restore both shared and
  simulated local state.
- unit/integration test files keep each boundary independently reviewable; the
  real-data smoke has an explicit opt-in gate.

Package `__init__.py` files may export the small public functions after their
modules exist. Empty VFL, HyFL, protocol, trainer, factory, or registry files
must not be added.

## 12. Testing plan

### Unit tests (CPU)

`test_ranking.py` should verify hand-calculated HR/NDCG values for positive
ranks inside/outside K, descending scores, the approved tie rule, candidate
order independence, exactly 99 fixed negatives without replacement,
all-known-positive exclusion, local-state lookup through the owning party,
and target-level rather than party-level averaging.

`test_fedavg.py` should verify weighting by positive interaction counts,
independence from negative/batch/epoch counts, dtype/shape/key validation,
fixed handling of identical non-floating shared state, rejection of differing
non-floating state, zero weights, non-finite values, and rejection of any
local user-embedding key in an uploaded state.

`test_hfl.py` should verify full participation, partial-participation count,
same-seed replay, round-dependent deterministic selections, no duplicates, and
use of `party_id` rather than `user_id` for selection. It should also
verify that local states persist across rounds and remain unchanged while a
party is not selected.

`test_fedncf.py` should verify input/output shape, finite local loss, no
mutation of the server shared input state, training examples drawn only from
the party's train IDs, deterministic per-epoch negative resampling, target-safe
negative use, positive-interaction count, optimizer reset, persistence of
updated local user rows, and exclusion of those rows from `ClientUpdate`.
A fixture should make a party own multiple users and make its ID differ from
its users' IDs.

`test_results.py` should round-trip a simulator checkpoint and exactly
restore the shared state, every party's user IDs/embedding rows, completed
round, and required metadata. It should also verify that checkpoint byte size
and local-state tensors never enter communication counters.

Existing Phase 2B tests remain unchanged and passing.

### Synthetic integration test (CPU)

`test_phase3_integration.py` should construct a tiny valid bundle with enough
unseen items, build parties, run at least two rounds, aggregate updates, and
evaluate fixed validation/test candidate maps. Assertions should include:

- completion without filesystem or GPU dependencies;
- finite shared and local model states and metrics in `[0, 1]`;
- exactly conserved party-owned interaction IDs;
- no target positive among sampled negatives;
- identical selected parties, final shared state, local states, metrics,
  communication counts, and structured result content for two runs with the
  same seed;
- user embeddings change only in their owning party state and never appear in
  a server update;
- validation occurs each round and test occurs once after the final round;
- correct operation with a grouped party so `party_id == user_id` cannot be
  assumed.

The test should use very small dimensions and batches and complete in seconds.

### Real MovieLens-1M smoke (opt-in)

`test_phase3_ml1m_smoke.py` should require an explicit environment flag such
as `THESISFEDREC_RUN_REAL_SMOKE=1`; otherwise it skips. It should use the
real cache/raw root without modifying dataset files, one selected device, one
round, a small deterministic set of selected parties, and a small
deterministic set of evaluation targets. The result must be marked
`smoke_only: true` and must not be reported as a thesis measurement.

The smoke verifies cache/load, partition, FedGMF shared/local initialization,
local update, shared-only FedAvg, local-state-aware evaluation, communication
accounting, and artifact writing. CPU is allowed for portability; on the
target server the explicit real-run check should use one device such as
`cuda:0`. Publication-scale client participation or evaluation is not part
of this test.

## 13. Researcher decisions required

No scientific or experimental decision now blocks Phase 3 implementation.
The FedGMF architecture, local/federated parameter boundary, objective,
optimizer, hyperparameters, negative sampling, aggregation weight, schedule,
evaluation protocol, validation/test usage, and communication accounting are
all frozen above.

Remaining choices are non-blocking implementation details: exact Python names,
the version field used in simulator checkpoints, atomic-write mechanics, and
the deterministic tensor-initialization helper. They must remain simple,
tested, recorded where relevant, and consistent with the run seed; none
requires a new architecture or researcher decision.

## 14. Implementation sequence

Each task should be one small reviewable change with focused tests, followed by
the full suite and `git diff --check`.

1. **Phase 3A — encode the frozen configuration.** Add the single experiment
   YAML and explicit validation to `utils/config.py`. No training yet.
2. **Phase 3B — shared ranking evaluation.** Implement approved candidate
   construction and HR/NDCG in `evaluation/ranking.py`, with hand-calculated
   CPU tests. Reuse the Phase 2B sampler unchanged.
3. **Phase 3C — FedAvg and party selection.** Implement `ClientUpdate`,
   positive-interaction-weighted shared-state aggregation, and deterministic
   HFL selection with unit tests, including rejection of local user state.
4. **Phase 3D — FedGMF and local training.** Implement the shared FedGMF
   module, minimal persistent `FedGMFLocalState`, and one explicit
   local-update function with CPU tests for ownership, epoch-seeded negative
   sampling, persistence, and shared-only uploads.
5. **Phase 3E — orchestration and results.** Implement the explicit HFL round
   loop, local-state-aware evaluator wiring, high-level runner, result writer,
   optional shared-plus-local simulator checkpoint, and narrow CLI. Add
   partition-level run metadata and deterministic artifact tests.
6. **Phase 3F — end-to-end validation.** Add the fast synthetic CPU integration
   test, then the opt-in small real-ML1M smoke on one selected CUDA device.

Only after Phase 3F passes should a separate, researcher-approved task define
and run publication-scale ML1M experiments. Phase 4 and thesis method
implementation remain outside this design.
