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
  -> FedNCF local training
  -> synchronous weighted FedAvg
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
- personalization or client-local parameter persistence;
- DOI4, PKGRec, FedPEFT, GCPN, or any new research method;
- early stopping or hyperparameter search until their semantics are approved;
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
| 4 | Initialize one global FedNCF model from bundle user/item counts and the approved model configuration. Obtain its initial full `state_dict`. | `src.models.fedncf` |
| 5 | Select party IDs without replacement. Full participation returns every party; partial participation uses a deterministic round seed. | `src.federation.hfl` |
| 6 | In ascending selected-party order, reset a local model to the current global state, form labeled examples only from that party's training interaction IDs, and run the configured local epochs. | FedNCF local-training function in `src.models.fedncf` |
| 7 | Return each party's full model state, party ID, local-example count, and scalar training summary. Feed it immediately to the aggregation accumulator rather than retaining every update. | FedNCF local training and `src.federation.fedavg` |
| 8 | Compute standard sample-weighted FedAvg over all successful selected-party updates. A failed party fails the round; Phase 3 does not silently drop it. | `src.federation.fedavg` |
| 9 | Load the aggregated state into the global model. Optimizer state is neither aggregated nor retained. | `src.federation.hfl` |
| 10 | At the approved schedule, evaluate the global model with shared, fixed candidate sets. Validation and test roles remain distinct. | `src.evaluation.ranking` |
| 11 | Append round metrics and write final metrics/runtime/partition records; optionally save the final approved checkpoint. | `src.utils.results` and `src.federation.runner` |

The runner should be a readable loop over rounds. It should not dispatch by
registry or hide the flow behind callbacks. Phase 3 may explicitly reject any
`federation.type` other than `hfl` with `NotImplementedError`; this leaves
the future protocol boundary clear without creating unused VFL classes.

## 4. Proposed module boundaries

### Common infrastructure

- Existing `DatasetBundle` and `PartyData` remain the dataset and
  client-data representations. No new `Client` class is needed.
- `federation/hfl.py` owns deterministic party selection and the synchronous
  HFL round loop. It deals only in `party_id` and `PartyData`.
- `federation/fedavg.py` owns a small `ClientUpdate` dataclass and weighted
  state aggregation. It does not know about users, datasets, or FedNCF.
- `federation/runner.py` owns run-level sequencing, validation/evaluation
  calls, and result-writer calls. It contains no local optimization math.
- `evaluation/ranking.py` owns candidate construction, scoring orchestration,
  HR/NDCG definitions, and aggregation across targets.
- `utils/results.py` writes plain YAML/JSON/CSV/text artifacts.
- Existing `utils/config.py`, `utils/seed.py`, and `utils/logging.py` are
  extended or reused rather than duplicated.

### Baseline-specific code

`models/fedncf.py` owns only the FedNCF `torch.nn.Module`, FedNCF
labeled-example construction, and its local optimization function. Keeping
these together in one initial file avoids introducing a generic trainer
package for a single model. The local function consumes shared data contracts
and returns the common `ClientUpdate` value; it does not perform aggregation
or evaluation.

Later research methods may provide their own model and local-update code under
`src/methods/<method_name>/`. If a second method demonstrates that a local
training helper is truly shared, it can then be moved to common
infrastructure. No abstract model base class or method registry is justified
in Phase 3.

The only reusable behavior seam needed by the HFL loop is structural:

```python
client_update(global_state, bundle, party, round_index, config, seed, device)
    -> ClientUpdate
server_update(global_state, client_updates) -> state_dict
```

Phase 3 wires `train_local_fedncf` and `aggregate_fedavg` through direct
imports and ordinary function arguments. There is no runtime discovery,
registration, service container, or base-class hierarchy. A later method can
supply functions with these roles while retaining the same visible round loop;
methods with non-HFL execution are not forced through this seam.

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

This rule immediately supports grouped HFL parties containing several users.
It also avoids embedding an HFL-only identity assumption into the model or
evaluator. Future VFL/HyFL protocols will need different execution logic, but
they can continue to use explicit party ownership and bundle-wide entity IDs.

## 6. Model and local-training interface

### FedNCF model boundary

No recommendation base class is needed. The initial module should expose a
normal `torch.nn.Module` with the following behavior:

```python
logits = model(user_ids, item_ids)
```

- `user_ids`: one-dimensional `torch.long` tensor of processed user IDs;
- `item_ids`: one-dimensional `torch.long` tensor of the same shape;
- output: one unnormalized floating-point relevance logit per pair, shape
  `(batch_size,)`;
- larger logits rank ahead of smaller logits.

For a pointwise implicit-feedback baseline, the proposed local objective is
mean binary cross-entropy with logits over positive labels (`1`) and sampled
negative labels (`0`). A sigmoid is therefore not applied inside
`forward()`. This loss and the exact FedNCF architecture still require
researcher approval; the interface does not depend on whether the approved
network is GMF, MLP, or NeuMF.

The simplest baseline federation scope is the complete model `state_dict`:
all approved trainable embeddings and network parameters are global and
aggregated. There are no persistent party-local parameters. Whether user
embeddings should instead remain local is scientifically material and must be
approved before implementation.

### Local training behavior

For one selected party and round, local training receives:

- the current global `state_dict`;
- the unmodified `DatasetBundle`;
- one `PartyData` selected by `party_id`;
- explicit optimization settings;
- a deterministic seed derived from `(run_seed, round_index, party_id)`;
- the selected device.

It performs these explicit actions:

1. copy/load the current global state into one local FedNCF instance;
2. read only `party.train_interaction_ids`;
3. call the existing `sample_negative_items()` for those positive
   interaction IDs, with the configured ratio, strategy, replacement policy,
   and approved excluded splits;
4. expand each positive interaction into one positive
   `(user_id, item_id, 1)` and the returned
   `(user_id, negative_item_id, 0)` records;
5. deterministically shuffle those records for each local epoch, use all
   records (`drop_last=False`), and train with the configured batch size;
6. create a fresh optimizer each selected party and round; do not carry or
   federate optimizer state;
7. return a `ClientUpdate` containing `party_id`, a CPU copy of the
   complete model state, `num_examples`, and mean loss.

`num_examples` is the size of the constructed labeled local dataset, before
multiplication by local epochs. Empty local training data is an explicit error
in the Phase 3 HFL baseline.

Returning complete states is simpler and easier to validate than deltas for
the first synchronous FedAvg baseline. To bound memory with thousands of
`per_user` parties, the HFL loop sends each returned state into a streaming
FedAvg accumulator and releases the local model/update before training the
next party. This preserves full-state semantics without storing thousands of
states simultaneously.

Training negative sampling and evaluation candidate construction remain
different operations:

- local training calls the Phase 2B sampler and consumes only its negative
  item IDs;
- the evaluator independently calls the same sampler for held-out targets,
  then inserts the target positive and controls candidate ordering;
- neither caller changes the sampler's target exclusion, known-positive
  exclusion, sorted output, capacity checks, or replacement semantics.

Whether training negatives are fixed for the whole run or resampled per round
is not specified by the current protocol. A per-round derived seed is a
reasonable proposal, but requires researcher approval.

## 7. FedAvg and participation semantics

For selected client updates `i = 1..m`, complete floating state tensor
`theta_i`, and positive local weight `n_i`, standard weighted FedAvg is:

```text
theta_global_next = sum_i(n_i * theta_i) / sum_i(n_i)
```

The proposed weight is the number of labeled local training examples created
for that client (one positive plus the configured negative ratio per retained
training interaction), not the number of batches and not that count multiplied
by local epochs. With one common negative ratio this is proportional to the
number of positive training interactions. The exact weighting basis must be
approved because it changes the experimental algorithm if ratios ever differ.

Aggregation rules are exact and intentionally narrow:

- every returned state must have exactly the global state's keys, shapes, and
  dtypes;
- floating tensors are accumulated on CPU in a fixed ascending-`party_id`
  order, using a higher-precision accumulator for float16/float32 and casting
  once to the global dtype;
- non-floating tensors are not averaged; every client value must be identical
  to the starting global value, otherwise aggregation raises an error;
- optimizer state, local loss, and counters are not model parameters and are
  not aggregated;
- after aggregation the global model is replaced by the resulting state;
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
The publication values for rounds and participation remain researcher choices.

## 8. Evaluation design

Evaluation is centralized shared infrastructure and operates on the global
model in `eval()` mode under `torch.no_grad()`.

### Candidate construction

For each validation or test target interaction:

1. obtain its processed `user_id` and held-out positive `item_id` from the
   bundle;
2. call `sample_negative_items()` with a fixed evaluation seed, replacement
   disabled, and the approved default exclusions `train`, `valid`, and
   `test`;
3. combine the one target positive with exactly the configured number of
   negative item IDs;
4. score every `(user_id, candidate_item_id)` pair with the global model.

Validation and test candidate maps should be generated once per run and reused
across rounds so metric changes reflect model changes rather than newly drawn
candidates. Their configuration and checksum are recorded. The current
sampler already raises if the eligible pool is too small; the evaluator must
propagate that failure. The configured candidate set must contain at least
`max(k)` items.

The exact number of evaluation negatives is not defined in
`EXPERIMENT_PROTOCOL.md` and is therefore an approval gate, not a code
default.

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

The current protocol requires tie behavior to be defined but does not define
it. The proposed deterministic rule is descending score followed by ascending
processed `item_id`; this must be approved and then tested. Candidate input
order must not decide a tie.

Validation must use only validation target IDs and test must use only test
target IDs. A conservative proposal is fixed training rounds, validation for
monitoring at a configured interval, and one test evaluation of the final
global state. Best-validation checkpoint selection, early stopping, and test
evaluation every round must not be added without an approved protocol.

## 9. Configuration

Phase 3 should use one self-contained YAML file per experiment. `load_yaml()`
loads it and a small explicit validator rejects missing keys, unknown enum
values, invalid ranges, and unknown top-level sections. No scientific field is
silently defaulted. The runner writes the resolved mapping back to the result
directory before training.

A minimal shape is:

```yaml
experiment:
  name: ml1m_fedncf_fedavg

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
  local_epochs: 2

model:
  name: fedncf
  architecture: <researcher approval required>
  embedding_dim: <researcher approval required>
  hidden_dims: <required only if the approved architecture uses them>

optimization:
  optimizer: <researcher approval required>
  learning_rate: <researcher approval required>
  batch_size: <researcher approval required>
  weight_decay: <researcher approval required>
  train_negatives_per_positive: <researcher approval required>
  resample_negatives_each_round: <researcher approval required>

evaluation:
  k: [10]
  negatives_per_positive: <researcher approval required>
  excluded_splits: [train, valid, test]
  tie_breaker: <researcher approval required>
  validation_interval: <researcher approval required>
  checkpoint_selection: <researcher approval required>

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

The values `rounds: 20`, `client_fraction: 1.0`, and `local_epochs: 2`
are existing protocol examples/defaults, not newly approved publication
settings. The ML1M negative training ratio of 4 exists in the Phase 2B dataset
YAML but is not defined by the thesis-wide experiment protocol; it remains an
explicit scientific field here.

One base seed is sufficient in the user-facing configuration. Deterministic
sub-seeds should use a documented `numpy.random.SeedSequence` over integer
components such as purpose, round, and party; they must not use Python's
process-randomized `hash()`. The resolved run metadata records the
derived-seed scheme. `set_random_seed()` is still called once, with its
existing best-effort CUDA qualification.

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
  local-example count, weighted mean training loss, validation metrics when
  scheduled, elapsed seconds, and communication bytes;
- `runtime.json`: start/end times, duration, Python/PyTorch/CUDA versions,
  selected device/GPU, dataset/cache manifest checksum, git commit and dirty
  flag, and best-effort reproducibility qualification;
- `run.log`: the existing logger's human-readable progress;
- `checkpoint.pt`: final or approved selected global model state only, when
  enabled.

`partition.json` is the minimal run-level owner for the missing Phase 2B
partition metadata. It records partition name/type/mode, configured partition
seed, actual party count, complete `party_id -> user_ids` assignment,
per-party train/validation/test/user/item counts, base-cache identity, and a
checksum of the canonical assignment. This avoids duplicating global metadata
into every `PartyData` and does not alter the data contract or cache layout.

For communication accounting, record upload and download bytes separately
from actual transmitted state tensors, then record their sum. This prevents an
ambiguous `communication_mb` value. The researcher should approve whether
the headline metric includes both directions before publication.

Writes should use standard-library JSON/CSV and PyYAML, with temporary files
renamed into place for final summaries where practical. No database,
dashboard, or external tracking dependency is needed.

## 11. Proposed file tree

Only the following additions or focused extensions are planned for Phase 3:

```text
configs/
└── experiments/
    └── ml1m_fedncf_fedavg.yaml
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
├── test_phase3_integration.py
└── test_phase3_ml1m_smoke.py
```

File responsibilities:

- `configs/experiments/ml1m_fedncf_fedavg.yaml`: one auditable, complete
  Phase 3 run configuration; no config-composition engine is required.
- `scripts/run_experiment.py`: parse only `--config`, call the runner, and
  return a nonzero exit code on failure. It contains no training logic.
- `src/evaluation/ranking.py`: shared candidate construction, model scoring,
  HR@K, NDCG@K, and target-level aggregation.
- `src/federation/fedavg.py`: `ClientUpdate` and standard weighted
  full-state aggregation.
- `src/federation/hfl.py`: deterministic party selection and the synchronous
  HFL round loop.
- `src/federation/runner.py`: the explicit run-level sequence and only the
  wiring between data, HFL, model, evaluation, and results.
- `src/models/fedncf.py`: the approved FedNCF network plus its labeled-example
  construction and local training function.
- `src/utils/config.py`: retain `load_yaml`; add narrow Phase 3
  key/type/range validation rather than a new configuration framework.
- `src/utils/results.py`: create a non-overwriting run directory and write
  the plain result artifacts.
- unit/integration test files keep each boundary independently reviewable; the
  real-data smoke has an explicit opt-in gate.

Package `__init__.py` files may export the small public functions after their
modules exist. Empty VFL, HyFL, protocol, trainer, factory, or registry files
must not be added.

## 12. Testing plan

### Unit tests (CPU)

`test_ranking.py` should verify hand-calculated HR/NDCG values for positive
ranks inside/outside K, descending scores, the approved tie rule, candidate
order independence, all-known-positive exclusion, and target-level rather
than party-level averaging.

`test_fedavg.py` should verify unequal sample weighting, equal-weight
behavior, dtype/shape/key validation, fixed handling of identical non-floating
state, rejection of differing non-floating state, zero weights, and non-finite
values.

`test_hfl.py` should verify full participation, partial-participation count,
same-seed replay, round-dependent deterministic selections, no duplicates, and
use of `party_id` rather than `user_id` for selection.

`test_fedncf.py` should verify input/output shape, finite local loss, no
mutation of the global input state, training examples drawn only from the
party's train IDs, target-safe negative use, correct returned example count,
optimizer reset, and explicit use of interaction user IDs. A fixture should
deliberately make a party own multiple users or make its ID differ from its
user's ID.

Existing Phase 2B tests remain unchanged and passing.

### Synthetic integration test (CPU)

`test_phase3_integration.py` should construct a tiny valid bundle with enough
unseen items, build parties, run at least two rounds, aggregate updates, and
evaluate fixed validation/test candidate maps. Assertions should include:

- completion without filesystem or GPU dependencies;
- finite model state and metrics in `[0, 1]`;
- exactly conserved party-owned interaction IDs;
- no target positive among sampled negatives;
- identical selected parties, final state, metrics, and structured result
  content for two runs with the same seed;
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

The smoke verifies cache/load, partition, FedNCF forward/local update, FedAvg,
shared metrics, and artifact writing. CPU is allowed for portability; on the
target server the explicit real-run check should use one device such as
`cuda:0`. Publication-scale client participation or evaluation is not part
of this test.

## 13. Researcher decisions required

Implementation can proceed on the engineering structure below, but no
publication baseline should be claimed until the scientific choices are fixed
in configuration and protocol documentation.

### Engineering decisions with recommended defaults

| Decision | Recommendation | Reason |
|---|---|---|
| Client update form | Complete model state | Simplest auditable FedAvg boundary; stream states to control memory. |
| Optimizer lifecycle | Fresh optimizer for every selected party/round | Avoids hidden client state and optimizer aggregation. |
| Partial-selection rounding | `max(1, ceil(fraction * N))` | Makes any positive fraction participate and is explicit. |
| Selection/execution order | Sample without replacement, then sort party IDs | Reproducible aggregation order. |
| Device | CPU for tests; one explicit `cuda:N` for real runs | Matches scope without distributed complexity. |
| Configuration | One validated YAML; no merge/override system | Saved configuration matches executed configuration. |
| Partition metadata | `partition.json` in each run | Closes the audit trail without changing Phase 2B contracts. |
| Results | YAML/JSON/CSV/log plus optional checkpoint | Meets the existing protocol with no new dependency. |
| Error policy | Fail a synchronous round on any selected-client failure | Avoids silently changing participation or weighting. |

### Scientific/experimental decisions requiring approval

1. **Exact FedNCF definition:** GMF, MLP, or NeuMF; embedding tables,
   embedding dimension, hidden-layer sizes, activations, dropout, and
   initialization. The name `FedNCF` alone is insufficient.
2. **Local objective:** approve pointwise `BCEWithLogitsLoss` versus a
   pairwise alternative and any regularization. This design proposes pointwise
   BCE but does not treat it as approved.
3. **Federated parameter scope:** approve aggregating the full model including
   user and item embeddings versus keeping any user representation local. This
   materially changes the meaning of FedNCF/FedAvg.
4. **Training negatives:** approve the ratio (the existing dataset YAML says
   4, but the experiment protocol does not), replacement policy, exclusions,
   and fixed-versus-per-round resampling.
5. **Evaluation candidates:** set the exact negative count, replacement
   policy, exclusions, fixed candidate reuse, and deterministic tie rule. The
   current protocol specifies HR@10/NDCG@10 but not candidate count or ties.
6. **FedAvg weight:** approve constructed labeled-example count as `n_i`
   versus positive interaction count. They are proportional only while every
   client uses the same negative ratio.
7. **Validation usage:** decide evaluation interval and whether validation is
   monitoring-only or selects a checkpoint/hyperparameters.
8. **Model selection and early stopping:** approve fixed final-round state
   (the conservative proposal), best-validation state, or an early-stopping
   rule, including tie handling.
9. **Training schedule:** approve publication values for rounds, participation
   fraction, local epochs, optimizer, learning rate, batch size, and weight
   decay. The protocol's 20/1.0/2 values are defaults, not immutable choices.
10. **Metric aggregation and ranking:** approve target-level macro averaging,
    descending logits, and item-ID tie breaking as the frozen thesis protocol.
11. **Efficiency headline:** decide whether reported communication includes
    both model download and update upload; the implementation should record
    both separately regardless.

The recommended initial validation policy, pending approval, is fixed rounds,
validation for monitoring only, no early stopping, and one final test
evaluation. It avoids using test results for model selection.

## 14. Implementation sequence

Each task should be one small reviewable change with focused tests, followed by
the full suite and `git diff --check`.

1. **Phase 3A — freeze experimental semantics and configuration.** Resolve the
   researcher decisions above, add the single experiment YAML, and add
   explicit validation to `utils/config.py`. No training yet.
2. **Phase 3B — shared ranking evaluation.** Implement approved candidate
   construction and HR/NDCG in `evaluation/ranking.py`, with hand-calculated
   CPU tests. Reuse the Phase 2B sampler unchanged.
3. **Phase 3C — FedAvg and party selection.** Implement `ClientUpdate`,
   weighted aggregation, and deterministic HFL selection with unit tests,
   including non-floating state and party/user separation.
4. **Phase 3D — FedNCF and local training.** After architecture/objective
   approval, implement the model and one explicit local-update function with
   CPU tests for sample ownership, negative use, and state return.
5. **Phase 3E — orchestration and results.** Implement the explicit HFL round
   loop, high-level runner, result writer, and narrow CLI. Add partition-level
   run metadata and deterministic artifact tests.
6. **Phase 3F — end-to-end validation.** Add the fast synthetic CPU integration
   test, then the opt-in small real-ML1M smoke on one selected CUDA device.

Only after Phase 3F passes should a separate, researcher-approved task define
and run publication-scale ML1M experiments. Phase 4 and thesis method
implementation remain outside this design.
