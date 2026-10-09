# Phase 2B MovieLens-1M Engineering Audit

Audit date: 2026-10-09

Baseline:

- branch: `codex/phase2b-audit`;
- commit: `6b0a994`;
- tag: `v0.2.1-ml1m-data`;
- original test result: `19 passed in 1.32s`.

## Scope

This audit covers the completed Phase 2B MovieLens-1M data foundation:

- raw `ratings.dat` parsing and implicit-feedback conversion;
- filtering, duplicate handling, and deterministic user/item remapping;
- temporal leave-one-out train/validation/test splitting;
- HFL `per_user` and `grouped_round_robin` partitioning;
- shared negative-item sampling;
- base cache serialization, manifest validation, and reconstruction;
- interaction-, target-item-, and partition-level leakage checks;
- deterministic behavior and obvious ML1M-scale engineering risks.

The audit does not cover Amazon data, recommendation models, evaluation
metrics, federation training, or any Phase 3 functionality.

The real dataset was read without modification from
`/data_nfs/yuanhaochen/data/ml-1m/ml-1m/ratings.dat`. Temporary cache
artifacts were written only under `/tmp`.

## Verified invariants

### Raw parsing and preprocessing

- `load_ml1m_ratings()` requires exactly four `::`-separated numeric
  fields and preserves source order, raw ratings, and integer timestamps.
- Preprocessing rejects non-positive raw user/item IDs, negative timestamps,
  and non-finite ratings before final ID assignment.
- The approved `all_observed` implicit-feedback conversion maps retained
  interactions to `float32` value `1.0`; it does not use rating magnitude
  as a preference threshold.
- The default `keep_latest` duplicate policy selects the greatest
  `(timestamp, source_index)` for each raw user-item pair. The minimum-user
  filter is rerun after deduplication.
- Canonical order is restored to source order before user, item, and
  interaction IDs are assigned.
- User and item IDs are assigned independently by first appearance and are
  contiguous from zero. `DatasetBundle` validates that interaction IDs
  cover the complete mapping namespaces.
- Two complete preprocessing runs over the real file produced identical
  mappings, interaction arrays, and split arrays.

Evidence:

- `tests/test_data_preprocess.py` covers raw parsing, filtering,
  deduplication, tie handling, contiguous mapping, and repeated-run equality.
- The real-data audit retained all `1,000,209` records and produced
  `6,040` users and `3,706` items.
- The raw file SHA-256 recorded by preprocessing was
  `506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f`.

### Temporal leave-one-out and leakage

- `leave_one_out_split()` orders each user's interactions by
  `(timestamp, interaction_id)`.
- The latest interaction is test, the second latest is validation, and all
  earlier interactions are training.
- Users with fewer than three retained interactions are removed by
  preprocessing; the splitter also rejects such input directly.
- Equal timestamps are resolved deterministically by interaction ID.
- `InteractionSplits` rejects duplicate IDs within a split and overlap
  between splits. `DatasetBundle` additionally requires exact coverage of
  all interaction IDs.
- A vectorized whole-ML1M check confirmed every training interaction precedes
  its user's validation interaction and every validation interaction precedes
  its test interaction under the approved ordering.
- The real split counts were:
  - train: `988,129`;
  - validation: `6,040`;
  - test: `6,040`.

Evidence:

- `test_leave_one_out_uses_interaction_id_to_break_timestamp_ties`.
- `test_leave_one_out_matches_temporal_order_for_every_user`, added by this
  audit.
- The real-data disjointness and full-coverage checks all passed.

### HFL partitioning

- `per_user` creates one explicit `PartyData` per retained user and is
  independent of the experiment seed.
- `grouped_round_robin` applies one seeded user permutation and assigns
  complete users round-robin to the requested number of parties.
- Both modes derive party interactions from the global split arrays, keep all
  of a user's interactions in one party, and derive item coverage from those
  interactions.
- Internal validation and tests require exact per-split interaction
  conservation and exact user conservation. Any loss, duplication, or
  cross-party user assignment fails validation.
- Party identity and user identity remain separate fields. Numeric IDs happen
  to coincide in the current `per_user` construction, which the data
  contract permits but does not define as an identity equivalence.
- On real ML1M data, two `per_user` builds with different seeds were
  identical, and two 100-party grouped builds with seed `2026` were
  identical.

Evidence:

- `tests/test_data_partition.py` verifies user/item coverage, split
  conservation, per-user seed independence, and grouped determinism.
- The real-data audit produced `6,040` per-user parties and 100 grouped
  parties with all checks passing.

### Shared negative sampling

- The sampler constructs known-positive exclusions only from the configured
  split names and always applies an additional target-specific exclusion.
- Commit `6b0a994` fixed the earlier boundary case in which a target item
  could remain eligible when its split was omitted from
  `excluded_splits`.
- The target-specific filter creates a new NumPy array and does not mutate the
  cached per-user eligible pool.
- Without replacement, insufficient candidates raise `ValueError` and
  returned IDs are unique. With replacement, repetition is explicitly
  permitted, but an empty true-negative pool still raises.
- Positive interaction IDs are sorted before seeded sampling, so the result is
  independent of caller input order under a fixed seed.
- All returned values are bundle-wide item IDs.
- Sampling remains shared and caller-configured; it does not embed a
  training-only or evaluation-only candidate protocol.

Evidence:

- `tests/test_negative_sampling.py` covers full known-positive exclusion,
  target exclusion with an omitted target split, empty pools under both
  replacement modes, genuine-negative selection, cached-pool integrity, and
  determinism.
- Whole-test-set real-data sampling returned ten distinct valid negatives for
  each of `6,040` targets, excluded every target, and was identical for
  reversed input order with seed `2026`.

### Cache, manifest, and portability

- Base cache round trips preserve dataset name, schema, mappings, all five
  interaction arrays, all three split arrays, and dataset metadata.
- Artifact SHA-256 checks reject modified cache files.
- This audit added explicit rejection of unsupported cache schema versions.
- Manifest counts are checked against the reconstructed bundle.
- NumPy files are loaded with `allow_pickle=False`; mapping artifacts are
  UTF-8 JSONL.
- The manifest records the raw-file SHA-256, canonical preprocessing
  configuration, seed, mapping/minimum/duplicate/implicit/split policies,
  counts, and artifact checksums.
- Dataset resolution accepts an explicit root, then
  `THESISFEDREC_DATA_ROOT`, then repository-local `data/raw`. Dataset
  configuration and stored source references use relative paths, and root
  traversal is rejected.
- A real ML1M save/load under `/tmp` preserved semantic equality exactly.

Evidence:

- `tests/test_data_cache.py` covers semantic round trip, artifact corruption,
  manifest equality, and unsupported schema rejection.
- `test_data_root_prefers_environment_and_paths_remain_relative` covers
  portable path resolution and traversal rejection.

### Reproducibility summary

The following operations were verified deterministic under the approved
configuration:

- raw-order-based ID mapping;
- temporal split and timestamp tie handling;
- `per_user` HFL assignment independent of seed;
- seeded `grouped_round_robin` assignment;
- seeded negative sampling independent of positive-ID input order;
- semantic cache reconstruction.

The preprocessing seed is recorded even though the current ML1M preprocessing
steps contain no stochastic operation.

## Defects found

### Unsupported cache schema loaded silently

- Severity: medium.
- Affected function: `src/data/cache.py::load_base_cache`.
- Defect: the loader copied any manifest `schema_version` into the returned
  bundle without checking whether the running code supported it. A temporary
  reproduction changed the manifest version to `999.0`; the pre-audit
  loader returned a bundle reporting schema `999.0`.
- Risk: a stale or future cache could be interpreted under incompatible
  assumptions without an explicit failure.
- Fix: the shared data-contract schema constant now lives in
  `src/data/contracts.py`; preprocessing and cache loading use that same
  constant, and loading rejects a missing or unsupported version before
  reading artifacts.
- Regression:
  `test_base_cache_rejects_unsupported_schema_version`.

No other Phase 2B data-correctness defect was found.

## Test coverage

- Original result: `19 passed in 1.32s`.
- Focused tests added:
  - exact temporal train/validation/test assignment for every fixture user;
  - unsupported cache schema rejection.
- Final result: `21 passed`.
- No GPU or training job was used.

## Remaining limitations

These limitations do not change the verified interaction data, mappings,
splits, party assignments, or sampled negatives.

1. The base manifest contains the inputs needed to identify the current ML1M
   cache, but it does not yet populate every target audit field listed in the
   full data contract, including raw byte size, parser name, explicit
   timestamp unit/timezone policy, code revision/dirty flag, and runtime
   library versions.
2. `load_base_cache()` validates internal integrity and schema compatibility,
   but its current interface does not accept an expected raw checksum or
   preprocessing configuration. Selecting the cache matching a requested
   experiment remains the caller's responsibility.
3. HFL assignment and counts are explicit and derivable from `PartyData`,
   but there is no partition-level metadata owner. Individual party metadata
   currently records only `hfl_mode`, while the target contract also calls
   for the seed, number of parties, complete assignment map, and per-party
   counts. Deciding whether those global fields belong in a future partition
   manifest/container or elsewhere requires an architectural choice; this
   audit did not duplicate them into every party.
4. The negative sampler caches a full eligible-item array for each requested
   user. This is acceptable for ML1M, but scales as requested users times item
   count. The deliberately heavy real-data audit (two processed bundles,
   multiple partitions, whole-test-set sampling, and a cache round trip in one
   process) peaked at approximately `478,640 KiB` RSS.
5. The non-default `keep_all` duplicate policy may place repeated user-item
   observations in different temporal splits. The approved ML1M default is
   `keep_latest`; the real file contained no duplicate user-item pair.

The partition-metadata ownership question should be resolved before cached or
published partition artifacts are treated as fully compliant with the target
reproducibility metadata contract. It does not alter current in-memory HFL
partition correctness.

## Phase 2B conclusion

**PASS WITH MINOR ISSUES**

The MovieLens-1M data foundation is ready to support Phase 3 baseline
development: preprocessing, mapping, temporal splitting, HFL partitioning,
negative sampling, and base cache reconstruction are correct and reproducible
for the approved protocol. Phase 3 was not started by this audit.

Before freezing publication-grade experiment artifacts, resolve the documented
partition-metadata ownership question and complete the remaining manifest audit
fields.
