# FL-BASE-SYNTHETIC qualification

## Scope and baseline

PARALLEL-W2 A qualifies the existing FedGMF/FedAvg/FIT-1 path at commit
`afd96ba8e22271267936c344495f9c52a0eaea06`, tree
`432fcca069b1681b7d2912ac94d849897670901e`. Only this document and
`tests/test_federation_foundation.py` are new. Production code, existing tests,
fixtures, configuration files, dependencies and experimental semantics do not
change. This is synthetic correctness evidence, not a qualified experimental
profile or evidence of recommendation benefit.

The coverage audit uses the implemented APIs, not historical design sketches:
`ClientUpdate` contains party ID, shared state and positive-interaction count;
it contains no loss/optimizer/local-user state. HFL currently retains uploads
until aggregation; no streaming/scalability claim follows. FedGMF constructs
Adam inside each local update. It persists local user embeddings, **not Adam
moments**. FedAvg normalizes positive counts over participating updates only.
Topic 1.1 joint-step weighting and persistent Adam are not baseline policies.

## Coverage matrix established before new tests ran

Test names below are exact; paths are relative to this repository. Existing
coverage is reused and selected for a fresh regression run, not rewritten just
to increase test counts.

| Requirement | Existing coverage retained | Remaining gap / new coverage |
|---|---|---|
| Per-user and grouped ownership, split conservation | `tests/test_data_partition.py::test_per_user_hfl_mapping_is_stable_and_seed_independent`, `test_grouped_round_robin_is_deterministic_and_keeps_users_intact` | No new partition tests needed. |
| Full/partial participation, sorted deterministic arbitrary party IDs | `tests/test_hfl.py::test_full_participation_returns_every_sorted_party`, `test_partial_participation_uses_ceil_without_replacement_and_sorts`, `test_same_seed_round_and_available_ids_reproduce_selection`, `test_non_contiguous_party_ids_are_not_treated_as_user_ids` | Actual unequal-workload rounds now exercise both full and partial participation with IDs 7, 41, 103. |
| Independent aggregation oracle | `tests/test_fedavg.py::test_fedavg_uses_only_positive_interaction_weight` pins scalar 7.5 from weights 1/3; `test_equal_weight_fedavg` covers arrays | The accepted real-round test computes its expectation using production `aggregate_fedavg` again. New `test_actual_round_weights_positive_counts_not_optimizer_work[full/partial]` uses independent NumPy arithmetic on real uploads, unequal positive counts, and rejects plausible wrong normalizations. |
| Round-start shared isolation and nonparticipant local state | `tests/test_hfl_round.py::test_client_shared_state_mutation_is_isolated`, `test_full_round_is_synchronous_ordered_weighted_and_immutable`, `test_partial_round_replaces_only_selected_states_and_reproduces`; `tests/test_runner.py::test_partial_participation_preserves_unselected_local_state` | No duplicate malicious-callback test. New real-update oracle also checks selected/nonselected local states and original shared inputs. Opaque callback-owned local objects are not generically deep-copied by HFL. |
| Actual local parameter updates, input nonmutation and repeatability | `tests/test_fedgmf.py::test_grouped_party_training_updates_only_returned_states`, `test_local_training_is_exactly_reproducible`, `test_updated_local_state_can_be_reused_next_round`; `tests/test_runner.py::test_same_seed_reproduces_final_states_and_metrics` | New `test_fixed_learnable_fit_with_finite_gradients_and_fresh_adam` observes actual loss/gradient finiteness, fresh optimizer state per update, and all intended shared/local changes. |
| Learnability on a deliberately learnable bounded fixture | Existing tests establish state change and lifecycle, not a fixed-example loss/separation improvement | The new fixed-fit case compares initial/final diagnostic loss and mean positive-minus-negative logit separation against an unchanged no-update reference. No per-step monotonicity assertion. |
| FIT-1 test-evaluation boundary with validation monitoring | `tests/test_fedgmf_fit.py::test_fit_only_builds_and_scores_validation_candidates` guards candidate construction, scoring and evaluation; `test_fit_retains_structural_ownership_checks`; `test_fit_and_legacy_runner_match_accepted_baseline` | Already covered directly; rerun instead of duplicating. Fit still reads held-out IDs for known-positive exclusion and ownership checks. |
| Sampling, candidate construction, seeds and final-state evaluation | `tests/test_fedgmf.py::test_training_uses_shared_negative_sampler_with_all_known_exclusions`, `test_negative_seed_changes_by_round_and_local_epoch`; `tests/test_ranking.py::test_candidate_construction_includes_positive_once_and_excludes_known`, `test_validation_and_test_use_distinct_seed_streams`; `tests/test_runner.py::test_two_round_grouped_experiment_wires_complete_lifecycle` | Retain these regressions and the complete existing negative-sampling module. Validation remains monitoring only; final test evaluation remains full-run-wrapper-only. |

## New checks and frozen controls

### Independent real-update aggregation (two cases)

Three disjoint user histories have lengths 3, 5 and 7, yielding exactly 1, 3
and 5 training positives after the production temporal leave-one-out splitter.
All 15 catalogue items are observed; there are no fabricated metadata-only
items. The public per-user partition builder supplies ownership, then party
IDs are explicitly changed to 41, 7 and 103 without changing user IDs.

| Party | Positives | Local epochs | Negatives/positive | Batch size | Actual Adam steps | Labeled examples across epochs |
|---|---:|---:|---:|---:|---:|---:|
| 41 | 1 | 1 | 1 | 2 | 1 | 2 |
| 7 | 3 | 2 | 2 | 4 | 6 | 18 |
| 103 | 5 | 3 | 4 | 4 | 21 | 75 |

These unequal per-party workloads are **test-only stimuli** to the HFL callback
and public local trainer, not a new runner configuration policy. Both fractions
1.0 and 0.5 use run seed 2026, round 0, embedding dimension 4, learning rate
0.03, weight decay 0 and CPU. The default uniform sampler, no replacement,
all-split exclusion and seed derivation are unchanged.

Observer wrappers count real optimizer steps and labeled examples, then invoke
the original implementations. Expectations for every shared tensor use NumPy
float64 weighted sums of returned values, divided by participating **positive**
counts. The prediction bias additionally must distinguish this answer from
batch-count, generated-negative-count, positive-count-times-epoch, and total
labeled-example weights. Partial participation also distinguishes normalization
by all parties' counts. Existing hand-calculated scalar FedAvg tests provide a
separate small known-value oracle.

### Fixed learnable FIT-1 case (one case)

Controls fixed in source before the first run:

- User 0 observes items 0/1/2/3; user 1 observes 4/5/6/7. Time increases in
  source order. Production splitting holds out the last two items per user,
  leaving training positives (0,0), (0,1), (1,4), (1,5).
- One grouped party, constructed with partition seed 11, has explicit ID 73
  and owns users 0 and 1. Eight catalogue items suffice: each user has four
  eligible negatives, exactly the requested count. Every epoch therefore
  samples the full eligible set, in production-seeded order. No sampling stub,
  replacement, exclusion change or candidate-rule change is used.
- Production `fit_fedgmf_hfl`: seed 2026, canonical `ncf_gmf` initialization,
  dimension 4, full participation, 12 rounds, 5 local epochs, learning rate
  0.05, weight decay 0, batch size 20, four negatives per positive. Exactly
  60 full-batch Adam steps, with a fresh Adam for each of the 12 local updates.
- A test-local copy of the reference configuration is validated. Validation
  uses four negatives and K=[1,3] because this fixture is small. No YAML or
  production default changes; no raw-data/artifact execution entry point.
- The fixed 20 diagnostic examples comprise each training positive followed
  by its four eligible negatives. The same labels/order are scored initially
  and finally. Loss uses independent stable NumPy logistic-loss arithmetic;
  separation is mean positive logit minus mean negative logit. These are
  training diagnostics, not held-out recommendation metrics.
- Predeclared assertions: final diagnostic loss < initial loss - 0.15;
  final separation > initial separation + 0.5. A separately copied initial
  state receives **no updates** and must retain exactly its initial logits,
  loss and separation. No adaptive stopping, stepwise-decrease requirement,
  seed search, threshold tuning or best-round selection.
- Pass-through observers check all 60 production BCE losses and all four
  parameter gradients before each Adam step, and verify the optimizer starts
  with empty state on each local update. They retain only 12 tiny optimizer
  objects for identity checks. Pytest monkeypatch restores all patched methods;
  no model, optimizer, objective, sampler or gradient is substituted.

## Execution evidence

Both commands below were executed on CPU on 2026-10-19 in A's isolated
worktree, with exit 0; these are newly executed results, not inherited PW1
review counts. Existing test implementations were reused unchanged.

| Run | Actual pytest result | Exit |
|---|---|---:|
| New module (two aggregation cases plus fixed learnability case) | **3 passed in 2.59s** | 0 |
| Nine selected existing regression modules | **265 passed in 31.13s** | 0 |

Environment: Conda `thesis-fedrec`; Python 3.11.16; NumPy 1.26.4;
PyTorch 2.7.1+cu126; PyYAML 6.0.3; pytest 8.4.2. `CUDA_VISIBLE_DEVICES=""`;
OMP/MKL/OPENBLAS threads each 1; actual PyTorch threads 1. All three
real-data/GPU opt-ins were unset. `PYTHONDONTWRITEBYTECODE=1`, explicit
worktree `PYTHONPATH`, and `-p no:cacheprovider` kept generated payloads out
of the checkout. The project root was set per process to
`/data_nfs/yuanhaochen/ThesisFedRec`.

External evidence directory:
`/data_nfs/yuanhaochen/ThesisFedRec/tmp/parallel_wave2.pypknz/fl_base_evidence.8MnPHU/`.
`pre-execution.log` records environment, imports resolving to A, and test/document
hashes before the first run. The test module stayed byte-identical thereafter:
SHA-256 `3eb5df3ae45bb1f2bd0b5958367ade00a8d3536d0641f7b1426814acf1792cf1`.
The coverage matrix, fixture, seeds, budgets and assertions preceded execution;
only this evidence section was completed afterward.

Exact test commands (working directory: A's `worktrees/fl-base-synthetic`):

```bash
python -m pytest -q -s tests/test_federation_foundation.py --basetemp=/data_nfs/yuanhaochen/ThesisFedRec/tmp/parallel_wave2.pypknz/fl_base_evidence.8MnPHU/pytest-new.xZDjZF --junitxml=/data_nfs/yuanhaochen/ThesisFedRec/tmp/parallel_wave2.pypknz/fl_base_evidence.8MnPHU/pytest-new.xml -p no:cacheprovider
python -m pytest -q tests/test_data_partition.py tests/test_hfl.py tests/test_hfl_round.py tests/test_fedavg.py tests/test_fedgmf.py tests/test_fedgmf_fit.py tests/test_runner.py tests/test_ranking.py tests/test_negative_sampling.py --basetemp=/data_nfs/yuanhaochen/ThesisFedRec/tmp/parallel_wave2.pypknz/fl_base_evidence.8MnPHU/pytest-regression.5lWFt9 --junitxml=/data_nfs/yuanhaochen/ThesisFedRec/tmp/parallel_wave2.pypknz/fl_base_evidence.8MnPHU/pytest-regression.xml -p no:cacheprovider
```

Within the evidence directory, `os-tmp-new.6ZndEL` and
`os-tmp-regression.A2UDnP` were the respective TMPDIR values. Each was a
fresh sibling of its pytest basetemp, not the same directory or an ancestor.
`pytest-new.log`/`.xml` and `pytest-regression.log`/`.xml` retain output and
JUnit cases; corresponding `.exit` files record exits. No separate full suite
or collection was run; the default full suite awaits the combined-candidate gate.

Observed fixed-example diagnostics:

| Quantity | Initial / no-update reference | Final |
|---|---:|---:|
| Mean logistic loss | 0.6931401310077763 | 2.597165987247666e-10 |
| Mean positive-minus-negative logit | 1.4862802572679357e-05 | 43.96778726577759 |

All 60 observed BCE losses and 240 parameter-gradient tensors were finite;
12 distinct fresh Adam instances each took five steps. Both predeclared
improvement assertions passed. Near-zero loss on these deliberately separable
training examples is not generalization evidence. Full-round selected parties
were `(7, 41, 103)`, positive counts `(3, 1, 5)`; partial-round selection was
`(7, 41)`, counts `(3, 1)`. Observed steps/examples matched the table above.

Failures/resolution: neither test command failed; no assertion, oracle, seed,
budget or fixture was revised after observing results. No production defect
was found or fixed. Setup did encounter Git's known NFS dubious-ownership
rejection (exit 128 for each new checkout); only the authorized exact-path,
command-scoped exceptions were used and recorded in `setup.log` and
`setup_mapping.json`. No persistent Git, ownership or permission change was
made, and the underlying UID-mapping cause remains undetermined.

## Boundaries and deferred work

No production fix is authorized here. A genuine new production failure must
be independently reproduced and reported as incomplete without weakening the
test, adding xfail/skip, changing the oracle or patching production.

Still required under separate authorization: real-ML1M learning diagnostics,
resource/memory/runtime measurement, bounded pilot calibration and scientifically
approved multi-seed experimental profiles. This work does not qualify real-data
convergence, recommendation benefit, GPU behavior, privacy, full Books scaling,
or Topic 1.1/1.2 algorithms. No real-data/GPU opt-in, training experiment, tuning,
graph/method training, default full-suite run or later-unit work is performed.
