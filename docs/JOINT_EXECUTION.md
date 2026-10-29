# Unit2: bounded joint pri-none execution

This unit connects the accepted DOI4 numerical kernels to the public plan,
delivery, optimizer-owner and ranking components. It is synchronous in-process
CPU execution with `pri=none`, `warmup=none`, and persistent dense Adam. It is
not a complete research method, distributed engine, research profile, or
real-data qualification. No initializer, RNG policy, source selection, private
branch, checkpoint, transport or configuration default is introduced.

## Direct public interfaces

Import from `src.federation.joint_execution` and `src.evaluation.terminal`;
package exports and the existing FedGMF/HFL runner are unchanged.

```python
fit_doi4_pri_none(
    *, rounds: tuple[RoundDrainPlan, ...],
    initial_values: dict[str, dict[str, torch.Tensor]],
    user_ids_by_party: dict[int, tuple[int, ...]],
    adapters: dict[int, BinaryInputAdapter],
    feature_runtimes: dict[int, FeatureRuntime],
    input_bindings: dict[int, tuple[str, str, int, str]],
    allowed_sources_by_receiver: dict[int, tuple[int, ...]],
    step_source_pools: dict[tuple[int, int], dict[int, tuple[int, ...]]],
    inference_source_pools: dict[int, tuple[int, ...]],
    inference_online_party_ids: tuple[int, ...],
    context_id: str, initialization_id: str, profile_id: str,
    learning_rate: float, betas: tuple[float, float], epsilon: float,
    validation_candidates: EvaluationCandidateSet,
    k_values: tuple[int, ...],
) -> Doi4PriNoneFit

score_doi4_pri_none(
    fit: Doi4PriNoneFit, user_ids: np.ndarray, item_ids: np.ndarray, *,
    expected_context_id: str,
) -> np.ndarray

evaluate_terminal_once(
    handle: TerminalEvaluationHandle, candidate_set: EvaluationCandidateSet, *,
    expected_terminal_id: str,
) -> dict[str, float | int]

observe_terminal_evaluation(handle: TerminalEvaluationHandle) -> dict[str, object]
```

Every fit argument is explicit. `Doi4PriNoneFit` and
`TerminalEvaluationHandle` reject public construction. A fit exposes
`terminal_id`, `scheduled_end_id`, `context_id`, `initialization_id`,
`profile_id`, `terminal_handle`, `round_receipts` and `step_receipts`.
Repeated `terminal_handle` reads return the same handle, not a reset opportunity.
`observe_parameters()` returns independent detached tensor copies;
`observe_bindings()` returns an independent nested content record, including
initial/final parameter digests, ordered plan/trace IDs, user routing, input
bindings, inference pools/online roster, VALID digest/k and terminal binding.
Neither observation exposes an optimizer, authoritative Parameter or live graph.

## Preflight and ownership

Registered parties are arbitrary nonnegative int64 IDs, not user IDs. Their
sorted unique user tuples are disjoint; an owner may have several users or none.
Plans must be prepared, have increasing distinct round indices, and contain
only registered parties and owned users. Exactly one explicit pool mapping is
required per nonterminal `(round_index, step_index)`, covering precisely that
step's loss receivers. Sorted unique source pools may be empty, but must fit
both the explicit permission ceiling and the plan's selected online roster.
Inference pools are independent and cover all registered receivers; their fixed
online roster is exactly the registered party roster.

The parameter mapping has `center` and exactly `party:<id>` owners. Shapes and
the nine center/five party names come from `doi4_parameter_specs`. All values
are finite dense strided CPU float32 or float64 with one common dtype. Valid
noncontiguous/nonleaf values are copied, without casts, into distinct owned leaf
Parameters. There is no guessed dimension, initialization or random draw.

Each binding is `(runtime_binding_id, map_id, schema_epoch, adapter_binding_id)`.
The fit retains the exact borrowed runtime and adapter handles in private copied
maps. Caller mutation of those maps cannot replace handles. Liveness and binding
are checked before work, each step, each monitor and each score call; closure is
an error. Fit never closes borrowed inputs. The adapter's interface does not
expose a map/runtime backpointer: outer preparation is responsible for authentic
map-row/table associations and permission claims. Digest-shaped strings alone
are not permission or source-provenance proof.

Before optimizer creation, the implementation checks every planned training
request and fixed VALID request using actual delivery and tensorization, plus
empty delivery to verify each field width. It discards preflight payloads and
does not encode them. This adds a traversal/delivery of consumed rows. It does
not hash the entire source content on every batch. Static future TEST grants
cannot be checked during fit because TEST candidates are not accepted there.

The outer coordinator constructs/checks each `TrainingTrace` against its bundle
and party roster before plan preparation. The fit receives plans, not traces or
a bundle. It binds ordered plan/trace identities and the caller's overall
context; it does not equate trace contexts for different selected rosters or
claim to independently authenticate held-out membership. VALID DTO contents are
copied and revalidated. K values are unique positive integers, sorted once;
the existing evaluator's behavior for k larger than a candidate set is retained.

Inputs must remain stable during validation and ownership copying. These are
synchronous copy boundaries, not atomic snapshots under concurrent writers.

## Joint step and update barrier

One persistent `OptimizerOwners` instance retains the same Parameters and Adam
states across every supplied step and round, with weight decay fixed to zero.
For each step:

1. Peek without advancing. Pack positive-first groups and freeze request
   geometry, version dependencies, expected contributions and a step identity.
2. For every requested source, perform real delivery, tensorization and encoding
   once. J=0, all-false masks and zero H remain executed Tensor paths. No source
   may return `None`; structural absence means omission from a declared pool.
3. Preserve each original source H graph and create an owning detached center
   H leaf. Center fusion z is detached into an independent local leaf. Direct
   score item rows and center-query item rows are separate owning leaves, as
   are gathered user rows.
4. Apply the supplied `m/N` once to each local scalar group loss. Compute local
   derivatives, then the center VJP. Scatter-add repeated direct and query item
   rows into one table gradient and user rows into their owner's table gradient.
5. Require every incident receiver-to-source derivative, including zeros, then
   sum in sorted receiver order and execute one VJP per original source H.
   Accumulate local encoder gradients and late shared-query gradients.
6. Validate the complete contribution barrier and all final gradient shapes,
   dtypes and finite values before one optimizer apply. Check its flat receipt
   against submitted Tensor/None activity, then and only then advance the cursor.

Contribution keys bind `(step_id, phase, sender, receiver_or_source,
payload_kind, parameter_version)`; geometry carries processed row IDs and tensor
shapes/dtypes. This small implementation conservatively binds the ordered
version/clock record for **all registered parameters** in every contribution,
which includes each actual dependency. It does not use Tensor `_version` as an
Adam clock. Missing, duplicate, stale, unexpected or wrong-geometry records fail
before apply. The expected set is frozen before forward computation; source
VJPs are also gated by the completed receiver set.

Accumulators start as `None`; an actual zero Tensor remains a Tensor. Source-only
user tables remain structurally absent. Shared-query gradients are `None` only
when no source executes. Owner versions and parameter clocks advance exactly
where Tensor gradients are submitted, including zeros. Loss weights are not
interpreted as Adam displacement ratios.

Pre-apply failure leaves that step unapplied, without rolling back earlier steps.
An exception after physical Adam begins may leave genuine partial updates and
is fail-stop: no retry, rollback, cursor advance or terminal result. Cleanup
closes only owned optimizer resources and preserves the primary exception.

## Monitoring and terminal state

Each completed round, including an empty round, evaluates the same fixed VALID
targets and k exactly once. Empty rounds perform no apply and never divide by
zero. Zero supplied rounds are rejected. No best-state choice, early stopping,
pre-round monitor or TEST candidate construction occurs in fit. A monitor
failure makes fit fail even if updates already occurred.

`StepReceipt` contains `step_id`, `round_index`, `step_index`, `plan_id`,
`trace_id`, `request_id`, `expected_contribution_digest`,
`completed_contribution_digest`, `receiver_ids`, `source_ids`,
`positive_group_counts`, `round_positive_count`, the existing flat
`optimizer_receipt` and immutable `cursor_after`.
`RoundReceipt` contains `round_index`, `plan_id`, `trace_id`,
`expected_step_count`, `completed_step_count`, `round_positive_count`, `skipped`,
`generation_before`, `generation_after`, immutable `validation_metrics`
key/value tuples and `validation_candidate_digest`.
These are trusted-local records, not automatically safe cross-party telemetry.

The scheduled-end ID binds the initial numeric contents, explicit labels,
ordered plans/traces, bindings/routing/pools, optimizer settings, VALID contents
and the last-supplied-round rule. The terminal ID additionally binds final
values and completed receipts. Numeric digests use shape, dtype and logical
contiguous little-endian bytes; scalar records use versioned compact sorted-key
UTF-8 JSON with `allow_nan=False`. No repr, object address or Python hash is an
identity. These hashes provide local consistency, not cryptographic provenance.

Finalization copies authoritative numeric values once into private scoring
storage and closes owners. Scores gather requested rows from that same storage,
not a full item-table clone per target. Plain native metadata-free rank-one
NumPy int64 user/item inputs must align and remain in the owned processed
domain. Scoring restores original order/repeats and returns an owning finite
float64 vector; empty input still checks bindings. The score path reuses actual
delivery/encoding/packing/fusion/local scoring under no-grad.

The method-neutral terminal binder is private and binds this actual scorer,
method-specific non-scoring validator, fixed k and one receipt. It does not
hard-code a feature/graph runtime, so a future purely numeric terminal can use
its own validator. Public evaluation accepts neither replacement scorer nor k.
The ready-state reentry guard is acquired before validation. Correctable
pre-start errors leave ready/attempt 0 without scoring; successful validation
sets started/attempt 1 immediately before the sole shared-evaluator call.
Success records complete; scoring/evaluator exceptions or interrupts record
failed where possible and propagate. Complete, failed or surviving started
states cannot be reused. Observations and returned metrics are independent
copies, and handle copying is rejected. This is cooperative single-process
bookkeeping, not hostile-Python protection or durable exactly-once semantics.

## Bounded acceptance and limits

The frozen synthetic recipe `UNIT2-JOINT-2x5-v1` uses 12 items, three users,
parties `(7,23,41)`, grouped two-user party 7, source-only party 41 with actual
empty-field delivery, and rounds `(0,2,5)` where round 2 is empty. The explicit
value formula, trace seed, dimensions and Adam settings are test inputs only,
not new research defaults. The expected six applies and three VALID monitors
are checked against a fully connected reference using accepted kernels and
persistent Adam, not new executor aggregation/receipt helpers. Float64 uses
rtol `1e-10`, atol `1e-12`; float32 uses rtol `2e-5`, atol `2e-6`. Tensor/None,
IDs, counts and order are compared exactly. Additional tests cover zero paths,
source-only roles, barriers, failure lifecycle, mutation isolation and a real
fit-derived one-shot terminal. Actual executed results belong in the external
task evidence, not an implied success claim in this contract.

Hard per-fit caps: 8 parties, 128 items, 32 users, 4 rounds, 32 nonempty steps,
256 positive groups, 2048 labeled rows, 16 fields per party, widths d/r/h at
most 16, K at most 4, and at most 16 VALID targets with 32 items each. Dense
gradient/Adam buffers scale with entire registered tables; preflight, source/H/z
graphs, candidate storage and observation clones also cost memory. No production
scale or recommendation-quality claim follows from this bounded integration.

Deferred: Unit3, private/warmup paths, formal initializer/data/source profiles,
real data/GPU/pilot, durable persistence and recovery, communication/privacy
mechanisms, VFL/HyFL execution and further research units. Existing split,
sampling, feature/graph, numerical kernels and FedGMF/HFL APIs are untouched.
