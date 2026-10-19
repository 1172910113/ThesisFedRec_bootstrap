# G3-A: one party's graph segment

This bounded public primitive adds lifecycle orchestration over G2. It is not
a graph trainer, a complete G3 implementation, COLLECT-12, or M2. The existing
FIT-1, FedGMF and Phase 3 runner paths are unchanged; none invokes this helper.

## Public API and responsibilities

Import `run_party_graph_segment` from `src.federation` or
`src.federation.graph_segment`:

```python
run_party_graph_segment(
    runtime: GraphRuntime, *,
    party_id: int,
    binding_id: str,
    batches: Iterable[Any],
    prepare_local: Callable[[dict[str, GraphView]], Any],
    step_batch: Callable[[Any, Any], None],
    finish_local: Callable[[Any], Any],
    release_local: Callable[[Any], None],
) -> Any
```

The trusted coordinator prepares and owns the G2 runtime, holds the expected
binding ID, supplies the batch stream and assembles the four trusted callbacks.
It closes the runtime explicitly when finished with all segments. The helper
passes only detached graphs to preparation, then opaque state and batches to
the other callbacks; it does not deliver runtime identities, indexes, source
metadata or canonical raw node IDs. Public vocabularies and complete processed
anchor mapping domains are still delivered under G2's rules. A callback's own
captured references remain its caller's responsibility. This is a software delivery boundary, not
a security boundary against hostile Python or a cryptographic privacy claim.

The helper uses only the public G2 projection path. Existing G2 ID, binding,
party, selection, local relabeling and detachment rules apply unchanged. Party
IDs need not be contiguous and are not user IDs. Source/sample ownership and
future model consumption are independent of this lifecycle operation.

## Execution and errors

1. Check that all four callbacks are callable without calling any of them.
   A non-callable raises `TypeError` before projection or stream effects.
2. Call `project_party_graphs` exactly once with the supplied runtime, party ID
   and expected binding. Invalid runtime/binding/party fails through G2 before
   preparation or obtaining the batch iterator.
3. Call `prepare_local(graphs)` exactly once. A valid no-access party receives
   `{}`; a visible empty graph remains an entry in that dictionary. Preparation
   returns opaque state, including `None` if appropriate. Immediately drop the
   helper's graph-dictionary reference, including when preparation raises.
4. Only after preparation, obtain the iterator and consume one batch at a time
   in its yielded order. Call `step_batch(state, batch)` exactly once per batch
   with the same state object throughout. Discard its return value and the
   helper's reference to that batch before advancing. Do not prefetch, materialize
   the stream or retain batch history. `StopIteration` from iterator advancement
   means normal exhaustion; a callback's `StopIteration` is a callback failure.
5. On normal exhaustion, call `finish_local(state)` exactly once, then
   `release_local(state)` exactly once, and only then return the finish result.
   Zero batches still project, prepare, finish and release once.
6. After successful preparation, iterator creation/advancement, step or finish
   failure still calls release exactly once. This also applies to `BaseException`
   subclasses such as interrupts. Failed preparation has no returned state to
   release: the preparation callback must release its own partial allocations.
7. If operation and release both fail, re-raise the original exception object
   with the cleanup exception as its explicit cause (`raise primary from cleanup`).
   The primary failure remains the raised exception; cleanup is not suppressed.
   A release-only failure is propagated and prevents a successful return.

The caller's runtime is never closed or replaced, even on callback failure.
Independent later segments may reuse it. A batch iterable/iterator is also
caller-owned: the helper does not invoke its optional `close()` method.

## Lifetime and bounded retention

Preparation may construct method-owned tensors and persistent optimizer-like
state once; the helper neither creates these nor recreates them per batch.
`release_local` owns method resource cleanup. The finish callback must return a
result usable after release **without retaining the segment payload**, for
example a detached small summary rather than the prepared state itself. No
generic introspection, deep copying or result ownership enforcement is attempted.

For sequential use the helper holds at most one active segment: one projected
dictionary until preparation returns, then one opaque state, current iterator
and current batch, plus the finish result during cleanup. It drops these local
payload references on exit. Prepared state may intentionally own projected
arrays; that is not an additional helper cache. There is no all-party payload
cache, retained batch history, LRU or cross-segment prepared-state reuse.

G2 still owns the reusable snapshot/index buffers for all referenced views.
Projection allocates the requested local graphs, including complete processed
user/item mapping arrays when those mappings exist. Their memory remains
O(processed users + processed items), even for small local selections. This
helper does not eliminate G2 setup costs or change G2's projection memory bounds.
It adds no source rehash, global validation, index construction, strict-G1b
resolution or unrelated-party/view scan after runtime preparation. Output graph
validation inside G2 still occurs for the selected payload.

Resource bounds of arbitrary callbacks, batch iterators and results are caller
responsibilities. Callers, closures, returned objects and exception tracebacks
can retain references after the helper releases its own. This code cannot force
destruction, reclaim allocator/RSS pages, revoke a caller's retained payload or
coordinate arbitrary concurrent callers. No locks, global leak detector or
cross-thread exclusion guarantee is introduced.

## Validation and boundaries

`tests/test_graph_segment.py` covers lazy event order, zero/one/many batches,
non-contiguous parties, absent/empty graphs, opaque state identity, callback and
G2 request validation, preparation/iterator/step/finish/cleanup failures, primary
exception preservation, runtime reuse and reference/mutation isolation. A direct
test-owned numerical reference loop checks repeated updates without using the
production segment loop. Instrumentation of actual G2/global aliases checks
that repeated 1/10/100-batch segments add no global work or unrelated projections.

The PARALLEL-W1 implementation gate runs these tests and named existing graph,
FIT-1, runner and FedGMF compatibility tests, with CPU synthetic inputs and real
data opt-ins unset. It does **not** run the default full suite. The separately
scheduled combined-candidate gate owns that later validation.

A separate external synthetic resource probe records small/larger graph cases
and sequential parties in fresh Python processes: source, snapshot, index,
projection and prepared-state unique backing buffers, plus process peak RSS.
Buffer accounting excludes Python object overhead and temporary scratch space;
RSS includes imports, allocator retention and other process allocations. Neither
timing nor exact RSS reclamation is a pass/fail threshold. This evidence is not
a Books-scale benchmark or a training-performance claim.

Deferred: method update rules, graph normalization/sampling, graph builders,
training integration, coordinator scheduling, COLLECT-12/full G3/M2, persistence,
resume, distributed execution, revocation and concurrent lifetime management.
