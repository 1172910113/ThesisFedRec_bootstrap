"""One party's sequential graph segment, with caller-owned numerical callbacks."""

from collections.abc import Callable, Iterable
from typing import Any

from src.data import GraphRuntime, GraphView, project_party_graphs


def run_party_graph_segment(
    runtime: GraphRuntime, *,
    party_id: int,
    binding_id: str,
    batches: Iterable[Any],
    prepare_local: Callable[[dict[str, GraphView]], Any],
    step_batch: Callable[[Any, Any], None],
    finish_local: Callable[[Any], Any],
    release_local: Callable[[Any], None],
) -> Any:
    """Project once, prepare once, step lazily, finish and release local state.

    The trusted coordinator supplies the runtime, expected binding and callbacks.
    Only detached graphs/state/batches reach callbacks; the runtime stays open.
    A successful prepare (even returning None) requires exactly one release.
    Failed preparation must clean up its own partial allocations. When operation
    and cleanup both fail, the original exception is raised with cleanup as its
    explicit cause. Cleanup alone failing also prevents a successful return.

    Finish must produce a result usable after release without retaining payload.
    This helper drops its own payload references, not references held elsewhere
    by callbacks, results or tracebacks. It provides no cross-thread exclusion.
    """
    for name, callback in (
        ("prepare_local", prepare_local), ("step_batch", step_batch),
        ("finish_local", finish_local), ("release_local", release_local),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")

    graphs = project_party_graphs(runtime, party_id=party_id, binding_id=binding_id)
    try:
        state = prepare_local(graphs)
    finally:
        del graphs

    iterator = result = None
    try:
        try:
            try:
                iterator = iter(batches)
            finally:
                del batches
            while True:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                try:
                    step_batch(state, batch)
                finally:
                    del batch
            result = finish_local(state)
        except BaseException as primary_error:
            try:
                release_local(state)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        else:
            release_local(state)
            return result
    finally:
        del state, iterator, result
