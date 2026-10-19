"""G3-A callback lifecycle, lazy streaming, isolation and public G2 reuse."""

from copy import deepcopy
import weakref

import numpy as np
import pytest

from src.data import close_graph_runtime, prepare_graph_runtime, project_party_graphs
from src.data import graph as graph_module
from src.data import graph_runtime as runtime_module
from src.federation import run_party_graph_segment
from src.federation import graph_segment as segment_module
from test_graph_resolution import ARRAY_FIELDS, _access, _assignment, _graph, _ids, bundle


def _prepare(bundle, accesses=None):
    if accesses is None:
        accesses = {7: (_access(),), 41: (_access([1, 2], [1]),), 901: ()}
    partition, parties = _assignment(bundle, accesses)
    return prepare_graph_runtime(
        bundle, partition, parties, source_id="synthetic-v1", split_id="split-v1",
        recipe_id="segment-v1",
    )


@pytest.fixture
def runtime(bundle):
    handle = _prepare(bundle)
    yield handle
    close_graph_runtime(handle)


def _run(runtime, **overrides):
    arguments = dict(
        party_id=7, binding_id=runtime.binding_id, batches=(),
        prepare_local=lambda graphs: {}, step_batch=lambda state, batch: None,
        finish_local=lambda state: None, release_local=lambda state: None,
    )
    arguments.update(overrides)
    return run_party_graph_segment(runtime, **arguments)


@pytest.mark.parametrize("count", [0, 1, 10, 100])
@pytest.mark.parametrize("party_id", [7, 41, 901])
def test_public_export_order_and_lazy_same_state(runtime, monkeypatch, count, party_id):
    assert run_party_graph_segment is segment_module.run_party_graph_segment
    events, states = [], []
    result = object()
    original_project = segment_module.project_party_graphs

    def project(*args, **kwargs):
        assert args == (runtime,)
        assert kwargs == {"party_id": party_id, "binding_id": runtime.binding_id}
        events.append("project")
        return original_project(*args, **kwargs)

    monkeypatch.setattr(segment_module, "project_party_graphs", project)

    class Batches:
        position = 0

        def __iter__(self):
            assert events == ["project", "prepare"]
            events.append("iter")
            return self

        def __next__(self):
            if self.position == count:
                events.append("exhausted")
                raise StopIteration
            value = self.position
            # No prefetch: the preceding batch has already been stepped.
            assert value == 0 or events[-1] == ("step", value - 1)
            events.append(("yield", value))
            self.position += 1
            return value

    def prepare(graphs):
        assert list(graphs) == ([] if party_id == 901 else ["relations"])
        if graphs:
            assert len(graphs["relations"].node_raw_ids) == (4 if party_id == 7 else 2)
        events.append("prepare")
        state = {"steps": 0}
        states.append(state)
        return state

    def step(state, batch):
        assert state is states[0]
        assert state["steps"] == batch
        state["steps"] += 1
        events.append(("step", batch))

    def finish(state):
        assert state is states[0] and state["steps"] == count
        events.append("finish")
        return result

    def release(state):
        assert state is states[0]
        events.append("release")
        state.clear()

    assert _run(runtime, party_id=party_id, batches=Batches(), prepare_local=prepare,
                step_batch=step, finish_local=finish, release_local=release) is result
    expected = ["project", "prepare", "iter"]
    for value in range(count):
        expected.extend([("yield", value), ("step", value)])
    assert events == expected + ["exhausted", "finish", "release"]
    assert states == [{}]


def test_visible_empty_graph_is_not_absent_and_none_state_is_valid(bundle):
    bundle.graph_views = {"relations": _graph(
        node_raw_ids=(), node_type_ids=_ids(), edge_index=np.empty((2, 0), dtype=np.int64),
        relation_ids=_ids(), user_to_node=_ids([-1, -1]), item_to_node=_ids([-1, -1, -1]),
    )}
    runtime = _prepare(bundle, {41: (_access(),)})
    events = []

    def prepare(graphs):
        assert list(graphs) == ["relations"]
        assert graphs["relations"].edge_index.shape == (2, 0)
        events.append("prepare")
        return None

    def step(state, batch):
        assert state is None
        events.append("step")

    def finish(state):
        assert state is None
        events.append("finish")
        return "done"

    def release(state):
        assert state is None
        events.append("release")

    assert _run(runtime, party_id=41, batches=[object()], prepare_local=prepare,
                step_batch=step, finish_local=finish, release_local=release) == "done"
    assert events == ["prepare", "step", "finish", "release"]
    close_graph_runtime(runtime)


@pytest.mark.parametrize("name", ["prepare_local", "step_batch", "finish_local", "release_local"])
@pytest.mark.parametrize("value", [None, 0, object()])
def test_all_callbacks_checked_before_projection_or_effects(runtime, monkeypatch, name, value):
    def forbidden(*args, **kwargs):
        pytest.fail("no projection or callback before checking all callables")

    monkeypatch.setattr(segment_module, "project_party_graphs", forbidden)
    callbacks = dict.fromkeys(("prepare_local", "step_batch", "finish_local", "release_local"), forbidden)
    callbacks[name] = value
    with pytest.raises(TypeError, match=f"{name} must be callable"):
        _run(runtime, batches=None, **callbacks)


@pytest.mark.parametrize("invalid", ["runtime", "binding", "binding_type", "closed", "unknown", "party_type"])
def test_runtime_request_failure_precedes_prepare_and_iteration(runtime, invalid):
    binding = runtime.binding_id
    arguments = dict(party_id=7, binding_id=binding)
    if invalid == "runtime":
        runtime = object()
    elif invalid == "binding":
        arguments["binding_id"] = "wrong"
    elif invalid == "binding_type":
        arguments["binding_id"] = None
    elif invalid == "closed":
        close_graph_runtime(runtime)
    elif invalid == "unknown":
        arguments["party_id"] = 42
    else:
        arguments["party_id"] = np.int64(7)

    def forbidden(*args):
        pytest.fail("invalid runtime request must have no callback/batch effects")

    class Batches:
        __iter__ = forbidden

    with pytest.raises((TypeError, ValueError)):
        run_party_graph_segment(
            runtime, **arguments, batches=Batches(), prepare_local=forbidden,
            step_batch=forbidden, finish_local=forbidden, release_local=forbidden,
        )


class _Cancelled(BaseException):
    """Also exercise cleanup for failures outside Exception."""


@pytest.mark.parametrize("stage", ["prepare", "iter", "next", "step", "finish"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, _Cancelled])
def test_primary_failure_and_exactly_once_cleanup(runtime, stage, cleanup_fails, error_type):
    primary, cleanup = error_type(stage), RuntimeError("cleanup")
    events = []
    state = object()

    def visit(name):
        events.append(name)
        if name == stage:
            raise primary

    def prepare(graphs):
        visit("prepare")
        return state

    class Batches:
        position = 0

        def __iter__(self):
            visit("iter")
            return self

        def __next__(self):
            if self.position:
                raise StopIteration
            self.position += 1
            visit("next")
            return "batch"

    def step(received, batch):
        assert received is state and batch == "batch"
        visit("step")

    def finish(received):
        assert received is state
        visit("finish")

    def release(received):
        assert received is state
        events.append("release")
        if cleanup_fails:
            raise cleanup

    with pytest.raises(error_type) as error:
        _run(runtime, batches=Batches(), prepare_local=prepare, step_batch=step,
             finish_local=finish, release_local=release)
    assert error.value is primary
    order = ["prepare", "iter", "next", "step", "finish"]
    assert events == order[:order.index(stage) + 1] + ([] if stage == "prepare" else ["release"])
    if cleanup_fails and stage != "prepare":
        assert error.value.__cause__ is cleanup
    else:
        assert error.value.__cause__ is None
    # Neither the failure nor cleanup closes/replaces the coordinator's handle.
    assert _run(runtime, finish_local=lambda received: "reusable") == "reusable"


@pytest.mark.parametrize("count", [0, 3])
def test_cleanup_only_failure_prevents_return(runtime, count):
    events = []
    cleanup = RuntimeError("release failed")

    def finish(state):
        events.append("finish")
        return "must not return"

    def release(state):
        events.append("release")
        raise cleanup

    with pytest.raises(RuntimeError) as error:
        _run(runtime, batches=range(count), finish_local=finish, release_local=release)
    assert error.value is cleanup
    assert events == ["finish", "release"]


@pytest.mark.parametrize("callback", ["step_batch", "finish_local"])
def test_callback_stop_iteration_is_not_stream_exhaustion(runtime, callback):
    primary = StopIteration("callback failure")
    events = []

    def fail(*args):
        raise primary

    def finish(state):
        events.append("finish")

    def release(state):
        events.append("release")

    callbacks = dict(finish_local=finish, release_local=release)
    callbacks[callback] = fail
    with pytest.raises(StopIteration) as error:
        _run(runtime, batches=[0], **callbacks)
    assert error.value is primary
    assert events == ["release"]


def test_helper_drops_projection_state_and_past_batch_references(runtime):
    graph_refs, state_refs, batch_refs = [], [], []

    class LocalState:
        pass

    class Batch:
        pass

    class Batches:
        position = 0

        def __iter__(self):
            assert graph_refs and all(ref() is None for ref in graph_refs)
            return self

        def __next__(self):
            assert all(ref() is None for ref in batch_refs)
            if self.position == 3:
                raise StopIteration
            self.position += 1
            batch = Batch()
            batch_refs.append(weakref.ref(batch))
            return batch

    def prepare(graphs):
        graph_refs.extend(weakref.ref(getattr(graphs["relations"], name)) for name in ARRAY_FIELDS)
        state = LocalState()
        state_refs.append(weakref.ref(state))
        return state

    def release(state):
        assert state is state_refs[0]()
        assert all(ref() is None for ref in graph_refs + batch_refs)

    assert _run(runtime, batches=Batches(), prepare_local=prepare,
                finish_local=lambda state: 123, release_local=release) == 123
    assert all(ref() is None for ref in graph_refs + state_refs + batch_refs)


@pytest.mark.parametrize("fail", [False, True])
def test_callback_payload_mutation_does_not_change_sources_or_other_deliveries(bundle, runtime, fail):
    source = deepcopy(bundle.graph_views["relations"])
    previous = project_party_graphs(runtime, party_id=7, binding_id=runtime.binding_id)

    def prepare(graphs):
        graph = graphs["relations"]
        for name in ARRAY_FIELDS:
            getattr(graph, name).fill(-999)
        graph.metadata["changed"] = ["local"]
        graphs.clear()
        return None

    def finish(state):
        if fail:
            raise RuntimeError("local failure")

    if fail:
        with pytest.raises(RuntimeError, match="local failure"):
            _run(runtime, prepare_local=prepare, finish_local=finish)
    else:
        _run(runtime, prepare_local=prepare, finish_local=finish)
    fresh = project_party_graphs(runtime, party_id=7, binding_id=runtime.binding_id)
    partial = project_party_graphs(runtime, party_id=41, binding_id=runtime.binding_id)
    assert len(partial["relations"].node_raw_ids) == 2
    close_graph_runtime(runtime)
    # Both detached outputs also outlive explicit retirement by the coordinator.
    for graph in (previous["relations"], fresh["relations"], bundle.graph_views["relations"]):
        graph.validate()
        for name in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(graph, name), getattr(source, name))
    assert previous["relations"].metadata == fresh["relations"].metadata == {}
    assert bundle.graph_views["relations"].metadata == source.metadata


def test_repeated_segments_do_not_repeat_global_work_or_visit_other_parties(bundle, monkeypatch):
    bundle.graph_views["other"] = _graph(name="other")
    counters = {}

    def instrument(module, name):
        original = getattr(module, name)
        key = (module.__name__, name)
        counters[key] = 0

        def counted(*args, **kwargs):
            counters[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    for module, names in (
        (runtime_module, ("validate_graph_partition", "graph_domain_fingerprint",
                          "graph_partition_checksum", "_validate_graph_sources",
                          "_build_graph_projection_index", "_copy_graph", "_json_fingerprint",
                          "_validate_graph_delivery", "_validate_graph_delivery_names")),
        (graph_module, ("graph_view_fingerprint", "graph_domain_fingerprint",
                        "_validate_graph_sources", "_build_graph_projection_index",
                        "_build_graph_mapping_index", "_json_fingerprint",
                        "resolve_party_graph_views", "validate_graph_partition",
                        "graph_partition_checksum", "_array_identity",
                        "_validate_graph_delivery", "_validate_graph_delivery_names")),
    ):
        for name in names:
            instrument(module, name)
    runtime = _prepare(bundle, {7: (_access(),), 41: (_access(name="other"),), 901: ()})
    baseline = dict(counters)
    assert counters[(runtime_module.__name__, "_build_graph_projection_index")] == 2
    assert counters[(runtime_module.__name__, "validate_graph_partition")] == 1
    projections, selected_views = [], []
    project = segment_module.project_party_graphs
    project_view = runtime_module._project_graph_view

    def counted_project(handle, **kwargs):
        projections.append(kwargs["party_id"])
        return project(handle, **kwargs)

    def counted_view(view, access, index):
        selected_views.append(view.name)
        return project_view(view, access, index)

    monkeypatch.setattr(segment_module, "project_party_graphs", counted_project)
    monkeypatch.setattr(runtime_module, "_project_graph_view", counted_view)
    for count in (1, 10, 100):
        for _ in range(2):
            _run(runtime, batches=range(count))
            _run(runtime, party_id=901, batches=range(count))
            assert counters == baseline
    assert projections == [7, 901] * 6
    assert selected_views == ["relations"] * 6
    close_graph_runtime(runtime)


@pytest.mark.parametrize("count", [0, 1, 10, 100])
@pytest.mark.parametrize("party_id,selected_edges", [(7, [0, 1, 2]), (41, [1]), (901, [])])
def test_numerical_result_matches_independent_direct_loop(bundle, runtime, count, party_id, selected_edges):
    # An intentionally tiny test-owned momentum-like update, NOT a method/optimizer.
    # Reference reads the declared source edge selection, never production projection.
    source = bundle.graph_views["relations"]
    edge_signal = sum(int(source.relation_ids[edge]) + 1 for edge in selected_edges)
    batches = tuple((float(i % 5) / 4, float(i % 3) / 2) for i in range(count))
    weight, velocity = 0.25, 0.0
    for value, target in batches:
        gradient = (weight * value + edge_signal - target) * value
        velocity = 0.5 * velocity + gradient
        weight -= 0.125 * velocity
    expected = {"weight": weight, "velocity": velocity, "steps": count}
    state_ids = []

    def prepare(graphs):
        signal = sum(int(relation) + 1 for graph in graphs.values() for relation in graph.relation_ids)
        state = dict(weight=0.25, velocity=0.0, steps=0, signal=signal)
        state_ids.append(id(state))
        return state

    def step(state, batch):
        assert id(state) == state_ids[0]
        value, target = batch
        gradient = (state["weight"] * value + state["signal"] - target) * value
        state["velocity"] = 0.5 * state["velocity"] + gradient
        state["weight"] -= 0.125 * state["velocity"]
        state["steps"] += 1

    def finish(state):
        return {name: state[name] for name in ("weight", "velocity", "steps")}

    def release(state):
        assert id(state) == state_ids[0]
        state.clear()

    result = _run(runtime, party_id=party_id, batches=iter(batches), prepare_local=prepare,
                  step_batch=step, finish_local=finish, release_local=release)
    assert result == expected
    assert batches == tuple((float(i % 5) / 4, float(i % 3) / 2) for i in range(count))
