# Method Interface

## 1. Goal

Each research method should remain self-contained while using shared data, federation, and evaluation infrastructure.

Avoid a large abstract base-class hierarchy.

The first implementation should use a small, explicit interface.

---

## 2. Minimum HFL interface

A method should provide behavior equivalent to:

```python
class Method:
    def initialize(self, dataset, model, config):
        ...

    def client_update(
        self,
        party_id,
        global_state,
        local_data,
        round_idx,
        stage=None,
    ):
        ...

    def server_update(
        self,
        global_state,
        client_updates,
        round_idx,
        stage=None,
    ):
        ...

    def evaluate_state(self):
        ...
```

Exact signatures may be refined during implementation.

The intent is more important than the precise class design.

---

## 3. Optional continual hooks

A continual method may additionally expose:

```python
def on_stage_start(self, stage, state):
    ...

def on_stage_end(self, stage, state):
    ...
```

Non-continual methods do not need to implement meaningful behavior for these hooks.

---

## 4. Protocol compatibility

Each method should declare or document supported federation types.

Examples:

```python
SUPPORTED_PROTOCOLS = {"hfl"}
```

Future DOI4 upgrade:

```python
SUPPORTED_PROTOCOLS = {"hfl", "vfl", "hybrid"}
```

The framework must not require every method to support every protocol.

---

## 5. Separation of responsibility

Shared framework owns:

- processed data;
- party partitions;
- experiment configuration;
- communication-round orchestration;
- common model backbones;
- common ranking evaluation;
- common logging.

Method owns:

- algorithm-specific modules;
- local objective modifications;
- special parameter-update rules;
- special aggregation logic;
- graph propagation;
- continual adaptation behavior.

---

## 6. Method-specific expectations

### DOI4 upgraded

Likely owns:

- heterogeneous representation modules;
- missing-modality/data handling;
- adaptive collaboration/aggregation;
- HyFL-specific research logic when implemented.

The shared data layer only exposes the heterogeneity; it should not encode DOI4's algorithm.

### PKGRec

Likely owns:

- graph propagation;
- relation-aware modeling;
- graph-based recommendation loss;
- PKGRec-specific server aggregation/graph enhancement.

Shared graph preprocessing may stay in `src/data/graph.py`.

### FedPEFT

Should be implemented mainly as:

- trainable-parameter selection;
- embedding PEFT modules;
- communication-payload control.

It should not own a private training engine.

### GCPN upgraded

Likely owns:

- stage-transition behavior;
- model expansion/adaptation modules;
- retention mechanisms.

Continual stage creation belongs to shared data infrastructure.
Federation execution belongs to the selected federation protocol.

---

## 7. No cross-method dependencies

Forbidden:

```python
from methods.pkgrec import ...
```

inside:

```text
methods/gcpn/
methods/fedpeft/
methods/doi4/
```

If two methods genuinely require the same component, first decide whether the component is research-specific or infrastructure-level.

Only infrastructure-level components should be moved into shared directories.

## 8. Public train-only FedGMF warmup (FIT-1)

The implemented entry point is imported from `src.federation.runner`:

```python
fit_fedgmf_hfl(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    config: Mapping[str, Any],
) -> FedGMFHFLFitResult
```

The frozen, slotted, keyword-only result dataclass contains exactly:

```python
final_shared_state: dict[str, torch.Tensor]
final_local_states: dict[int, FedGMFLocalState]
validation_candidates: EvaluationCandidateSet
round_summaries: tuple[FedGMFHFLRoundSummary, ...]
```

It has no test candidates, test metrics, or placeholder test fields. As with
the existing full in-memory runner, callers supply an already validated
configuration and explicit parties; this entry point does not load data,
build partitions, expand configuration schemas, or write result artifacts.
Each call starts fresh, seeds and initializes the model using the existing
scheme, runs all configured rounds, and returns final-round state. Fixed
validation candidates are reused for monitoring each round, never early
stopping or best-round selection. Errors propagate without returning a
partially completed fit result. Inputs are not modified; returned tensors
and dictionaries remain mutable, not deeply frozen or a resume checkpoint.

Test-candidate construction/scoring/evaluation is absent from fit. The
unchanged all-split known-positive sampling exclusion and structural checks
for validation/test user ownership still read held-out IDs. Train-only here
means no test evaluation, not isolation from every held-out identifier.
Party IDs remain distinct from user IDs, including grouped/arbitrary-ID
parties supported by the existing runner.

`run_fedgmf_hfl_experiment()` remains the full-run wrapper: it constructs
validation and test candidates before its shared training loop and evaluates
test once at final state. `execute_phase3_experiment()` retains its existing
artifacts. FIT-1 only exposes reusable warmup state; it does not implement
later research stages, message collection, graph training, Books execution,
or establish M2 readiness, recommendation quality, or full Books scalability.
