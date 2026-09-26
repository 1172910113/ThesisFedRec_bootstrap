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
