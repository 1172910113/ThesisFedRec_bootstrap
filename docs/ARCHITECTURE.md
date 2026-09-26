# Architecture

## 1. Goal

The repository provides one shared experimental foundation for four thesis methods while keeping each research method independent.

The architecture should support the following relationship:

```text
Dataset
  ↓
Partition / Scenario
  ↓
Federation Protocol
  ↓
Recommendation Backbone
  ↓
Research Method
  ↓
Unified Evaluation
```

The first implementation milestone focuses on HFL. VFL and HyFL are reserved at the interface level but do not need complete implementations initially.

---

## 2. Main layers

### 2.1 Data layer

Location:

```text
src/data/
```

Responsibilities:

- raw dataset loading;
- ID remapping;
- train/validation/test split;
- negative sampling;
- metadata construction;
- graph construction;
- continual-stage construction;
- federated-party partitioning.

The data layer must not import research methods.

The key distinction is:

```text
sample partition
feature partition
```

These are independent.

- HFL: mainly sample/user partition.
- VFL: mainly feature/view partition with overlapping samples/users.
- HyFL: both may differ.

### 2.2 Federation layer

Location:

```text
src/federation/
```

Planned files:

```text
runner.py
protocol.py
hfl.py
vfl.py
hybrid.py
```

Responsibilities:

- orchestration of communication rounds;
- federation-protocol-specific execution;
- interaction with methods and evaluation.

`runner.py` must not contain HFL-specific assumptions.

Initial requirement:

- `HFLProtocol`: implemented and runnable.
- `VFLProtocol`: interface reserved; complete implementation not required.
- `HybridProtocol`: interface reserved; complete implementation not required.

The protocol layer may have different internal procedures. VFL must not be forced through FedAvg-style HFL logic.

### 2.3 Model layer

Location:

```text
src/models/
```

Initial shared backbones:

- FedNCF
- PFedRec, if migration remains straightforward

Method-specific models stay under their own method directory if they are not reusable.

### 2.4 Method layer

Location:

```text
src/methods/
```

```text
doi4/
pkgrec/
fedpeft/
gcpn/
```

Each method owns its algorithm-specific logic.

Methods must not import each other.

A method may support one or more federation protocols.

Examples:

```text
FedPEFT: HFL initially
PKGRec: HFL initially
DOI4 upgraded: HFL + future VFL/HyFL
GCPN upgraded: continual HFL first, future VFL/HyFL adaptation
```

### 2.5 Evaluation layer

Location:

```text
src/evaluation/
```

Shared metrics:

- HR@K
- NDCG@K
- communication
- trainable parameters
- runtime

Continual metrics:

- stage-wise ranking performance
- forgetting

Method-specific metrics may be added, but common metrics and their definitions must not change between methods.

---

## 3. Federation runner

The runner coordinates experiments at a high level.

Conceptually:

```python
state = protocol.setup(dataset, method, config)

for stage in stage_manager:
    method.on_stage_start(stage)

    for round_idx in range(config.num_rounds):
        state = protocol.train_round(
            method=method,
            state=state,
            round_idx=round_idx,
            stage=stage,
        )

    evaluator.evaluate(...)
```

For non-continual experiments, the stage manager contains one stage.

The runner must not assume:

- one client equals one user;
- all parties have identical features;
- all methods exchange model parameters;
- all methods use FedAvg.

---

## 4. Continual learning

Continual learning is not a federation protocol.

It is an outer experimental dimension:

```text
continual × HFL
continual × VFL
continual × HyFL
```

A lightweight stage manager should control:

- stage boundaries;
- stage-specific data views;
- stage-wise evaluation;
- calls to `on_stage_start` and `on_stage_end`.

---

## 5. Simplicity constraints

Do not introduce:

- plugin frameworks;
- registry-heavy designs;
- deep class inheritance;
- method-specific trainer hierarchies;
- generic distributed systems abstractions not needed by the thesis.

Prefer:

- dataclasses;
- plain functions;
- one or two small protocol classes;
- explicit configuration;
- local method implementations.
