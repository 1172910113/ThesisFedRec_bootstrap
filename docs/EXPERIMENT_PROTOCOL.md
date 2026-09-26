# Experiment Protocol

## 1. Purpose

This document defines the thesis-wide default experimental protocol.

The goal is to make results comparable across methods without forcing methods into an invalid research setting.

---

## 2. Default recommendation setting

Task:

```text
implicit-feedback top-K recommendation
```

Default primary metrics:

```text
HR@10
NDCG@10
```

Optional additional K values may be reported, but HR@10 and NDCG@10 remain the common thesis metrics.

---

## 3. Datasets

Primary:

- MovieLens-1M
- Amazon Industrial and Scientific

Optional development/smoke-test dataset:

- MovieLens-100K

All methods should run on both primary datasets whenever their required information is available.

---

## 4. Split

Default:

```text
leave-one-out with temporal ordering
```

Per user:

1. latest interaction → test;
2. second latest → validation;
3. remaining → train.

Any deviation must be documented in the experiment configuration and result metadata.

---

## 5. Random seeds

Use one shared seed utility.

Recommended thesis seed set:

```text
2026
2027
2028
2029
2030
```

Final reported results should use mean ± standard deviation where feasible.

Debug runs may use only one seed.

---

## 6. Federated training

Common configuration fields:

```yaml
federation:
  type: hfl
  rounds: 20
  client_fraction: 1.0
  local_epochs: 2
```

These are defaults, not immutable constants.

Every experiment must log:

- number of rounds;
- selected party/client fraction;
- local epochs;
- optimizer;
- learning rate;
- batch size;
- random seed.

---

## 7. Evaluation

Evaluation code must be shared across methods.

Method-specific code must not implement private copies of HR/NDCG unless required for debugging.

The evaluator must clearly define:

- candidate construction;
- ranking direction;
- handling of ties;
- treatment of seen items.

---

## 8. Efficiency reporting

Common efficiency metrics:

```text
communication volume
trainable parameter count
wall-clock runtime
```

Where relevant, additionally report:

```text
number of locally optimized samples
peak GPU memory
```

Communication should be counted from the actual payload where practical rather than inferred only from theoretical parameter counts.

---

## 9. Continual-learning reporting

For GCPN-style continual experiments, additionally report:

- per-stage HR@10 and NDCG@10;
- average current-stage performance;
- historical-stage retention;
- forgetting.

The definition of forgetting must remain fixed once adopted.

---

## 10. Result format

Every run should produce:

```text
results/<experiment_name>/
├── config.yaml
├── metrics.json
├── round_metrics.csv
├── runtime.json
└── checkpoint.pt   # optional
```

`metrics.json` should contain common keys when applicable:

```json
{
  "HR@10": 0.0,
  "NDCG@10": 0.0,
  "communication_mb": 0.0,
  "trainable_params": 0,
  "training_time_sec": 0.0
}
```

Continual experiments may add:

```json
{
  "forgetting": 0.0,
  "stage_metrics": []
}
```

---

## 11. Legacy reproduction vs thesis evaluation

When migrating an existing published method, distinguish:

### Legacy-compatible reproduction

Use original assumptions as closely as possible to check algorithmic fidelity.

### Thesis unified evaluation

Use the unified datasets, split, metrics, and runtime conventions.

Do not tune or alter the migrated method merely to hide discrepancies between legacy and unified results.
