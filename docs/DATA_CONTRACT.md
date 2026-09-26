# Data Contract

## 1. Purpose

All thesis methods should consume a consistent processed-data representation whenever their research setting permits.

The data model must support HFL now and leave a clean path to VFL and HyFL later.

---

## 2. Core entities

The framework distinguishes:

- user;
- item;
- federated party.

A federated party is not necessarily a user.

Notation:

```text
user_id
item_id
party_id
```

All processed IDs must be contiguous integer IDs starting from zero unless a specific method has a documented reason not to do so.

Raw IDs must be preserved in mapping files.

---

## 3. DatasetBundle

The conceptual processed dataset contains:

```python
DatasetBundle:
    users
    items
    interactions

    user_features
    item_features
    item_text

    graph
    timestamps

    train
    valid
    test

    parties
    metadata
```

Fields may be `None` when unavailable.

Research methods should read from this shared representation rather than implement their own raw-data parsing.

---

## 4. PartyData

Each federated party should be describable as:

```python
PartyData:
    party_id
    user_ids
    item_ids
    interaction_ids
    feature_names
    local_features
```

Not every field is required in every scenario.

This representation must support:

### HFL

Different parties mainly hold different samples/users.

### VFL

Different parties may cover overlapping users/samples but hold different feature subsets or views.

### HyFL

Both sample coverage and feature coverage may differ.

---

## 5. Partitioning

Partition logic belongs in:

```text
src/data/partition.py
```

Planned functions:

```python
build_hfl_partition(...)
build_vfl_partition(...)
build_hybrid_partition(...)
```

Only HFL must be implemented in the first milestone.

VFL/HyFL functions may initially be placeholders, but the data representation must make their future implementation possible without changing the core dataset schema.

---

## 6. Common datasets

### Primary

- MovieLens-1M
- Amazon Industrial and Scientific

### Optional development dataset

- MovieLens-100K

The same processed dataset should be reusable through multiple views.

Examples:

```text
interaction view
feature view
graph view
continual-stage view
```

The goal is not to create different preprocessing pipelines for each method.

---

## 7. Train/validation/test split

Default implicit-recommendation protocol:

- latest interaction → test;
- second latest interaction → validation;
- remaining interactions → training.

The exact implementation must be deterministic.

Users without enough interactions must be handled by one documented rule.

A method may require a different split only when its research question makes the default invalid. Such exceptions must be documented explicitly.

---

## 8. Negative sampling

Training negative sampling and evaluation candidate construction must be defined centrally.

Method implementations must not silently redefine negative sampling.

Configuration should expose the relevant parameters, for example:

```yaml
negative_sampling:
  train_ratio: 4
  strategy: uniform
```

Evaluation should follow one thesis-wide protocol unless an exception is explicitly documented.

---

## 9. Feature and modality representation

Feature availability must be represented explicitly.

For party \(c\), conceptually:

```text
feature_names_c
modalities_c
```

Missing features/modalities should not be silently filled unless the experiment specifically defines an imputation rule.

This is especially important for the upgraded DOI4 heterogeneous-data study.

---

## 10. Graph view

PKGRec-compatible graph data should be built from the same processed dataset.

Conceptually:

```text
nodes
edges
relation_types
node_features
```

Graph construction belongs in shared data utilities if it is dataset-level preprocessing.

PKGRec-specific propagation and graph learning remain inside `src/methods/pkgrec/`.

---

## 11. Continual view

Continual stages should be derived from the same base dataset.

Conceptually:

```python
ContinualView:
    stages
    stage_train
    stage_valid
    stage_test
```

Possible stage criteria include:

- chronological windows;
- domain/category increments;
- controlled distribution shifts.

The exact definition must be configured and logged.

---

## 12. Reproducibility artifacts

Each processed dataset should save:

- raw-to-processed ID mappings;
- preprocessing configuration;
- split metadata;
- random seed;
- counts of users/items/interactions;
- optional checksum/version marker.

Two runs with the same raw data and configuration should produce identical processed splits.
