# Roadmap

## Phase 0 — Legacy analysis

Goal:

Understand the four legacy/current research implementations before writing the unified framework.

Deliverable:

```text
docs/LEGACY_ANALYSIS.md
```

No new algorithm implementation in this phase.

---

## Phase 1 — Project and environment bootstrap

Create:

- project directory structure;
- Conda environment;
- basic package layout;
- pytest setup;
- configuration loading;
- shared seed utility.

Acceptance:

```bash
conda env create -f environment.yml
pytest
```

must succeed.

No research method is required yet.

---

## Phase 2 — Unified data foundation

Implement:

- MovieLens-1M loader;
- Amazon Industrial loader;
- deterministic ID mapping;
- leave-one-out split;
- centralized negative-sampling utility;
- HFL party partition;
- processed-data metadata/checksums.

Reserve interfaces for:

- feature partition;
- VFL partition;
- HyFL partition;
- graph view;
- continual view.

Acceptance:

Two preprocessing runs with the same configuration must produce identical split statistics and mappings.

---

## Phase 3 — Minimal HFL baseline

Implement:

- FedNCF;
- HFL protocol;
- FedAvg baseline;
- HR@10;
- NDCG@10;
- result recorder.

Acceptance:

A complete ML1M FedNCF+FedAvg experiment runs end to end.

Add a tiny synthetic regression/smoke test.

---

## Phase 4 — FedPEFT migration

First migrated thesis method.

Goals:

- preserve core PEFT behavior;
- use shared HFL runner;
- use shared datasets;
- use shared evaluation;
- no method-specific trainer.

Validate legacy-compatible behavior where feasible.

---

## Phase 5 — PKGRec migration

Add:

- graph view;
- PKGRec graph model;
- PKGRec-specific aggregation/update.

Reuse:

- shared datasets;
- HFL runner;
- ranking evaluation.

---

## Phase 6 — DOI4 upgraded method

This is not a mechanical migration.

Develop the new heterogeneous-data method using the unified data representations.

Target:

- HFL implementation first;
- architecture/data contracts must keep VFL/HyFL extension feasible.

---

## Phase 7 — GCPN upgraded method

Reimplement the continual-learning idea in the unified framework.

Target:

- continual HFL first;
- preserve an implementation path to continual VFL/HyFL.

Continual stages should be an outer experimental dimension, not a separate federation protocol.

---

## Phase 8 — Cross-method thesis experiments

Once the four methods are stable:

- unify datasets;
- unify backbones where scientifically valid;
- unify evaluation;
- run shared ablations;
- study interaction among information scope, update strategy, and continual adaptation.
