# Codex Bootstrap Task

Use this as the first implementation task after the repository is created.

## Task

Read:

- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA_CONTRACT.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/METHOD_INTERFACE.md`
- `docs/MIGRATION_MATRIX.md`
- `docs/ROADMAP.md`

Then create only the project skeleton for Phase 1.

## Requirements

Create:

```text
src/
├── data/
├── federation/
├── models/
├── methods/
│   ├── doi4/
│   ├── pkgrec/
│   ├── fedpeft/
│   └── gcpn/
├── evaluation/
└── utils/

scripts/
tests/
data/raw/
data/processed/
results/
```

Add minimal Python package files where needed.

Implement only:

1. YAML config loading;
2. shared random-seed utility;
3. basic logging helper;
4. one trivial pytest test verifying package import and deterministic seed behavior.

Do **not** implement:

- recommendation models;
- dataset loaders;
- FedAvg;
- HFL/VFL/HyFL protocols;
- any thesis method.

## Simplicity constraint

Do not add:

- registries;
- dependency injection;
- plugins;
- generic factories;
- deep inheritance;
- Hydra or other configuration frameworks.

Plain YAML + Python dictionaries/dataclasses are sufficient.

## Done when

The task is complete when:

```bash
conda env create -f environment.yml
conda activate thesis-fedrec
pytest -q
```

runs successfully, and the repository tree matches the architecture documents.

Report:

- files created;
- tests run;
- exact test result;
- any environment issue encountered.
