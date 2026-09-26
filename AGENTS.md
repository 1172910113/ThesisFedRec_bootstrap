# Project instructions for Codex

## Project purpose

This repository implements a unified experimental framework for a PhD thesis on federated recommendation.

The framework must support four research lines:

- upgraded DOI4 heterogeneous-data method;
- PKGRec;
- FedPEFT;
- upgraded GCPN continual federated recommendation.

FedLSPKD and FedSLLM are not part of the initial implementation.

## Core engineering principles

1. **Keep the code simple.**
   Prefer explicit functions and small classes. Avoid deep inheritance, registries, dependency injection, plugin systems, and unnecessary factories.

2. **Do not abstract prematurely.**
   Do not introduce a shared abstraction unless at least two current methods need it or it is required to preserve HFL/VFL/HyFL extensibility.

3. **Methods must remain independent.**
   Code under `src/methods/<method_name>/` must not import another research method.
   Truly shared functionality belongs in `src/data`, `src/federation`, `src/models`, `src/evaluation`, or `src/utils`.

4. **Do not create method-specific training engines.**
   Shared training/evaluation infrastructure must be reused.
   Method-specific logic should remain inside the method directory.

5. **Do not assume client == user.**
   Users, items, and federated parties are distinct concepts.
   A party can represent a user/device in HFL or an organization/data holder in VFL/HyFL.

6. **Separate sample partition and feature partition.**
   The data layer must be able to describe:
   - sample/user partitioning;
   - feature/view partitioning.
   HFL primarily varies the first; VFL primarily varies the second; HyFL may vary both.

7. **HFL first, but do not hard-code HFL assumptions.**
   The first runnable version should fully support HFL.
   VFL and HyFL may initially raise `NotImplementedError`, but the data model and federation runner must leave a clean implementation path.

8. **Do not fake VFL using the HFL aggregation loop.**
   VFL may require different forward/backward and representation-exchange procedures.

9. **Continual learning is orthogonal to federation type.**
   It should be possible in principle to combine continual learning with HFL, VFL, or HyFL.

10. **Reproducibility is mandatory.**
    Do not silently change:
    - dataset preprocessing;
    - user/item ID mapping;
    - train/validation/test splitting;
    - negative sampling;
    - evaluation candidate construction;
    - random seeds;
    - evaluation metrics.

11. **Legacy code is read-only reference.**
    Do not refactor legacy repositories into the new codebase line-by-line.
    Extract algorithmic behavior and reimplement it using the unified infrastructure.

12. **Every meaningful change requires validation.**
    Add or update tests when changing shared infrastructure.
    Before finishing a task, run the relevant tests and report exactly what passed and failed.

## Required documents

Read these before modifying architecture:

- `docs/ARCHITECTURE.md`
- `docs/DATA_CONTRACT.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/METHOD_INTERFACE.md`
- `docs/MIGRATION_MATRIX.md`
- `docs/ROADMAP.md`

## Coding conventions

- Python 3.10.
- PyTorch is the core deep-learning framework.
- Type hints for shared public interfaces.
- Keep functions short and behavior explicit.
- Prefer dataclasses and plain dictionaries to elaborate configuration frameworks.
- YAML is used for experiment configuration.
- `pytest` is used for tests.
- All random seeds should be set through one shared utility.

## Definition of done for implementation tasks

A task is complete only when:

1. the requested functionality is implemented;
2. relevant tests pass;
3. no existing tests regress;
4. the implementation follows the architecture documents;
5. any changed experiment assumptions are explicitly documented.
