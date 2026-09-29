You are continuing the 10-day unattended ThesisFedRec workflow.

Work only on branch:

codex/unattended-7days

Read AGENTS.md and docs/UNATTENDED_STATUS.md before doing substantial work.

The stable baseline is:

- commit: 6b0a994
- tag: v0.2.1-ml1m-data
- Phase 1 completed
- Phase 2A DATA_CONTRACT finalized
- Phase 2B ML1M pipeline completed
- target-safe negative sampling completed
- baseline test count: 19 passing tests

Do not redo completed work.

## Ordered gates

### Gate 2 — Phase 2B audit

Audit the completed ML1M data foundation:

- preprocessing invariants
- ID mapping
- duplicate handling
- temporal split
- cache round trip
- per_user HFL partition
- grouped_round_robin partition
- negative sampling
- data leakage
- determinism
- obvious memory/time inefficiencies

Allowed:

- add tests for already-approved behavior
- fix only clearly demonstrated implementation defects
- improve documentation

Do not change DATA_CONTRACT semantics.

Write:

docs/PHASE2B_AUDIT.md

### Gate 3 — Legacy analysis

Perform deep read-only code analysis of:

PKGRec:
https://github.com/1172910113/pkg_rec

FedPEFT:
https://github.com/young1010/FedPEFT

Do not modify those legacy repositories.

Write:

docs/legacy/PKGREC_ANALYSIS.md
docs/legacy/FEDPEFT_ANALYSIS.md

For each analyze:

- environment/dependencies
- datasets
- preprocessing
- ID mapping
- split
- negative sampling
- client/party definition
- backbone
- model parameters
- local objective
- local update
- server aggregation
- communication payload
- evaluation
- seed handling
- algorithmic components that must be preserved
- obsolete engineering that should not be migrated
- compatibility risks with ThesisFedRec

If paper/code behavior requires scientific interpretation, record a blocker.
Do not guess.

### Gate 4 — Migration plans

Update:

docs/MIGRATION_MATRIX.md

Create:

docs/exec-plans/fedpeft_migration.md
docs/exec-plans/pkgrec_migration.md

Do not implement PKGRec or FedPEFT.

### Gate 5 — Minimal Phase 3 design

Design only:

ML1M
+ per-user HFL
+ FedNCF
+ FedAvg
+ HR@10
+ NDCG@10

Write:

docs/PHASE3_BASELINE_DESIGN.md

Keep the architecture minimal.

Do not assume party and user are the same concept.

Do not implement VFL or HyFL, but do not create interfaces that prevent
their later implementation.

### Gate 6 — Minimal baseline implementation

Only after Gate 5 is complete and internally consistent, implement:

- minimal FedNCF
- HFL protocol
- FedAvg
- shared HR@10 / NDCG@10 evaluation
- training entry point
- synthetic end-to-end tests
- small real ML1M smoke test

Do not implement:

- PFedRec
- Amazon
- PKGRec
- FedPEFT
- DOI4
- GCPN
- VFL
- HyFL
- graph learning
- continual learning

## Absolute restrictions

Do not:

- modify anything under /data_nfs/yuanhaochen/data
- download or replace datasets
- start Amazon Phase 2C
- change the Conda environment
- redesign DATA_CONTRACT
- make new scientific decisions
- introduce registries, plugin frameworks, dependency injection, or deep inheritance
- merge or push to main
- force push
- perform destructive Git operations

## Blocker conditions

If work requires:

- a new scientific assumption
- interpretation of conflicting paper/code behavior
- changing evaluation semantics
- changing DATA_CONTRACT
- architecture changes affecting multiple thesis methods
- researcher approval

record it in:

docs/UNATTENDED_BLOCKERS.md

Do not guess.

If possible, continue another already-approved safe task.

## Validation

Before every code commit:

python -m pytest -q
git diff --check

Never commit failing code.

Use small atomic commits.

Push successful commits only to:

origin/codex/unattended-7days

## Status tracking

Maintain:

docs/UNATTENDED_STATUS.md

After every run record:

- current gate
- task completed this run
- files changed
- last validated commit
- exact test result
- blockers
- scientific decisions required
- next safe task

Complete only one reasonably scoped task per invocation.

Do not try to complete all remaining gates in a single run.

If all allowed gates finish early:

- audit code
- strengthen tests for approved behavior
- improve documentation
- do not begin another milestone.

The UNATTENDED_STATUS.md update is part of the current run.
Before ending the run, include all intended repository changes,
including the status update, in a validated commit and push that
commit to origin/codex/unattended-7days.

Do not leave intended changes uncommitted at the end of a successful run.