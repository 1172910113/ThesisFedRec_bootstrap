# Migration Matrix

This is the initial migration plan. It should be updated after code-level inspection of all legacy repositories.

| Research line | Current reference | Migration type | Shared infrastructure | Method-specific core to preserve/upgrade | Initial federation target | Notes |
|---|---|---|---|---|---|---|
| 1.1 DOI4 upgraded | legacy DOI4 code/paper, repository TBD | Reimplementation + extension | data, partition, evaluation, common backbones | heterogeneous-data representation, heterogeneous collaboration, missing/incomplete information handling, adaptive aggregation | HFL first; reserve VFL/HyFL | Do not mechanically port old code |
| 1.2 PKGRec | https://github.com/1172910113/pkg_rec | Migration / clean reimplementation | data, graph construction, evaluation, HFL runner | graph propagation, relation modeling, PKGRec-specific enhancement | HFL | Legacy code may be outdated; treat as behavioral reference |
| 2.1 FedPEFT | https://github.com/young1010/FedPEFT | Migration / modular reimplementation | data, models, evaluation, HFL runner | PEFT embedding strategies, trainable-parameter selection, communication reduction | HFL | Should become a method/module, not a private trainer |
| 2.2 GCPN upgraded | legacy GCPN code/paper, repository TBD | Reimplementation + extension | data, continual stage builder, evaluation, federation protocol | continual adaptation, parameter/model evolution, retention mechanism | HFL first; reserve VFL/HyFL | Continual learning must remain orthogonal to federation type |
| 3.1 FedLSPKD | current work | Out of scope for v0.1 | — | — | — | Do not integrate initially |
| 3.2 FedSLLM | current work | Out of scope for v0.1 | — | — | — | Do not integrate initially |

## Code-inspection checklist

For every legacy method, document:

1. dependency versions;
2. dataset preprocessing;
3. ID remapping;
4. train/validation/test split;
5. negative sampling;
6. definition of client/party;
7. recommendation backbone;
8. local objective;
9. server aggregation;
10. communication payload;
11. evaluation protocol;
12. random-seed handling;
13. algorithmic assumptions that must be preserved;
14. engineering details that can be discarded.

## Migration rule

Do not modernize legacy code in place unless needed for comparison.

The new repository should reimplement the required algorithmic behavior using the unified thesis framework.
