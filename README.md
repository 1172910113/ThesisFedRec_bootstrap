# ThesisFedRec

Unified experimental framework for the PhD thesis on federated recommendation.

## Scope of v0.1

The first version focuses on four research lines:

1. DOI4 upgraded method: heterogeneous-data collaborative recommendation.
2. PKGRec: graph-structured relation modeling for federated recommendation.
3. FedPEFT: parameter-efficient federated recommendation.
4. GCPN upgraded method: continual federated recommendation, with HFL support first and interfaces reserved for VFL/HyFL.

FedLSPKD and FedSLLM are intentionally **out of scope** for v0.1.

## Design goals

- One Conda environment for all four research lines.
- One data foundation and one evaluation protocol.
- HFL is implemented first.
- VFL and HyFL are not required in the first milestone, but core data and federation interfaces must not make them difficult to add later.
- Continual learning is orthogonal to federation type.
- Research methods remain independent under `src/methods/<method_name>/`.
- Prefer explicit, simple code over framework-heavy abstractions.

## Planned structure

```text
ThesisFedRec/
├── environment.yml
├── README.md
├── AGENTS.md
├── configs/
│   ├── datasets/
│   └── methods/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── src/
│   ├── data/
│   ├── federation/
│   ├── models/
│   ├── methods/
│   │   ├── doi4/
│   │   ├── pkgrec/
│   │   ├── fedpeft/
│   │   └── gcpn/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── tests/
└── results/
```

## Initial datasets

Primary thesis benchmarks:

- MovieLens-1M
- Amazon Industrial and Scientific

Optional development dataset:

- MovieLens-100K

All methods should use the same processed dataset representation whenever the research setting permits.

## Initial evaluation

Common ranking metrics:

- HR@10
- NDCG@10

Common efficiency metrics:

- communication volume
- trainable parameter count
- wall-clock training time

Continual-learning methods may additionally report forgetting and stage-wise performance.

## Legacy references

- PKGRec: https://github.com/1172910113/pkg_rec
- FedPEFT: https://github.com/young1010/FedPEFT

Legacy code is treated as algorithmic reference, not as the software foundation of this repository.
