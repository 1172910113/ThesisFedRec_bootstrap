# Phase 3 Baseline Validation

## Validation record

- Date: 2026-10-12 UTC
- Code commit tested: `81a90d7b2c6e07ebd5635300932900a2f7f70e7a`
- Branch: `codex/phase3f-validation`
- Normal test suite: `129 passed, 1 skipped in 2.88s`
- Focused synthetic E2E and smoke-config validation: `2 passed in 2.05s`
- Conclusion: **PASS**

The skipped normal-suite test is the intentionally opt-in real ML1M/GPU
smoke test. The synthetic E2E test exercised validated configuration, real
HFL partitioning, FedGMF initialization and local training, synchronous
FedAvg, shared ranking evaluation, artifact writing, and same-seed replay on
CPU. Only the raw/cache resolution boundary was replaced by a synthetic
`DatasetBundle`.

## Real ML1M smoke

Command:

```bash
THESISFEDREC_DATA_ROOT=/data_nfs/yuanhaochen/data \
/home/yuanhaochen/miniconda3/envs/thesis-fedrec/bin/python \
  scripts/run_experiment.py \
  --config configs/experiments/ml1m_fedgmf_fedavg_smoke.yaml
```

Dataset identity:

- dataset: MovieLens-1M, schema `1.0`, raw input;
- ratings reference: `ml-1m/ml-1m/ratings.dat`;
- SHA-256: `506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f`;
- processed counts: 6,040 users, 3,706 items, 1,000,209 interactions.

Execution:

- device: `cuda:0`, NVIDIA L40, compute capability 8.9;
- PyTorch: `2.7.1+cu126`; CUDA was available;
- partition: HFL `grouped_round_robin`, 32 parties, seed 2026;
- partition checksum: `192268c512954beaed9b2016c947336bb90a0c3072d658d4705d9e02b102dcbc`;
- schedule: one round, client fraction 0.03125, one selected party;
- selected party: 27, with 33,938 positive training interactions;
- runtime provenance: tested commit recorded with `git_dirty: false` at launch.

Smoke-only metrics — **not thesis results**:

| Split | HR@10 | NDCG@10 | Targets |
|---|---:|---:|---:|
| Validation | 0.22036423841059602 | 0.11918452286363332 | 6,040 |
| Test | 0.21903973509933775 | 0.11767401789684927 | 6,040 |

## Artifact verification

The ignored run directory was
`tmp/phase3_smoke_results/ml1m_fedgmf_fedavg_smoke/seed_2026_20261012T032800629241Z`.

- `config.yaml` was semantically equal to the smoke configuration.
- `metrics.json` contained finite validation/test HR@10 and NDCG@10 in
  `[0, 1]`.
- `round_metrics.csv` contained exactly one row and one selected party;
  download and upload were 118,628 bytes each and total communication was
  237,256 bytes.
- `partition.json` recorded 32 grouped parties and the expected assignment
  checksum.
- `runtime.json` recorded `cuda:0`, NVIDIA L40, the tested clean Git commit,
  and the expected ML1M checksum.
- `run.log` contained configuration, dataset, partition, run, round, final
  evaluation, and artifact-directory lifecycle events.
- No checkpoint was written, as configured.

Phase 3 is validated end to end for the defined synthetic CPU regression and
the one-round real ML1M/GPU smoke scope. CUDA reproducibility remains
best-effort, as documented by the baseline protocol.
