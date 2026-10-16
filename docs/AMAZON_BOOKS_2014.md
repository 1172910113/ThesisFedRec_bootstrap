# Amazon Books 2014 official 5-core: canonical data unit

## Source identity, not benchmark equivalence

The selected source is the Books **official 5-core** from
[the Amazon Reviews 2014 source page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
The links resolved from that page are:

- [reviews_Books_5.json.gz](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz)
  — reference count 8,898,041 reviews;
- [meta_Books.json.gz](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz)
  — reference count 2,370,585 products in the full Books metadata source.

These are source checks, not processed-size assumptions. The audit reports
record and unique-item counts separately, including duplicate metadata IDs.
A count mismatch fails acceptance and must be investigated, not repaired by
subsampling or switching versions. This unit does not establish equivalence
with the LightGCN or KGAT Amazon-book benchmark. Kindle and Amazon 2018/2023
are not substitutes; Industrial and Scientific remains deferred.

## Storage and acquisition

Activate the existing environment and set the project root for each process:

```bash
source /home/yuanhaochen/miniconda3/etc/profile.d/conda.sh
conda activate thesis-fedrec
export THESISFEDREC_DATA_ROOT=/data_nfs/yuanhaochen/ThesisFedRec
```

Use only needed directories beneath that root:

```text
raw/amazon_books_2014/reviews_Books_5.json.gz
raw/amazon_books_2014/meta_Books.json.gz
manifests/                         # source URL, variant, size, local SHA-256
audits/                            # count, degree, coverage, integrity reports
tmp/                               # unique acquisition/validation directories
```

`processed/amazon_books_2014_5core/`, `cache/`, `features/`, `graphs/`, and
`results/` are reserved for separately approved work, not created by the data
functions. There is no processed-cache writer in this unit. Existing schema
`1.0` behavior and nonempty FeatureView/GraphView rejection remain unchanged.
Do not place real payloads in Git or symlink them outside this project root.
Do not move/delete other projects' sources or historical results.

Before acquisition, locate existing archives, verify their identities, and
check storage capacity. Download only the official URLs into a unique `tmp/`
directory with partial names ending in `.gz`; use bounded connection/transfer
timeouts and retries and report transfer progress. Preserve incomplete files
as partials and valid existing archives unchanged. Validate complete gzip
streams, required records, counts, and locally computed SHA-256 before
finalizing to `raw/amazon_books_2014/` without overwriting an existing source.
The code/test/audit command never downloads data. A local SHA-256 is file
identity, not a checksum independently published by the source provider.

No pretrained model is needed. Future model artifacts/caches must use
`/data_nfs/yuanhaochen/llm` with exact revisions and explicit local loading;
this unit neither creates nor downloads them.

## Implemented interfaces

Exported by `src.data`:

```python
iter_amazon_books_reviews(reviews_relative_path=REVIEWS_PATH, *, data_root=None)
# iterator of (reviewerID: str, asin: str, overall: float, unixReviewTime: int)

preprocess_amazon_books(reviews_relative_path=REVIEWS_PATH, *, data_root=None, seed=2026)
# DatasetBundle(name="amazon_books_2014_5core")

audit_amazon_books_metadata(bundle, metadata_relative_path=METADATA_PATH, *, data_root=None)
# plain coverage/source-identity dictionary; does not alter the bundle
```

Paths are relative to the explicit root, otherwise `THESISFEDREC_DATA_ROOT`,
otherwise repository `data/raw` for small fixtures. Absolute paths and root
escapes fail through existing path resolution. The data-only YAML at
`configs/datasets/amazon_books_2014_5core.yaml` documents these defaults; it is
not an executable Phase 3 experiment config and is not consumed by the frozen
Phase 3 validator/runner. The standalone audit CLI takes explicit path options.

## Parsing and frozen processing rules

Compressed sources are read incrementally as UTF-8, one record at a time.
JSON is preferred. Original Python dictionary literals use `ast.literal_eval`,
never `eval`; duplicate dictionary keys, non-dictionary records, and malformed
records fail with line context. No list of all review/metadata dictionaries
is materialized. Plain-text equivalents are supported for small fixtures.

Each review requires nonblank string `reviewerID` and `asin`, numeric finite
`overall` in `[1, 5]` excluding booleans, and a nonnegative int64-range integer
`unixReviewTime` excluding booleans. Strings are preserved exactly, including
leading zeros, with no integer conversion, case folding, trimming, ASIN-length
restriction, or parent-ASIN interpretation. Invalid required records fail;
they are not silently filtered. Review text and other review-derived attributes
are discarded, never supplied as training features. Source timestamps are Unix
seconds, with no local-time conversion.

The fixed pipeline is:

1. Parse the entire official input; collect compact integer columns and unique
   string tables. Temporary interning codes are internal storage indices, not
   a second exported recommendation namespace.
2. Apply only the existing minimum-three-reviews-per-user split requirement.
3. For each exact `(reviewerID, asin)`, keep the greatest timestamp; equal
   timestamps keep the later source line. Reuse the existing numeric-column
   duplicate helper without changing ML1M behavior.
4. Recheck the minimum three **retained distinct interactions** per user.
   Report both filters and deduplication. Do not iteratively prune items/users
   to restore 5-core or add 10-core, time-window, or rating-threshold filters.
5. Restore retained source-line order. Assign independent contiguous user/item
   IDs by first appearance, keeping exact string keys. Metadata-only ASINs never
   create recommendation items. Interaction IDs follow retained source order.
6. Set every observed interaction's value to `float32(1.0)`, including low
   ratings. Reuse `leave_one_out_split`: per user, sort by `(timestamp,
   interaction_id)`; latest is test, second latest validation, rest training.

The seed is recorded and validated but does not randomize parsing, mapping,
deduplication, or splitting. No global RNG state changes. Output has empty
feature/graph/continual collections and no selected party partition. Existing
HFL partition functions work without equating users with parties.

The degree audit reports raw review counts, counts after deduplication of
initially eligible users, and final unique-pair degrees. For users and items it
reports surviving/removed entities, min/max degrees, and counts below 3 and 5.
An upstream 5-core label is **not** a claim that post-transformation degrees
remain 5-core. No such pruning is performed. Final counts include all split
sizes, and every retained user has exactly one validation and one test entry.

## Metadata audit, not feature availability or leakage approval

Metadata requires an exact string `asin` and is joined only to the current
processed item map. The audit reports source records, unique metadata IDs,
duplicate IDs/extra records, duplicate processed-item IDs, matched items,
missing items/examples, and excluded unique metadata-only items. Missing
metadata or duplicates are reported rather than dropping interactions or
selecting an arbitrary feature-record winner. Malformed metadata fails.

For each observed field, and each `related.<relation>` entry, report source
record presence/nonempty counts and unique matched-item presence/nonempty
counts. Matched counts are a union across duplicate records **for auditing
only**. None, blank strings, and empty collections count as empty; numeric zero
and False are not implicitly missing. Counts do not assert semantic usability,
complete features, valid relations, or permission to share private information.

No encoders, features, graph edges, or metadata-driven pruning are produced.
Relations, sales ranks, prices, and other snapshot-dependent fields require a
later source/time-policy review before model use. This unit grants no use of
held-out reviews or their derived statistics as training features.

## CPU acceptance and reports

With archives already available, run:

```bash
env -u THESISFEDREC_RUN_REAL_FEATURES -u THESISFEDREC_RUN_REAL_SMOKE \
  THESISFEDREC_RUN_REAL_BOOKS=1 python -m pytest -q -s \
  tests/test_amazon_books.py::test_real_amazon_books_data_acceptance

python scripts/audit_amazon_books.py \
  --audit-output audits/books_2014_acceptance.json \
  --source-manifest manifests/books_2014_sources.json
```

The separate Books opt-in skips when unset. Once enabled, unavailable, corrupt,
or invalid inputs fail rather than skip or download. It checks source reference
counts, raw degree lower bounds, canonical namespace order, unique retained
pairs, temporal/disjoint/complete splits, empty information views, and SHA-256
integrity after processing; metadata coverage is reported even when incomplete.
The CLI prints JSON and returns 0 only when acceptance checks pass, otherwise
1 for reference/invariant mismatches. Optional JSON reports use exclusive
creation under `audits/` and `manifests/`; existing reports are never overwritten.
They contain source/processing identities and mapping/array checksums, not
durable processed datasets. SHA-256 values for arrays hash their contiguous
native array bytes; ID columns/splits are int64 and values float32. Mapping
hashes use processed-ID-ordered UTF-8 JSONL with exact raw strings.

Run synthetic and complete regression suites with all three real-data flags
unset. The existing ML1M feature and GPU-smoke opt-ins remain separate. No GPU
work, experiment sweeps, training, or recommendation-quality evidence belongs
to this unit. Report actual environment versions: the existing environment is
Python 3.11, despite the historical Python 3.10 convention; 3.10 compatibility
is not claimed without testing. No dependencies or environment changes are needed.

## Executed validation — 2026-10-16

Both official archives were acquired into task-specific partial files, fully
parsed through gzip EOF, checked against the reference counts, and finalized
without overwriting existing sources. Final files are under the project root's
`raw/amazon_books_2014/` directory:

| Archive | Bytes | Records | Locally computed SHA-256 |
| --- | ---: | ---: | --- |
| `reviews_Books_5.json.gz` | 3,223,678,899 | 8,898,041 | `d22b38a7066e2ac3b25ffb66bc1264d8974ebe8f9dc51ed2e4b0f6ed43358a65` |
| `meta_Books.json.gz` | 825,767,823 | 2,370,585 | `1738066f8d61caa3f5e0bfd9427eba7519187102ca8e8d2aa7462a894d01981f` |

Observed raw and retained counts were identical: **603,668 users, 367,982
items, 8,898,041 interactions**. No duplicate pairs or below-minimum users were
removed. User/item minimum degrees were both 5 before and after processing;
maximum degrees were 23,222 and 7,440 respectively. These are observed full
dataset degrees, not a minimum-degree claim about the training split.
Train/validation/test sizes were **7,690,705 / 603,668 / 603,668**.

Metadata contained 2,370,585 unique ASINs and no duplicate IDs. All 367,982
processed items matched; none lacked metadata. The 2,002,603 metadata-only
items were excluded from the recommendation namespace. Nonempty matched-item
coverage included title 286,303, description 237,904, categories 367,982,
image URL 367,128, price 352,223, brand 18, and related-item fields 343,733.
Complete record coverage does not imply complete individual fields or approval
to use snapshot-dependent attributes/relations in a model.

Executed checks (all exit 0):

- focused synthetic/shared-data regressions: 71 passed, 1 skipped;
- full regression with all real-data opt-ins unset: 712 passed, 3 skipped;
- separate Books CPU opt-in: 1 passed in 383.98s;
- `git diff --check`: passed.

The default skips were Books acceptance, ML1M real features, and the existing
ML1M/GPU smoke. Only the Books opt-in was subsequently run. No training or GPU
smoke was executed. Environment: Python 3.11.16, NumPy 1.26.4, PyTorch
2.7.1+cu126 (CUDA runtime 12.6), PyYAML 6.0.3, pytest 8.4.2; unchanged.

The real test's JSON output and log were retained, without a second preprocessing
pass, in these external project-root paths:

- `audits/books_2014_acceptance_2026-10-16.json`;
- `audits/books_2014_acceptance_2026-10-16.log`;
- `manifests/books_2014_sources_2026-10-16.json`.

The audit includes source, implementation, mapping, interaction-array, and
split-array SHA-256 values. All eleven acceptance checks passed, including
unchanged source hashes after processing. This is data-contract acceptance,
not benchmark equivalence, feature readiness, or recommendation-quality evidence.
