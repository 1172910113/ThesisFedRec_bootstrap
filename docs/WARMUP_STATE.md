# FedGMF terminal warmup handoff

The direct submodule src.federation.warmup_state captures terminal values
already returned by FIT-1, persists a strict local artifact and materializes
independent method-ready CPU values. It does not run another fit, select an
earlier validation checkpoint, initialize, resume, restore an optimizer/RNG,
score, sample or evaluate.

## Public interface

~~~python
capture_fedgmf_warmup(
    fit_result, *, data_binding, partition_binding, profile_binding,
    code_revision, expected_num_items, expected_num_users,
    expected_party_user_ids,
) -> FedGMFWarmupSnapshot
materialize_fedgmf_warmup(snapshot)
# -> (E, w, b, dict[int, FedGMFLocalState])
save_fedgmf_warmup(snapshot, path) -> None
load_fedgmf_warmup(
    path, *, expected_context_id, expected_snapshot_id, max_file_bytes,
) -> FedGMFWarmupSnapshot
~~~

All shown arguments are required. Paths accept str or pathlib.Path, following
the existing local-field-map policy. Callers choose an authorized destination;
there is no default repository artifact root.

FedGMFWarmupSnapshot is factory-only, frozen, slotted and keyword-only.
Ordinary construction fails. Its exact fields are protocol_version=1,
context_id, snapshot_id, data_binding, partition_binding, profile_binding,
code_revision, num_items, num_users, embedding_dim, dtype, party_user_ids,
terminal_round_count, round_summaries and buffers. Retained values are only
built-in immutable scalars and recursive tuples/bytes:

- ownership: tuple[(party_id, tuple[user_id,...]), ...];
- rounds: tuple[(round_index, tuple[selected_party_id,...], positive_count), ...];
- buffers: tuple[(name, dtype, shape_tuple, nbytes, sha256, raw_bytes), ...].

No result, Tensor/ndarray/header, mapping, party, candidate, metric, RNG,
closure or open file is retained. Frozen/factory-only Python records are not
hostile-reflection security boundaries. Save and materialize validate the
entire current record, buffer hashes/user-row semantics and both identities.

## Capture boundary

Input must be exact FedGMFHFLFitResult; full-run results, subclasses and duck
types fail. Only final_shared_state, final_local_states and three structural
round-summary fields are consumed. Validation candidates and validation_metrics
are never traversed; whole-record asdict is not used. The input record's type
does not authenticate training history, convergence or user update coverage.

Data/partition/profile bindings are exact built-in lowercase64-hex strings,
code_revision exact lowercase40-hex. These are trusted caller assertions,
not independent raw-data, Git, profile or external-log verification. Expected
item/user counts and derived embedding dimension are positive built-in integers
at most 2**63-1; bool, NumPy integers and subclasses fail.

Ownership is a nonempty exact tuple of exact two-tuples, with strictly increasing
built-in nonnegative party IDs at most 2**63-1. Each nonempty user tuple strictly
increases within [0,U). Actual declared element count is compared with U before
union checks; users are disjoint and cover exactly 0..U-1 without constructing
a giant range/array for a small declaration. Party IDs are a separate domain.
This full-population artifact cannot represent a small cohort as complete warmup.

Shared state is an exact dict with exactly the built-in string keys
item_embeddings, prediction_weight and prediction_bias. Tensors are CPU,
nonnested, torch.strided, finite, all float32 or all float64. Parameter, nonleaf,
strided logical values are supported. Original shapes must be [I,d], [1,d], [1]
before conversion to method E[I,d], w[d], b[]. Reshape does not sanitize wrong
legacy shapes. No device/dtype conversion is offered.

Local state is an exact dict with exactly the declared built-in party keys.
Each exact FedGMFLocalState is revalidated: CPU strided int64 one-dimensional
user_ids match ownership row for row; CPU strided floating embeddings have
shape[U_p,d], shared dtype and finite values. Rows are not inferred from party
IDs or rearranged. Source fields may share storage; snapshot copies logical
values without propagating that alias relationship.

Summaries are a nonempty exact tuple of exact FedGMFHFLRoundSummary, indices
0..R-1, nonempty sorted selected-ID tuples contained in the roster, and positive
built-in counts at most 2**63-1. Only these returned facts persist. No authenticated
zero-test, minibatch/Adam counts, all-party participation or per-user update claim
is made. FIT's existing held-out ownership and all-split negative exclusions
are legal; absent test evaluation does not mean held-out-ID isolation.

Caller keeps sources stable during calls. Snapshot APIs do not change source
values/grad/storage/version or Python/NumPy/Torch RNG. No concurrent transaction
snapshot or revocation mechanism is provided.

## Buffers and canonical identities

All buffers use C-order little-endian float32/float64 or signed int64 logical
bytes, with no stride/header/padding. Negative-zero bits are preserved.
Exactly 3+2P buffers occur in this order:

| Name | Dtype | Shape |
| --- | --- | --- |
| item_embeddings | D | [I,d] |
| score_weight | D | [d] |
| score_bias | D | [] (one element) |
| party.p.user_ids, parties sorted | int64 | [U_p] |
| party.p.user_embeddings, after that party's IDs | D | [U_p,d] |

Shapes are exact tuples of built-in integers; actual dimensions are positive
except scalar's empty shape. Python integer products for numel and nbytes must
fit 2**63-1 before copying/decoding/allocating output. Declared nbytes, exactbytes
length and raw SHA256 must match. User-buffer values still match ownership.

CJSON is UTF-8 json.dumps(obj,sort_keys=True,separators=(',',':'),
ensure_ascii=False,allow_nan=False), without trailing LF. Context has exactly:

~~~text
format='thesisfedrec.fedgmf_warmup.context', version=1,
data_binding, partition_binding, profile_binding, code_revision,
num_items, num_users, embedding_dim, dtype,
party_user_ids=[{party_id,user_ids:[...]}, ...]
~~~

context_id=SHA256(CJSON(context)). A descriptor has exactly
name,dtype,shape:[...],nbytes,sha256. Snapshot has exactly:

~~~text
format='thesisfedrec.fedgmf_warmup.snapshot', version=1, context_id,
terminal_round_count,
round_summaries=[{round_index,selected_party_ids:[...],
                  total_positive_interactions}, ...],
buffers=[descriptor, ...]
~~~

snapshot_id=SHA256(CJSON(snapshot)). It binds terminal bytes and returned
round facts, not metrics/candidates. Equal metrics cannot reconstruct a snapshot.
These are local consistency identities, not signatures, source proof or
nonforgeable evidence of training execution.

## Strict file I/O and materialization

File object has exactly format='thesisfedrec.fedgmf_warmup.file',version=1,
context_id,snapshot_id,context,snapshot,tensors=[{name,data},...].
Each data is canonical standard-base64 ASCII for that buffer. Physical file
equals CJSON(file)+one LF. UTF-8 BOM, CRLF, extra whitespace/lines, duplicate
keys anywhere, extra/missing keys, unknown version/schema/name/dtype, nonfinite
constants and wrong integer types fail. Corrupt files are not normalized.
No pickle/torch/npz format or fallback is supported.

Load validates path, expected IDs and positive exact max_file_bytes before stat.
It rejects zero/over-cap/impossible files, reads bounded chunks, and checks
observed stat/read length changes and actual size against cap. Huge legal caps
including 2**63-1 do not become an overflowing Python read-size argument.
Same-size concurrent replacements are not comprehensively detected; complete
canonical bytes, trusted expected identities and hashes must still match.

All context/descriptor shapes, capacities, expectations, encoded lengths and
file-byte budgets are checked before base64 decode/output. Each nbytes and
their sum must fit already-read file bytes. Data must have exact ASCII length
4*ceil(nbytes/3); strict decode and identical reencoding reject whitespace,
URL-safe characters, noncanonical padding/padbits. Length/hash, finite floats
and semantic user rows are validated before returning. max_file_bytes bounds
this call's physical file I/O, not RSS or global disk usage.

Save validates first, uses same-parent mkstemp, complete write, flush/fsync,
then exclusive os.link(temp,target). Finally only its own temporary name is
unlinked. Parent directories are not created. Existing files/directories/
symlinks (including dangling) and racing targets are not overwritten.
OSError/FileExistsError/MemoryError propagate; published targets are not
removed to simulate rollback. No parent-directory fsync, power-loss durability
or general recovery transaction is promised.

Materialize validates first and returns independent contiguous CPU no-grad
E/w/b and a new sorted exact dict of new FedGMFLocalState objects. Outputs do
not alias source/snapshot bytes, other buffers or prior deliveries; output
value/header mutation does not affect another delivery. No Parameter/optimizer
is created and Python/NumPy/Torch RNG is not advanced.

## Fixed validation and cost

Literal independent oracle uses I3/U3/d2, ownership7→(0,2)/41→(1), rounds
(0,(7,41),5),(1,(41),2), E[[1,-0.0],[2,3],[-4,0.5]],
legacyw[[0.25,-0.5]], bias[-0.0], p7[[0.5,-1],[2,0.25]],
p41[[-0.75,4]]. Independent stdlib struct/CJSON/SHA encodes float32/64,
all raw bytes/identities/files compared exactly, without numeric tolerance.
Bindings are synthetic trusted literals, not real source digests.

Exactly one new one-round low-dimensional synthetic FIT→capture→save/load→
materialize chain guards test candidate/scoring/evaluation zeros; validation
is permitted. The authorized old test_fedgmf_fit legacyfull-wrapper regression
does score synthetic test candidates, outside that new-chain zero claim.
Runner/model/sampling/evaluation/configuration/exports are unchanged.

All four APIs cost at least O(weights+ownership+rounds). Base64 adds roughly
4/3 rawbytes plus JSON and complete ownership. File bytes, JSON strings,
decoded buffers, finite-check arrays, immutable bytes and materialized outputs
may coexist. No O(batch), production throughput, precise RSS, Python3.10/GPU,
recommendation benefit/privacy or full1.1/1.2/G3/U4/M2 qualification follows.
Warmup duration/seed/d/LR, research population/cohort/self/roster, optimizer
persistence and real pilot remain separately scoped.

Wrong records/containers/members/elements/dtypes raise TypeError; wrong
shape/domain/capacity/content/identity/canonical/schema/unsupported values
raise ValueError. File I/O and unrelated MemoryError propagate unchanged.
Failure does not modify input, prior output or old target.
