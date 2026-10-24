# MovieLens-1M public item–genre graph

This unit converts a complete, already built `FeatureView` into a public
canonical graph, then assigns six public genre fields to each explicitly
supplied party by `party_id % 3`. Parties and users are different domains.
The three functions are imported directly from their submodules; package
exports, the metadata parser, runtime and method adapters are unchanged.

```python
from src.data.ml1m_graph import (
    ml1m_item_genre_feature_binding,
    build_ml1m_item_genre_graph,
)
from src.data.graph_scenarios import build_public_genre_block_partition
```

The APIs are:

```python
ml1m_item_genre_feature_binding(bundle: DatasetBundle, *, genre_view: FeatureView) -> str
build_ml1m_item_genre_graph(
    bundle: DatasetBundle, *, genre_view: FeatureView,
    expected_feature_binding: str,
) -> GraphView
build_public_genre_block_partition(
    bundle: DatasetBundle, parties: tuple[PartyData, ...], *, graph_name: str,
    expected_graph_fingerprint: str, public_party_ids: tuple[int, ...],
) -> tuple[GraphPartition, tuple[PartyData, ...]]
```

## Complete source and narrow binding

The bundle and feature view must be exact records. The item map is a plain
`dict` of positive built-in integer MovieIDs to built-in integer processed
IDs, bijective onto `0..I-1`. Boolean, NumPy and subclass IDs are rejected.
User-domain validation follows the existing public `graph_domain_fingerprint`.
Interactions, splits, candidates and evaluation results do not construct the
graph or decide grants. Inputs must remain stable during each call; these
functions do not provide concurrent snapshots or revocation.

The feature schema is `item_genres` / `item` / `tabular`. `entity_ids` are
native metadata-free plain `int64[I]` in processed order; `values` are plain
native metadata-free `float32[I,18]`, finite and exactly binary, with at least
one positive cell per nonempty item row. The mask is `None` or plain native
metadata-free `bool[I,18]` containing only false. Strided and read-only plain
arrays are accepted. Array subclasses, masked arrays and alternative dtypes
are rejected before content operations. The exact tuple of built-in strings
is fixed in lexical order:

```
Action, Adventure, Animation, Children's, Comedy, Crime, Documentary,
Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi,
Thriller, War, Western
```

Only these 20 source metadata keys are read; other keys and values are not
traversed, hashed, copied or retained:

| Key | Required assertion |
| --- | --- |
| builder | `ml1m_item_genres` |
| builder_version | exact built-in integer `1` |
| source_relative_path | nonempty exact string, no outer whitespace or Unicode Cc; identity text only |
| source_encoding | `latin-1` |
| source_format | `MovieID::Title::Genres` |
| source_size_bytes | exact integer `1..2**63-1` |
| source_sha256 | exact lowercase 64-character SHA-256 string |
| vocabulary_source | `MovieLens-1M README: MOVIES FILE DESCRIPTION` |
| vocabulary_policy | `fixed` |
| genre_vocabulary | exact list of the 18 exact strings above |
| encoding | `multi_hot` |
| row_order | `processed_item_id` |
| column_order | `lexical_genre_name` |
| raw_movie_id_policy | `positive_ascii_decimal` |
| genre_token_policy | `exact_known_unique_nonempty` |
| missing_metadata_policy | `error` |
| duplicate_movie_id_policy | `error` |
| metadata_only_policy | `validate_and_exclude` |
| item_mapping_sha256 | current mapping digest described below |
| counts | plain dictionary with exactly the eight keys below |

Fixed text values must be exact built-in strings. Every count is an exact
built-in integer in `0..2**63-1`. `source_records` and
`unique_metadata_items` both equal `I + metadata_only_items` and are positive;
`processed_items = matched_items = I`, `missing_items = 0`, `genre_count = 18`,
and `positive_entries` equals the number of positive feature cells. These
source fields are checked assertions, not independent authentication of a raw
file. No file named by `source_relative_path` is opened. Metadata-only records
do not add catalogue nodes.

The mapping digest hashes UTF-8 JSONL, ordered by processed ID, using exactly
`json.dumps({'raw_id': raw, 'processed_id': processed}, ensure_ascii=False,
sort_keys=True) + '\n'`, including the default JSON spaces. The binding hashes
UTF-8 canonical JSON with `sort_keys=True`, `separators=(',', ':')`,
`ensure_ascii=True`, `allow_nan=False`, and no trailing newline:

```text
{
  format: 'thesisfedrec.ml1m_item_genre_feature_binding', version: 1,
  domain_fingerprint: graph_domain_fingerprint(bundle),
  item_mapping_sha256: <recomputed mapping digest>,
  view: {
    name, entity_kind, modality, feature_names: <list>,
    entity_ids: {shape: <list>, dtype: 'int64', sha256: <array digest>},
    values: {shape: <list>, dtype: 'float32', sha256: <array digest>},
    missing_mask: null or {shape: <list>, dtype: 'bool', sha256: <array digest>}
  },
  source_metadata: <new built-in values for exactly the 20 keys>
}
```

Array digests cover complete logical C-order bytes, with little-endian int64
and float32 encodings and C-order bool bytes. `None` and an explicit false
mask bind differently but generate identical edges. Validation precedes the
binding digest. The graph builder compares the recomputed digest against the
mandatory exact lowercase SHA-256 expectation before allocating graph output.
Computing an expectation from the same newly received view establishes local
consistency only. A future pilot needs a trusted public layer to freeze the
expectation and authenticate sources, maps and arrays separately.

## Canonical graph and public field grants

The returned, unattached graph is `item_genres_graph`. Raw node IDs are the
processed-order MovieIDs followed by the 18 genre strings. Node types are
`('item', 'genre')`, with item types zero and genre types one. Each positive
cell `(i,j)` creates `(i,I+j)` in increasing `(i,j)` order. All relations are
zero with names `('has_genre',)`. There are no inverse, behavioral, weighted,
self or user edges. `user_to_node` is `None`; `item_to_node` is the complete
identity map. Numeric arrays are new owning C-order native int64 arrays,
including empty arrays. Node and edge counts are checked with Python integers
against int64 capacity before output allocation.

Graph metadata contains only builder `ml1m_item_genre_graph`, builder_version
`1`, schema_revision `ml1m.item-genre.v1`, source_feature_binding,
source_domain_fingerprint, source_item_mapping_sha256, source_sha256 and
counts `{item_count, genre_count, edge_count}`. Source paths and arbitrary
source metadata are excluded. Output arrays are mutable but isolated from
source arrays and other builds. Later consumers must recheck the current
graph fingerprint; metadata is excluded from the public graph fingerprint
and does not authenticate provenance.

A trusted caller attaches the graph after checking name collisions. The
partition function requires the complete canonical current graph and its
expected public fingerprint, even for an empty roster. It checks exact
container/array types, domain, raw row order, all anchors, types, relation,
row coverage and edge order. It accepts only graph name `item_genres_graph`.

`public_party_ids` is an exact sorted unique tuple of built-in integers in
`0..2**63-1`. `parties` is an exact tuple of exact `PartyData` records with
unique IDs whose set equals this packet. This caller assertion does not
prove population completeness. Input party order is preserved. Existing graph
summaries must be empty or name this graph alone. New summaries use
`dataclasses.replace`, retain the same sample array objects and feature/label
fields, and copy only the outer metadata dictionary. They do not change
`bundle.parties` or expand its internal contiguous-ID convention.

For group `g = party_id % 3`, select genre columns `j % 3 == g` across the
entire canonical catalogue. Explicit sorted int64 node selections contain
all six granted genre nodes, including unused genres, plus incident item
nodes. Explicit sorted edge selections contain precisely those positive
edges. Empty edge selections are arrays, never `None`. Immutable `GraphAccess`
descriptors may be shared within a group.

The partition is `ml1m_public_genre_blocks3_catalogue_domain.r1`, bound to the
public domain and current graph. A nonempty roster binds exactly that graph;
an empty roster binds no source graphs. Metadata is only the public packet
`recipe_id='ml1m.public_genre_blocks3.v1'`, `group_count=3`,
`assignment='party_id_mod3'`, sorted full_public_party_ids and sorted
`party_groups=[{party_id, group_id}]`. Public `validate_graph_partition`
validates the result. Sample counts, splits, eligibility, coverage, graph
validity, loss and response norms never choose a group or replacement party.

## Fixed synthetic integration oracle

The four items map `{10:0, 20:1, 30:2, 40:3}`. Positive cells are `(0,0)`,
`(0,1)`, `(1,0)`, `(1,2)`, `(2,4)`, `(2,5)`, `(3,17)`, giving edges
`(0,4),(0,5),(1,4),(1,6),(2,8),(2,9),(3,21)` with IDs 0 through 6.
The full public IDs are `(7,41,90)` and input party order is `(41,7,90)`.
Users are a separate domain; fixture interactions cover every mapped user
and item from construction.

| Party/group | Edge IDs | Node IDs | Local item map | Local edge rows |
| --- | --- | --- | --- | --- |
| 90/0 | 0,2 | 0,1,4,7,10,13,16,19 | 0,1,-1,-1 | `[0,1]`, `[2,2]` |
| 7/1 | 1,4 | 0,2,5,8,11,14,17,20 | 0,-1,1,-1 | `[0,1]`, `[2,3]` |
| 41/2 | 3,5,6 | 1,2,3,6,9,12,15,18,21 | -1,0,1,2 | `[0,1,2]`, `[3,4,8]` |

Group 0 has a degree-two auxiliary node; the other two groups have singleton
auxiliary nodes in this fixed fixture. This does not alter the assignment.
The existing `prepare_graph_runtime` / `project_party_graphs` /
`close_graph_runtime` APIs produce local owning arrays, anonymous `n0`… raw
IDs and empty metadata. Public vocabulary remains visible. Full item maps
still allocate O(I) per projection; this is not an O(local edges) operation.
No unselected item becomes a phantom local node. Runtime bindings must be
renewed after graph, domain or access changes. Closing a runtime rejects
subsequent projections; already returned owning payloads remain usable.

`I=0` is a legal data graph with 18 auxiliary nodes and zero edges, and each
party retains six local auxiliary nodes and an empty item map. Its source
metadata still describes at least one validated metadata-only record. This
does not imply trainability: the existing method adapter rejects zero items.

## Validation and limits

New tests independently construct the canonical JSON and byte-hash oracle,
assert the hand-computed graph and projections, and cover strict types,
metadata whitelist, invalid schema/content/bindings, capacity boundaries,
mutation isolation and independence from legal sample/split changes. The
focused invocation also runs the existing graph runtime and graph-semantic
adapter tests as compatibility regressions. Those baseline tests include
numerical and gradient checks; the new production modules import neither
Torch nor research methods. Synthetic source digests are test literals, not
claims about real MovieLens files.

Wrong records, containers, elements and dtypes raise `TypeError`; wrong
shapes, domains, contents, bindings, duplicate IDs and unsupported schemas
raise `ValueError`. Reused public functions retain their existing exception
contracts. No real source, cache, training, evaluation, pilot, roster-learning,
basis, U4, DP or privacy mechanism is introduced. Public-field availability
is an engineered treatment of existing public metadata; it establishes no
natural genre privacy, new external KG, graph utility, H1 result,
recommendation benefit, performance scale claim or frozen paper profile.
