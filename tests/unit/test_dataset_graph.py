import random

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.expand import ExpansionPool
from ac_zero.datasets.graph import ConstructionGraph, Edge, _trivial_root
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import MultiplyRelatorsMove, move_from_json


def _seed_graph(rank: int = 2) -> ConstructionGraph:
    root = _trivial_root(rank)
    return ConstructionGraph({root.content_hash: root})


def _records(pool: ExpansionPool, presentation: BalancedPresentation) -> list:
    (records,) = list(pool.expand([presentation]))
    return records


# --- content-hash cache -----------------------------------------------------


def test_content_hash_is_cached_and_pure() -> None:
    pres = BalancedPresentation.standard(3)
    first = pres.content_hash
    assert pres._content_hash == first  # populated after first read
    assert pres.content_hash is first  # same object returned, not recomputed

    # Caching must not change equality or the serialized hash.
    fresh = BalancedPresentation.standard(3)
    assert pres == fresh
    assert fresh.to_json()["content_hash"] == first


# --- expand_group -----------------------------------------------------------


def test_expand_group_returns_compact_length_changing_neighbours() -> None:
    with ExpansionPool(rank=2, total_length_cap=48, workers=1) as pool:
        records = _records(pool, BalancedPresentation.standard(2))
    assert records, "the trivial group has neighbours"
    for record in records:
        # Each record is a rebuildable, verified construction step.
        parent = BalancedPresentation.standard(2)
        child = move_from_json(record.move).apply(parent)
        assert child.content_hash == record.child_hash
        assert child.content_hash != parent.content_hash  # length-changing only
        assert tuple(r.letters for r in child.relators) == record.letters
        assert record.reverse_delta >= 1


def test_expand_group_drops_neighbours_over_the_length_cap() -> None:
    # With a cap of 1, every neighbour that grows a relator past length 1 is
    # filtered inside the worker so it never reaches the merge.
    pres = BalancedPresentation.standard(2)
    with ExpansionPool(rank=2, total_length_cap=1, workers=1) as pool:
        records = _records(pool, pres)
    for record in records:
        assert sum(len(word) for word in record.letters) <= 1


def test_pool_fans_out_without_changing_results() -> None:
    catalog_pres = BalancedPresentation.standard(2)
    with ExpansionPool(rank=2, total_length_cap=48, workers=1) as inline:
        inline_records = _records(inline, catalog_pres)
    # A spawned pool over a batch big enough to trip the lazy threshold must
    # return byte-identical records, in the same order, as the inline path.
    batch = [catalog_pres] * 600
    with ExpansionPool(rank=2, total_length_cap=48, workers=4) as pooled:
        pooled_records = list(pooled.expand(batch))
        assert pooled._executor is not None  # crossed _SPAWN_AFTER_GROUPS
    assert all(chunk == inline_records for chunk in pooled_records)


def test_pool_stays_inline_for_short_runs() -> None:
    with ExpansionPool(rank=2, total_length_cap=48, workers=4) as pool:
        list(pool.expand([BalancedPresentation.standard(2)] * 8))
        assert pool._executor is None  # too little work to justify spawning


# --- ConstructionGraph.merge (co-optimal recording) -------------------------


def test_merge_records_new_groups_with_their_construction_edge() -> None:
    graph = _seed_graph()
    root = next(iter(graph.nodes.values()))
    with ExpansionPool(rank=2, total_length_cap=48, workers=1) as pool:
        records = _records(pool, root.presentation)
    added = graph.merge(root, records)
    assert added == len(records)
    for record in records:
        node = graph.nodes[record.child_hash]
        assert node.difficulty == 1
        assert node.predecessors == [Edge(root.content_hash, move_from_json(record.move))]


def test_merge_appends_equal_depth_edges_and_ignores_longer_ones() -> None:
    graph = _seed_graph()
    root = next(iter(graph.nodes.values()))
    child_hash = "deadbeef"
    move = MultiplyRelatorsMove(0, 1).to_json()

    class _Rec:
        def __init__(self, move, child_hash, delta):
            self.move, self.child_hash, self.reverse_delta = move, child_hash, delta
            self.letters = tuple(r.letters for r in root.presentation.relators)

    # First edge at depth 1 creates the node.
    graph.merge(root, [_Rec(move, child_hash, 1)])
    node = graph.nodes[child_hash]
    assert len(node.predecessors) == 1

    # A different depth-1 edge to the same child is kept as a co-optimal move.
    other = MultiplyRelatorsMove(1, 0).to_json()
    graph.merge(root, [_Rec(other, child_hash, 1)])
    assert len(node.predecessors) == 2

    # A longer construction (a depth-1 parent -> child would arrive at depth 2)
    # is discarded: neither the edge set nor the difficulty changes.
    graph.merge(node, [_Rec(move, child_hash, 1)])
    assert node.difficulty == 1
    assert len(node.predecessors) == 2


def test_merge_supersedes_a_longer_construction_and_reopens_the_node() -> None:
    graph = _seed_graph()
    root = next(iter(graph.nodes.values()))
    child_hash = "cafe"
    letters = tuple(r.letters for r in root.presentation.relators)

    class _Rec:
        def __init__(self, move, delta):
            self.move, self.child_hash, self.reverse_delta = move, child_hash, delta
            self.letters = letters

    # Plant a suboptimal node at depth 5, marked exhausted.
    graph.merge(root, [_Rec(MultiplyRelatorsMove(0, 1).to_json(), 1)])
    node = graph.nodes[child_hash]
    node.difficulty = 5
    node.exhausted = True

    # A depth-1 construction from the root strictly improves it: whole edge set
    # replaced and the node re-opened so the improvement can propagate.
    graph.merge(root, [_Rec(MultiplyRelatorsMove(1, 0).to_json(), 1)])
    assert node.difficulty == 1
    assert node.exhausted is False
    assert node.predecessors == [Edge(root.content_hash, MultiplyRelatorsMove(1, 0))]


def test_select_batch_smallest_is_ordered_and_bounded() -> None:
    graph = _seed_graph()
    root = next(iter(graph.nodes.values()))
    with ExpansionPool(rank=2, total_length_cap=48, workers=1) as pool:
        graph.merge(root, _records(pool, root.presentation))
    rng = random.Random(0)
    batch = graph.select_batch("smallest", rng, size=3, short_bias=2.0)
    assert len(batch) == 3
    lengths = [node.total_length for node in batch]
    assert lengths == sorted(lengths)  # shortest groups first


def test_catalog_matches_expansion_move_count() -> None:
    # Every catalog move is applied once per expansion; guard the two stay in sync.
    catalog = ActionCatalog(2)
    with ExpansionPool(rank=2, total_length_cap=48, workers=1) as pool:
        records = _records(pool, BalancedPresentation.standard(2))
    assert len(records) <= len(catalog.moves)
