from src.models.graph import NodeKind, NotationGraph, Relation
from src.models.primitives import PrimitiveSplit


def graph_from_split(split: PrimitiveSplit) -> NotationGraph:
    """Convert a PrimitiveSplit into a NotationGraph."""
    graph = NotationGraph()
    graph.staff_space = split.staff_space
    graph.staff_lines = split.staff_lines
    node_ids = []

    # Add nodes for each primitive
    for primitive in split.primitives:
        node = graph.add_node(kind=primitive.kind, primitive=primitive)
        node_ids.append(node.id)

    # Add edges for head-to-stem relationships
    for head_idx, stem_idx in split.head_to_stem:
        graph.add_edge(
            src=node_ids[head_idx],
            dst=node_ids[stem_idx],
            relation=Relation.HEAD_TO_STEM,
        )

    # Add edges for stem-to-beam relationships
    for stem_idx, beam_idx in split.stem_to_beam:
        graph.add_edge(
            src=node_ids[stem_idx],
            dst=node_ids[beam_idx],
            relation=Relation.STEM_TO_BEAM,
        )

    return graph


def compute_staff_positions(graph: NotationGraph) -> None:
    if not graph.staff_lines:
        return

    ref_y = max(graph.staff_lines)
    half_space = graph.staff_space / 2.0

    for node in graph.nodes.values():
        x0, y0, x1, y1 = node.primitive.bbox
        center_y = (y0 + y1) / 2.0
        node.attrs["staff_pos"] = (ref_y - center_y) / half_space

    for x_order, node in enumerate(
        sorted(graph.nodes.values(), key=lambda n: n.primitive.bbox[0])
    ):
        node.attrs["x_order"] = x_order


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _x_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True if the two bounding boxes share any horizontal extent."""
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    return ax0 < bx1 and bx0 < ax1


def build_spatial_edges(graph: NotationGraph) -> None:
    """Add spatial relation edges between nodes (the BUILD phase)."""
    space = graph.staff_space
    if space <= 0:
        return

    align_tol = 0.5 * space
    above_gap = 0.5 * space
    nodes = [n for n in graph.nodes.values() if n.primitive is not None]

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            a_box, b_box = a.primitive.bbox, b.primitive.bbox
            (a_cx, a_cy), (b_cx, b_cy) = _bbox_center(a_box), _bbox_center(b_box)

            same_column = _x_overlaps(a_box, b_box)
            if same_column or abs(a_cx - b_cx) <= align_tol:
                graph.add_edge(a.id, b.id, Relation.ALIGNED_VERTICAL)

            # ABOVE only within the same column; smaller y = higher on the page.
            if same_column:
                if b_cy - a_cy > above_gap:
                    graph.add_edge(a.id, b.id, Relation.ABOVE)
                elif a_cy - b_cy > above_gap:
                    graph.add_edge(b.id, a.id, Relation.ABOVE)


def _absorb(graph: NotationGraph, consumed_ids, new_id: int) -> None:
    """Replace the consumed nodes with ``new_id``, re-pointing external edges.

    Edges between two consumed nodes (internal to the pattern) are dropped;
    edges from a consumed node to an outside node are re-created on ``new_id``,
    preserving relation and direction. Then the consumed nodes are removed.
    """
    consumed = set(consumed_ids)
    added: set[tuple[int, int, Relation]] = set()
    for edge in list(graph.edges):
        if edge.src not in consumed and edge.dst not in consumed:
            continue
        src = new_id if edge.src in consumed else edge.src
        dst = new_id if edge.dst in consumed else edge.dst
        if src == dst:
            continue  # internal edge -> would be a self-loop, drop it
        key = (src, dst, edge.relation)
        if key in added:
            continue
        added.add(key)
        graph.add_edge(src, dst, edge.relation)
    for node_id in consumed_ids:
        graph.remove_node(node_id)


def _heads_on_stem(graph: NotationGraph, stem) -> list:
    return [
        n
        for n in graph.neighbors(stem.id, Relation.HEAD_TO_STEM)
        if n.kind == NodeKind.NOTEHEAD
    ]


def rule_note(graph: NotationGraph) -> bool:
    """One stem carrying exactly one notehead -> note."""
    for stem in graph.nodes_of_kind(NodeKind.STEM):
        heads = _heads_on_stem(graph, stem)
        if len(heads) != 1:
            continue
        head = heads[0]

        note = graph.add_node(NodeKind.NOTE)
        for key in ("staff_pos", "x_order"):
            if key in head.attrs:
                note.attrs[key] = head.attrs[key]
        # keep the consumed primitives for the semantic passes (duration, pitch)
        note.attrs["notehead"] = head.primitive
        note.attrs["stem"] = stem.primitive

        _absorb(graph, (head.id, stem.id), note.id)
        return True
    return False


def rule_chord(graph: NotationGraph) -> bool:
    """One stem carrying two or more noteheads -> chord."""
    for stem in graph.nodes_of_kind(NodeKind.STEM):
        heads = _heads_on_stem(graph, stem)
        if len(heads) < 2:
            continue
        heads = sorted(heads, key=lambda h: h.attrs.get("staff_pos", 0.0))

        chord = graph.add_node(NodeKind.CHORD)
        chord.attrs["staff_positions"] = [h.attrs.get("staff_pos") for h in heads]
        chord.attrs["noteheads"] = [h.primitive for h in heads]
        chord.attrs["stem"] = stem.primitive
        x_orders = [h.attrs["x_order"] for h in heads if "x_order" in h.attrs]
        if x_orders:
            chord.attrs["x_order"] = min(x_orders)

        _absorb(graph, tuple(h.id for h in heads) + (stem.id,), chord.id)
        return True
    return False


def rule_beamed_group(graph: NotationGraph) -> bool:
    """Two or more notes/chords sharing one beam -> beamed_group."""
    for beam in graph.nodes_of_kind(NodeKind.BEAM):
        members = [
            n
            for n in graph.neighbors(beam.id, Relation.STEM_TO_BEAM)
            if n.kind in (NodeKind.NOTE, NodeKind.CHORD)
        ]
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda n: n.attrs.get("x_order", 0))

        group = graph.add_node(NodeKind.BEAMED_GROUP)
        # keep the member nodes; they carry pitch/duration for encoding
        group.attrs["members"] = members
        group.attrs["beam"] = beam.primitive
        if "x_order" in members[0].attrs:
            group.attrs["x_order"] = members[0].attrs["x_order"]

        _absorb(graph, tuple(m.id for m in members) + (beam.id,), group.id)
        return True
    return False


REWRITE_RULES = (rule_chord, rule_note, rule_beamed_group)


def run_rewrite(graph: NotationGraph) -> None:
    """Apply rewrite rules until none fires (the WEED + INCORPORATE loop)."""
    max_iterations = max(1, len(graph.nodes) * 10)
    for _ in range(max_iterations):
        if not any(rule(graph) for rule in REWRITE_RULES):
            return
