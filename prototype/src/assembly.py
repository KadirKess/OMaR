from src.graph import NotationGraph, Relation
from src.primitives import PrimitiveSplit


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
