import unittest

import numpy as np

from src.assembly import graph_from_split
from src.graph import Relation
from src.primitives import Primitive, PrimitiveKind, PrimitiveSplit, SplitStatus


def make_primitive(kind, x0, y0, x1, y1, confidence=1.0):
    """Build a primitive whose mask covers the given bounding box."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return Primitive(kind=kind, mask=mask, confidence=confidence)


class GraphFromSplitTests(unittest.TestCase):
    def test_single_note_creates_two_nodes_and_one_edge(self):
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 30, 24, 42)
        stem = make_primitive(PrimitiveKind.STEM, 22, 10, 24, 40)
        split = PrimitiveSplit(
            primitives=(head, stem),
            head_to_stem=((0, 1),),
            status=SplitStatus.ACCEPTED,
        )

        graph = graph_from_split(split)

        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        edge = graph.edges[0]
        self.assertEqual(edge.relation, Relation.HEAD_TO_STEM)
        # edge must connect the notehead node to the stem node
        self.assertEqual(graph.nodes[edge.src].kind, PrimitiveKind.NOTEHEAD)
        self.assertEqual(graph.nodes[edge.dst].kind, PrimitiveKind.STEM)

    def test_nodes_wrap_original_primitives(self):
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 30, 24, 42)
        split = PrimitiveSplit(primitives=(head,), status=SplitStatus.ACCEPTED)

        graph = graph_from_split(split)

        (node,) = graph.nodes.values()
        self.assertIs(node.primitive, head)
        self.assertEqual(node.kind, PrimitiveKind.NOTEHEAD)

    def test_stem_to_beam_edges(self):
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 30, 24, 42)
        stem = make_primitive(PrimitiveKind.STEM, 22, 10, 24, 40)
        beam = make_primitive(PrimitiveKind.BEAM, 22, 8, 60, 12)
        split = PrimitiveSplit(
            primitives=(head, stem, beam),
            head_to_stem=((0, 1),),
            stem_to_beam=((1, 2),),
            status=SplitStatus.ACCEPTED,
        )

        graph = graph_from_split(split)

        self.assertEqual(len(graph.nodes), 3)
        relations = sorted(e.relation for e in graph.edges)
        self.assertEqual(
            relations, sorted((Relation.HEAD_TO_STEM, Relation.STEM_TO_BEAM))
        )
        beam_edge = next(e for e in graph.edges if e.relation == Relation.STEM_TO_BEAM)
        self.assertEqual(graph.nodes[beam_edge.src].kind, PrimitiveKind.STEM)
        self.assertEqual(graph.nodes[beam_edge.dst].kind, PrimitiveKind.BEAM)

    def test_staff_metrics_are_stashed_on_graph(self):
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 30, 24, 42)
        split = PrimitiveSplit(
            primitives=(head,),
            status=SplitStatus.ACCEPTED,
            staff_space=10.0,
            staff_line=2.0,
        )

        graph = graph_from_split(split)

        self.assertEqual(graph.staff_space, 10.0)
        self.assertEqual(graph.staff_line, 2.0)


if __name__ == "__main__":
    unittest.main()
