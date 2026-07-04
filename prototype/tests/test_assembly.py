import unittest

import numpy as np

from src.assembly import (
    build_spatial_edges,
    compute_staff_positions,
    graph_from_split,
    run_rewrite,
)
from src.graph import NodeKind, Relation
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
            staff_lines=(6.0, 16.0, 26.0, 36.0, 46.0),
        )

        graph = graph_from_split(split)

        self.assertEqual(graph.staff_space, 10.0)
        self.assertEqual(graph.staff_lines, (6.0, 16.0, 26.0, 36.0, 46.0))


class ComputeStaffPositionsTests(unittest.TestCase):
    STAFF_LINES = (10.0, 20.0, 30.0, 40.0, 50.0)
    STAFF_SPACE = 10.0

    def graph_with(self, *primitives):
        split = PrimitiveSplit(
            primitives=primitives,
            status=SplitStatus.ACCEPTED,
            staff_space=self.STAFF_SPACE,
            staff_lines=self.STAFF_LINES,
        )
        graph = graph_from_split(split)
        compute_staff_positions(graph)
        return graph

    def only_node(self, graph):
        (node,) = graph.nodes.values()
        return node

    def test_note_on_bottom_line_is_position_zero(self):
        # bbox centered on y=50 (the bottom reference line)
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 46, 24, 54)
        node = self.only_node(self.graph_with(head))
        self.assertAlmostEqual(node.attrs["staff_pos"], 0.0)

    def test_note_one_space_higher_is_position_one(self):
        # centered on y=45, one half-space (5px) above the bottom line
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 41, 24, 49)
        node = self.only_node(self.graph_with(head))
        self.assertAlmostEqual(node.attrs["staff_pos"], 1.0)

    def test_note_on_next_line_up_is_position_two(self):
        # centered on y=40 (second line from bottom)
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 36, 24, 44)
        node = self.only_node(self.graph_with(head))
        self.assertAlmostEqual(node.attrs["staff_pos"], 2.0)

    def test_ledger_note_above_staff_is_large_positive(self):
        # centered on y=5, above the top line (y=10)
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 2, 24, 8)
        node = self.only_node(self.graph_with(head))
        self.assertGreater(node.attrs["staff_pos"], 8.0)

    def test_x_order_follows_left_to_right(self):
        left = make_primitive(PrimitiveKind.NOTEHEAD, 5, 46, 15, 54)
        right = make_primitive(PrimitiveKind.NOTEHEAD, 40, 46, 50, 54)
        graph = self.graph_with(right, left)  # inserted right-first on purpose
        by_x0 = {n.primitive.bbox[0]: n.attrs["x_order"] for n in graph.nodes.values()}
        self.assertEqual(by_x0[5], 0)
        self.assertEqual(by_x0[40], 1)

    def test_no_staff_lines_is_a_noop(self):
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 46, 24, 54)
        split = PrimitiveSplit(
            primitives=(head,),
            status=SplitStatus.ACCEPTED,
            staff_space=self.STAFF_SPACE,
            staff_lines=(),
        )
        graph = graph_from_split(split)
        compute_staff_positions(graph)
        node = self.only_node(graph)
        self.assertNotIn("staff_pos", node.attrs)
        self.assertNotIn("x_order", node.attrs)


class BuildSpatialEdgesTests(unittest.TestCase):
    STAFF_SPACE = 10.0
    STAFF_LINES = (10.0, 20.0, 30.0, 40.0, 50.0)

    def graph_with(self, *primitives):
        split = PrimitiveSplit(
            primitives=primitives,
            status=SplitStatus.ACCEPTED,
            staff_space=self.STAFF_SPACE,
            staff_lines=self.STAFF_LINES,
        )
        graph = graph_from_split(split)
        build_spatial_edges(graph)
        return graph

    def relations(self, graph):
        return [e.relation for e in graph.edges]

    def test_stacked_noteheads_are_aligned_vertically(self):
        # two noteheads in the same x-column, one above the other (a chord)
        lower = make_primitive(PrimitiveKind.NOTEHEAD, 20, 44, 34, 52)
        upper = make_primitive(PrimitiveKind.NOTEHEAD, 20, 24, 34, 32)
        graph = self.graph_with(lower, upper)
        self.assertIn(Relation.ALIGNED_VERTICAL, self.relations(graph))

    def test_above_edge_points_from_higher_to_lower(self):
        lower = make_primitive(PrimitiveKind.NOTEHEAD, 20, 44, 34, 52)
        upper = make_primitive(PrimitiveKind.NOTEHEAD, 20, 24, 34, 32)
        graph = self.graph_with(lower, upper)
        above = [e for e in graph.edges if e.relation == Relation.ABOVE]
        self.assertEqual(len(above), 1)
        # src is the higher primitive (smaller y), dst the lower one
        src, dst = graph.nodes[above[0].src], graph.nodes[above[0].dst]
        self.assertEqual(src.primitive, upper)
        self.assertEqual(dst.primitive, lower)

    def test_horizontally_separated_primitives_get_no_edges(self):
        # two noteheads far apart in x, same height -> no spatial relation
        left = make_primitive(PrimitiveKind.NOTEHEAD, 5, 44, 15, 52)
        right = make_primitive(PrimitiveKind.NOTEHEAD, 45, 44, 55, 52)
        graph = self.graph_with(left, right)
        self.assertEqual(graph.edges, [])

    def test_no_staff_space_is_a_noop(self):
        lower = make_primitive(PrimitiveKind.NOTEHEAD, 20, 44, 34, 52)
        upper = make_primitive(PrimitiveKind.NOTEHEAD, 20, 24, 34, 32)
        split = PrimitiveSplit(
            primitives=(lower, upper),
            status=SplitStatus.ACCEPTED,
            staff_space=0.0,
            staff_lines=self.STAFF_LINES,
        )
        graph = graph_from_split(split)
        build_spatial_edges(graph)
        self.assertEqual(graph.edges, [])


class RewriteTests(unittest.TestCase):
    def test_notehead_and_stem_collapse_into_a_note(self):
        head = make_primitive(PrimitiveKind.NOTEHEAD, 10, 44, 24, 52)
        stem = make_primitive(PrimitiveKind.STEM, 22, 20, 24, 46)
        split = PrimitiveSplit(
            primitives=(head, stem),
            head_to_stem=((0, 1),),
            status=SplitStatus.ACCEPTED,
            staff_space=10.0,
            staff_lines=(10.0, 20.0, 30.0, 40.0, 50.0),
        )
        graph = graph_from_split(split)
        compute_staff_positions(graph)
        head_pos = graph.nodes[0].attrs["staff_pos"]

        run_rewrite(graph)

        kinds = [n.kind for n in graph.nodes.values()]
        self.assertEqual(kinds, [NodeKind.NOTE])
        (note,) = graph.nodes.values()
        # staff_pos carried over, consumed primitives stashed for later passes
        self.assertAlmostEqual(note.attrs["staff_pos"], head_pos)
        self.assertIs(note.attrs["notehead"], head)
        self.assertIs(note.attrs["stem"], stem)
        # the consumed nodes and their edge are gone
        self.assertEqual(graph.edges, [])

    def test_two_noteheads_on_one_stem_become_a_chord(self):
        low = make_primitive(PrimitiveKind.NOTEHEAD, 20, 44, 34, 52)
        stem = make_primitive(PrimitiveKind.STEM, 32, 20, 34, 46)
        high = make_primitive(PrimitiveKind.NOTEHEAD, 20, 24, 34, 32)
        split = PrimitiveSplit(
            primitives=(low, stem, high),
            head_to_stem=((0, 1), (2, 1)),
            status=SplitStatus.ACCEPTED,
            staff_space=10.0,
            staff_lines=(10.0, 20.0, 30.0, 40.0, 50.0),
        )
        graph = graph_from_split(split)
        compute_staff_positions(graph)

        run_rewrite(graph)

        kinds = [n.kind for n in graph.nodes.values()]
        self.assertEqual(kinds, [NodeKind.CHORD])
        (chord,) = graph.nodes.values()
        self.assertEqual(chord.attrs["noteheads"], [low, high])  # sorted low->high
        self.assertEqual(len(chord.attrs["staff_positions"]), 2)
        self.assertLess(
            chord.attrs["staff_positions"][0], chord.attrs["staff_positions"][1]
        )

    def test_two_notes_sharing_a_beam_become_a_beamed_group(self):
        head_a = make_primitive(PrimitiveKind.NOTEHEAD, 10, 44, 24, 52)
        stem_a = make_primitive(PrimitiveKind.STEM, 22, 18, 24, 46)
        head_b = make_primitive(PrimitiveKind.NOTEHEAD, 40, 44, 54, 52)
        stem_b = make_primitive(PrimitiveKind.STEM, 52, 18, 54, 46)
        beam = make_primitive(PrimitiveKind.BEAM, 22, 14, 54, 18)
        split = PrimitiveSplit(
            primitives=(head_a, stem_a, head_b, stem_b, beam),
            head_to_stem=((0, 1), (2, 3)),
            stem_to_beam=((1, 4), (3, 4)),
            status=SplitStatus.ACCEPTED,
            staff_space=10.0,
            staff_lines=(10.0, 20.0, 30.0, 40.0, 50.0),
        )
        graph = graph_from_split(split)
        compute_staff_positions(graph)

        run_rewrite(graph)

        kinds = [n.kind for n in graph.nodes.values()]
        self.assertEqual(kinds, [NodeKind.BEAMED_GROUP])
        (group,) = graph.nodes.values()
        members = group.attrs["members"]
        self.assertEqual(len(members), 2)
        self.assertTrue(all(m.kind == NodeKind.NOTE for m in members))
        # members ordered left-to-right
        self.assertLess(members[0].attrs["x_order"], members[1].attrs["x_order"])

    def test_unmatched_node_survives(self):
        blob = make_primitive(PrimitiveKind.UNKNOWN, 10, 10, 24, 24)
        split = PrimitiveSplit(
            primitives=(blob,),
            status=SplitStatus.UNCERTAIN,
            staff_space=10.0,
        )
        graph = graph_from_split(split)

        run_rewrite(graph)

        (node,) = graph.nodes.values()
        self.assertEqual(node.kind, NodeKind.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
