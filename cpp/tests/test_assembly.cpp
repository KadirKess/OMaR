#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

#include "reconstruction/assembly.h"

namespace
{

	omr::Primitive make_primitive(omr::PrimitiveKind kind, int x0, int y0,
								  int x1, int y1, float confidence = 1.0f)
	{
		omr::Mask mask(64, 64);
		for (int y = y0; y < y1; ++y)
			for (int x = x0; x < x1; ++x)
				mask.set(y, x, 255);
		return omr::Primitive(kind, std::move(mask), confidence);
	}

	omr::PrimitiveSplit
	split_with(std::vector<omr::Primitive> primitives,
			   std::vector<std::pair<int, int>> head_to_stem = {},
			   std::vector<std::pair<int, int>> stem_to_beam = {})
	{
		omr::PrimitiveSplit split;
		split.primitives = std::move(primitives);
		split.head_to_stem = std::move(head_to_stem);
		split.stem_to_beam = std::move(stem_to_beam);
		split.status = omr::SplitStatus::ACCEPTED;
		split.staff_space = 10.0f;
		split.staff_lines = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };
		return split;
	}

	std::vector<omr::Relation> relations(const omr::NotationGraph& graph)
	{
		std::vector<omr::Relation> out;
		for (const auto& edge : graph.edges)
			out.push_back(edge.relation);
		return out;
	}

} // namespace

TEST(GraphFromSplit, SingleNoteCreatesTwoNodesAndOneEdge)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 30, 24, 42),
		  make_primitive(omr::PrimitiveKind::STEM, 22, 10, 24, 40) },
		{ { 0, 1 } });

	const omr::NotationGraph graph = omr::graph_from_split(split);

	ASSERT_EQ(graph.nodes.size(), 2u);
	ASSERT_EQ(graph.edges.size(), 1u);
	const auto& edge = graph.edges[0];
	EXPECT_EQ(edge.relation, omr::Relation::HEAD_TO_STEM);
	EXPECT_EQ(graph.nodes.at(edge.src)->kind, omr::NodeKind::NOTEHEAD);
	EXPECT_EQ(graph.nodes.at(edge.dst)->kind, omr::NodeKind::STEM);
}

TEST(GraphFromSplit, NodesWrapOriginalPrimitives)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 30, 24, 42) });

	const omr::NotationGraph graph = omr::graph_from_split(split);

	ASSERT_EQ(graph.nodes.size(), 1u);
	const auto node = graph.nodes.begin()->second;
	EXPECT_EQ(node->primitive, &split.primitives[0]);
	EXPECT_EQ(node->kind, omr::NodeKind::NOTEHEAD);
}

TEST(GraphFromSplit, StemToBeamEdges)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 30, 24, 42),
		  make_primitive(omr::PrimitiveKind::STEM, 22, 10, 24, 40),
		  make_primitive(omr::PrimitiveKind::BEAM, 22, 8, 60, 12) },
		{ { 0, 1 } }, { { 1, 2 } });

	const omr::NotationGraph graph = omr::graph_from_split(split);

	ASSERT_EQ(graph.nodes.size(), 3u);
	auto rels = relations(graph);
	std::sort(rels.begin(), rels.end());
	std::vector<omr::Relation> expected{ omr::Relation::HEAD_TO_STEM,
										 omr::Relation::STEM_TO_BEAM };
	std::sort(expected.begin(), expected.end());
	EXPECT_EQ(rels, expected);
	const auto beam_edge = std::find_if(
		graph.edges.begin(), graph.edges.end(), [](const omr::Edge& edge) {
			return edge.relation == omr::Relation::STEM_TO_BEAM;
		});
	ASSERT_NE(beam_edge, graph.edges.end());
	EXPECT_EQ(graph.nodes.at(beam_edge->src)->kind, omr::NodeKind::STEM);
	EXPECT_EQ(graph.nodes.at(beam_edge->dst)->kind, omr::NodeKind::BEAM);
}

TEST(GraphFromSplit, StaffMetricsAreStashedOnGraph)
{
	omr::PrimitiveSplit split;
	split.primitives.push_back(
		make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 30, 24, 42));
	split.status = omr::SplitStatus::ACCEPTED;
	split.staff_space = 10.0f;
	split.staff_lines = { 6.0f, 16.0f, 26.0f, 36.0f, 46.0f };

	const omr::NotationGraph graph = omr::graph_from_split(split);

	EXPECT_DOUBLE_EQ(graph.staff_space, 10.0);
	ASSERT_EQ(graph.staff_lines.size(), 5u);
	EXPECT_DOUBLE_EQ(graph.staff_lines[0], 6.0);
	EXPECT_DOUBLE_EQ(graph.staff_lines[4], 46.0);
}

TEST(ComputeStaffPositions, NoteOnBottomLineIsPositionZero)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 46, 24, 54) });
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::compute_staff_positions(graph);

	const auto node = graph.nodes.begin()->second;
	ASSERT_TRUE(node->has_attr("staff_pos"));
	EXPECT_DOUBLE_EQ(node->get<double>("staff_pos").value(), 0.0);
}

TEST(ComputeStaffPositions, NoteOneSpaceHigherIsPositionOne)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 41, 24, 49) });
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::compute_staff_positions(graph);

	const auto node = graph.nodes.begin()->second;
	ASSERT_TRUE(node->has_attr("staff_pos"));
	EXPECT_DOUBLE_EQ(node->get<double>("staff_pos").value(), 1.0);
}

TEST(ComputeStaffPositions, NoteOnNextLineUpIsPositionTwo)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 36, 24, 44) });
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::compute_staff_positions(graph);

	const auto node = graph.nodes.begin()->second;
	ASSERT_TRUE(node->has_attr("staff_pos"));
	EXPECT_DOUBLE_EQ(node->get<double>("staff_pos").value(), 2.0);
}

TEST(ComputeStaffPositions, LedgerNoteAboveStaffIsLargePositive)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 2, 24, 8) });
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::compute_staff_positions(graph);

	const auto node = graph.nodes.begin()->second;
	ASSERT_TRUE(node->has_attr("staff_pos"));
	EXPECT_GT(node->get<double>("staff_pos").value(), 8.0);
}

TEST(ComputeStaffPositions, XOrderFollowsLeftToRight)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 40, 46, 50, 54),
		  make_primitive(omr::PrimitiveKind::NOTEHEAD, 5, 46, 15, 54) });
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::compute_staff_positions(graph);

	for (const auto& [id, node] : graph.nodes)
	{
		(void)id;
		ASSERT_TRUE(node->has_attr("x_order"));
		if (node->primitive->bbox.x0 == 5)
			EXPECT_EQ(node->get<int>("x_order").value(), 0);
		if (node->primitive->bbox.x0 == 40)
			EXPECT_EQ(node->get<int>("x_order").value(), 1);
	}
}

TEST(ComputeStaffPositions, NoStaffLinesIsNoop)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 46, 24, 54) });
	split.staff_lines.clear();
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::compute_staff_positions(graph);

	const auto node = graph.nodes.begin()->second;
	EXPECT_FALSE(node->has_attr("staff_pos"));
	EXPECT_FALSE(node->has_attr("x_order"));
}

TEST(BuildSpatialEdges, StackedNoteheadsAreAlignedVertically)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 20, 44, 34, 52),
		  make_primitive(omr::PrimitiveKind::NOTEHEAD, 20, 24, 34, 32) });
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::build_spatial_edges(graph);

	const auto rels = relations(graph);
	EXPECT_NE(
		std::find(rels.begin(), rels.end(), omr::Relation::ALIGNED_VERTICAL),
		rels.end());
}

TEST(BuildSpatialEdges, AboveEdgePointsFromHigherToLower)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 20, 44, 34, 52),
		  make_primitive(omr::PrimitiveKind::NOTEHEAD, 20, 24, 34, 32) });
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::build_spatial_edges(graph);

	std::vector<omr::Edge> above;
	for (const auto& edge : graph.edges)
		if (edge.relation == omr::Relation::ABOVE)
			above.push_back(edge);
	ASSERT_EQ(above.size(), 1u);
	const auto src = graph.nodes.at(above[0].src);
	const auto dst = graph.nodes.at(above[0].dst);
	EXPECT_EQ(src->primitive, &split.primitives[1]);
	EXPECT_EQ(dst->primitive, &split.primitives[0]);
}

TEST(BuildSpatialEdges, HorizontallySeparatedPrimitivesGetNoEdges)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 5, 44, 15, 52),
		  make_primitive(omr::PrimitiveKind::NOTEHEAD, 45, 44, 55, 52) });
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::build_spatial_edges(graph);

	EXPECT_TRUE(graph.edges.empty());
}

TEST(BuildSpatialEdges, NoStaffSpaceIsNoop)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 20, 44, 34, 52),
		  make_primitive(omr::PrimitiveKind::NOTEHEAD, 20, 24, 34, 32) });
	split.staff_space = 0.0f;
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::build_spatial_edges(graph);

	EXPECT_TRUE(graph.edges.empty());
}

TEST(Rewrite, NoteheadAndStemCollapseIntoNote)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 44, 24, 52),
		  make_primitive(omr::PrimitiveKind::STEM, 22, 20, 24, 46) },
		{ { 0, 1 } });
	omr::NotationGraph graph = omr::graph_from_split(split);
	omr::compute_staff_positions(graph);
	const double head_pos = graph.nodes.at(0)->get<double>("staff_pos").value();

	omr::run_rewrite(graph);

	ASSERT_EQ(graph.nodes.size(), 1u);
	const auto note = graph.nodes.begin()->second;
	EXPECT_EQ(note->kind, omr::NodeKind::NOTE);
	EXPECT_DOUBLE_EQ(note->get<double>("staff_pos").value(), head_pos);
	EXPECT_EQ(note->get<const omr::Primitive*>("notehead").value(),
			  &split.primitives[0]);
	EXPECT_EQ(note->get<const omr::Primitive*>("stem").value(),
			  &split.primitives[1]);
	EXPECT_TRUE(graph.edges.empty());
}

TEST(Rewrite, TwoNoteheadsOnOneStemBecomeChord)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 20, 44, 34, 52),
		  make_primitive(omr::PrimitiveKind::STEM, 32, 20, 34, 46),
		  make_primitive(omr::PrimitiveKind::NOTEHEAD, 20, 24, 34, 32) },
		{ { 0, 1 }, { 2, 1 } });
	omr::NotationGraph graph = omr::graph_from_split(split);
	omr::compute_staff_positions(graph);

	omr::run_rewrite(graph);

	ASSERT_EQ(graph.nodes.size(), 1u);
	const auto chord = graph.nodes.begin()->second;
	EXPECT_EQ(chord->kind, omr::NodeKind::CHORD);
	const auto noteheads =
		chord->get<std::vector<const omr::Primitive*>>("noteheads");
	ASSERT_TRUE(noteheads.has_value());
	ASSERT_EQ(noteheads->size(), 2u);
	EXPECT_EQ((*noteheads)[0], &split.primitives[0]);
	EXPECT_EQ((*noteheads)[1], &split.primitives[2]);
	const auto staff_positions =
		chord->get<std::vector<double>>("staff_positions");
	ASSERT_TRUE(staff_positions.has_value());
	ASSERT_EQ(staff_positions->size(), 2u);
	EXPECT_LT((*staff_positions)[0], (*staff_positions)[1]);
}

TEST(Rewrite, TwoNotesSharingBeamBecomeBeamedGroup)
{
	auto split = split_with(
		{ make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 44, 24, 52),
		  make_primitive(omr::PrimitiveKind::STEM, 22, 18, 24, 46),
		  make_primitive(omr::PrimitiveKind::NOTEHEAD, 40, 44, 54, 52),
		  make_primitive(omr::PrimitiveKind::STEM, 52, 18, 54, 46),
		  make_primitive(omr::PrimitiveKind::BEAM, 22, 14, 54, 18) },
		{ { 0, 1 }, { 2, 3 } }, { { 1, 4 }, { 3, 4 } });
	omr::NotationGraph graph = omr::graph_from_split(split);
	omr::compute_staff_positions(graph);

	omr::run_rewrite(graph);

	ASSERT_EQ(graph.nodes.size(), 1u);
	const auto group = graph.nodes.begin()->second;
	EXPECT_EQ(group->kind, omr::NodeKind::BEAMED_GROUP);
	const auto members = group->get<std::vector<omr::NodePtr>>("members");
	ASSERT_TRUE(members.has_value());
	ASSERT_EQ(members->size(), 2u);
	EXPECT_TRUE(std::all_of(members->begin(), members->end(),
							[](const omr::NodePtr& member) {
								return member->kind == omr::NodeKind::NOTE;
							}));
	EXPECT_LT((*members)[0]->get<int>("x_order").value(),
			  (*members)[1]->get<int>("x_order").value());
}

TEST(Rewrite, UnmatchedNodeSurvives)
{
	omr::PrimitiveSplit split;
	split.primitives.push_back(
		make_primitive(omr::PrimitiveKind::UNKNOWN, 10, 10, 24, 24));
	split.status = omr::SplitStatus::UNCERTAIN;
	split.staff_space = 10.0f;
	omr::NotationGraph graph = omr::graph_from_split(split);

	omr::run_rewrite(graph);

	ASSERT_EQ(graph.nodes.size(), 1u);
	EXPECT_EQ(graph.nodes.begin()->second->kind, omr::NodeKind::UNKNOWN);
}
