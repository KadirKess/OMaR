#include <gtest/gtest.h>
#include <vector>

#include "reconstruction/assembly.h"
#include "reconstruction/semantics.h"

namespace
{

	omr::Primitive filled_head(int x0, int y0, int x1, int y1)
	{
		omr::Mask mask(64, 80);
		for (int y = y0; y < y1; ++y)
			for (int x = x0; x < x1; ++x)
				mask.set(y, x, 255);
		return omr::Primitive(omr::PrimitiveKind::NOTEHEAD, std::move(mask),
							  1.0f);
	}

	omr::Primitive open_head(int x0, int y0, int x1, int y1)
	{
		omr::Mask mask(64, 80);
		for (int y = y0; y < y1; ++y)
			for (int x = x0; x < x1; ++x)
				mask.set(y, x, 255);
		for (int y = y0 + 2; y < y1 - 2; ++y)
			for (int x = x0 + 2; x < x1 - 2; ++x)
				mask.set(y, x, 0);
		return omr::Primitive(omr::PrimitiveKind::NOTEHEAD, std::move(mask),
							  1.0f);
	}

	omr::Primitive stem(int x0, int y0, int x1, int y1)
	{
		omr::Mask mask(64, 80);
		for (int y = y0; y < y1; ++y)
			for (int x = x0; x < x1; ++x)
				mask.set(y, x, 255);
		return omr::Primitive(omr::PrimitiveKind::STEM, std::move(mask), 1.0f);
	}

	omr::Primitive beam(int x0, int y0, int x1, int y1)
	{
		omr::Mask mask(64, 80);
		for (int y = y0; y < y1; ++y)
			for (int x = x0; x < x1; ++x)
				mask.set(y, x, 255);
		return omr::Primitive(omr::PrimitiveKind::BEAM, std::move(mask), 1.0f);
	}

	struct BuiltGraph
	{
		omr::PrimitiveSplit split;
		omr::NotationGraph graph;
	};

	BuiltGraph build(std::vector<omr::Primitive> primitives,
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
		omr::NotationGraph graph = omr::graph_from_split(split);
		omr::compute_staff_positions(graph);
		omr::run_rewrite(graph);
		return BuiltGraph{ std::move(split), std::move(graph) };
	}

} // namespace

TEST(Duration, FilledHeadWithStemIsQuarter)
{
	BuiltGraph built = build(
		{ filled_head(20, 44, 34, 52), stem(32, 20, 34, 46) }, { { 0, 1 } });

	omr::assign_durations(built.graph);

	ASSERT_EQ(built.graph.nodes.size(), 1u);
	const auto note = built.graph.nodes.begin()->second;
	EXPECT_DOUBLE_EQ(note->get<double>("duration").value(), omr::QUARTER);
}

TEST(Duration, OpenHeadWithStemIsHalf)
{
	BuiltGraph built = build(
		{ open_head(20, 40, 34, 54), stem(32, 16, 34, 42) }, { { 0, 1 } });

	omr::assign_durations(built.graph);

	ASSERT_EQ(built.graph.nodes.size(), 1u);
	const auto note = built.graph.nodes.begin()->second;
	EXPECT_DOUBLE_EQ(note->get<double>("duration").value(), omr::HALF);
}

TEST(Duration, BeamedNotesAreEighths)
{
	BuiltGraph built =
		build({ filled_head(10, 44, 24, 52), stem(22, 18, 24, 46),
				filled_head(40, 44, 54, 52), stem(52, 18, 54, 46),
				beam(22, 14, 54, 18) },
			  { { 0, 1 }, { 2, 3 } }, { { 1, 4 }, { 3, 4 } });

	omr::assign_durations(built.graph);

	ASSERT_EQ(built.graph.nodes.size(), 1u);
	const auto group = built.graph.nodes.begin()->second;
	EXPECT_EQ(group->kind, omr::NodeKind::BEAMED_GROUP);
	const auto members = group->get<std::vector<omr::NodePtr>>("members");
	ASSERT_TRUE(members.has_value());
	for (const auto& member : *members)
		EXPECT_DOUBLE_EQ(member->get<double>("duration").value(), omr::EIGHTH);
}

TEST(Pitch, BottomLineIsE4InTreble)
{
	BuiltGraph built = build(
		{ filled_head(20, 46, 34, 54), stem(32, 22, 34, 48) }, { { 0, 1 } });

	omr::assign_pitches(built.graph, "treble");

	ASSERT_EQ(built.graph.nodes.size(), 1u);
	const auto note = built.graph.nodes.begin()->second;
	EXPECT_EQ(note->get<std::string>("pitch").value(), "E4");
}

TEST(Pitch, NextLineUpIsG4)
{
	BuiltGraph built = build(
		{ filled_head(20, 36, 34, 44), stem(32, 12, 34, 40) }, { { 0, 1 } });

	omr::assign_pitches(built.graph, "treble");

	ASSERT_EQ(built.graph.nodes.size(), 1u);
	const auto note = built.graph.nodes.begin()->second;
	EXPECT_EQ(note->get<std::string>("pitch").value(), "G4");
}

TEST(Pitch, BassClefBottomLineIsG2)
{
	BuiltGraph built = build(
		{ filled_head(20, 46, 34, 54), stem(32, 22, 34, 48) }, { { 0, 1 } });

	omr::assign_pitches(built.graph, "bass");

	ASSERT_EQ(built.graph.nodes.size(), 1u);
	const auto note = built.graph.nodes.begin()->second;
	EXPECT_EQ(note->get<std::string>("pitch").value(), "G2");
}

TEST(Validation, FourQuartersFillAFourFourMeasure)
{
	std::vector<omr::Primitive> primitives;
	std::vector<std::pair<int, int>> head_to_stem;
	for (int i = 0; i < 4; ++i)
	{
		const int x = 5 + i * 15;
		primitives.push_back(filled_head(x, 44, x + 12, 52));
		primitives.push_back(stem(x + 10, 20, x + 12, 46));
		head_to_stem.emplace_back(2 * i, 2 * i + 1);
	}
	BuiltGraph built = build(std::move(primitives), std::move(head_to_stem));

	omr::assign_durations(built.graph);
	const omr::MeasureCheck check =
		omr::validate_measure_durations(built.graph, { 4, 4 });

	EXPECT_TRUE(check.is_valid);
	EXPECT_DOUBLE_EQ(check.actual, 1.0);
	EXPECT_DOUBLE_EQ(check.expected, 1.0);
}

TEST(Validation, MissingBeatIsInvalid)
{
	BuiltGraph built = build(
		{ filled_head(20, 44, 34, 52), stem(32, 20, 34, 46) }, { { 0, 1 } });

	omr::assign_durations(built.graph);
	const omr::MeasureCheck check =
		omr::validate_measure_durations(built.graph, { 4, 4 });

	EXPECT_FALSE(check.is_valid);
	EXPECT_DOUBLE_EQ(check.actual, 0.25);
}
