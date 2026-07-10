#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "deepscores/io.h"
#include "reconstruction/assembly.h"
#include "reconstruction/semantics.h"

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

	omr::PrimitiveSplit quarter_note_split(int origin_offset = 0)
	{
		omr::PrimitiveSplit split;
		split.primitives.push_back(make_primitive(omr::PrimitiveKind::NOTEHEAD,
												  10 + origin_offset, 46,
												  24 + origin_offset, 54));
		split.primitives.push_back(make_primitive(omr::PrimitiveKind::STEM,
												  22 + origin_offset, 20,
												  24 + origin_offset, 46));
		split.head_to_stem = { { 0, 1 } };
		split.status = omr::SplitStatus::ACCEPTED;
		split.staff_space = 10.0f;
		split.staff_lines = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };
		return split;
	}

	omr::PrimitiveSplit beamed_pair_split()
	{
		omr::PrimitiveSplit split;
		split.primitives.push_back(
			make_primitive(omr::PrimitiveKind::NOTEHEAD, 10, 44, 24, 52));
		split.primitives.push_back(
			make_primitive(omr::PrimitiveKind::STEM, 22, 18, 24, 46));
		split.primitives.push_back(
			make_primitive(omr::PrimitiveKind::NOTEHEAD, 40, 44, 54, 52));
		split.primitives.push_back(
			make_primitive(omr::PrimitiveKind::STEM, 52, 18, 54, 46));
		split.primitives.push_back(
			make_primitive(omr::PrimitiveKind::BEAM, 22, 14, 54, 18));
		split.head_to_stem = { { 0, 1 }, { 2, 3 } };
		split.stem_to_beam = { { 1, 4 }, { 3, 4 } };
		split.status = omr::SplitStatus::ACCEPTED;
		split.staff_space = 10.0f;
		split.staff_lines = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };
		return split;
	}

	omr::NotationGraph finish_graph(const omr::PrimitiveSplit& split,
									bool pitches = false)
	{
		omr::NotationGraph graph = omr::graph_from_split(split);
		omr::compute_staff_positions(graph);
		omr::run_rewrite(graph);
		omr::assign_durations(graph);
		if (pitches)
			omr::assign_pitches(graph);
		return graph;
	}

	omr::DatasetImage image()
	{
		return omr::DatasetImage{ "42", "page.png", 200, 200, {} };
	}

	std::unordered_map<std::string, omr::DatasetCategory> fake_categories()
	{
		return {
			{ "135",
			  { "135",
				"stem",
				"deepscores",
				7,
				{ { "name", "stem" },
				  { "annotation_set", "deepscores" },
				  { "color", 7 } } } },
			{ "208",
			  { "208",
				"noteheadBlackOnLine",
				"deepscores",
				8,
				{ { "name", "noteheadBlackOnLine" },
				  { "annotation_set", "deepscores" },
				  { "color", 8 } } } },
			{ "209",
			  { "209",
				"noteheadBlackInSpace",
				"deepscores",
				9,
				{ { "name", "noteheadBlackInSpace" },
				  { "annotation_set", "deepscores" },
				  { "color", 9 } } } },
			{ "300",
			  { "300",
				"beam",
				"deepscores",
				11,
				{ { "name", "beam" },
				  { "annotation_set", "deepscores" },
				  { "color", 11 } } } },
			{ "1",
			  { "1",
				"clefG",
				"deepscores",
				2,
				{ { "name", "clefG" },
				  { "annotation_set", "deepscores" },
				  { "color", 2 } } } },
			{ "999",
			  { "999",
				"restQuarter",
				"muscima++",
				5,
				{ { "name", "restQuarter" },
				  { "annotation_set", "muscima++" },
				  { "color", 5 } } } },
		};
	}

} // namespace

TEST(NoteheadCatName, ClassifiesLineSpaceAndMissingRows)
{
	EXPECT_EQ(
		omr::notehead_cat_name({ 0, 46, 14, 54 }, { 10, 20, 30, 40, 50 }, 2.0),
		"noteheadBlackOnLine");
	EXPECT_EQ(
		omr::notehead_cat_name({ 0, 41, 14, 49 }, { 10, 20, 30, 40, 50 }, 2.0),
		"noteheadBlackInSpace");
	EXPECT_EQ(omr::notehead_cat_name({ 0, 41, 14, 49 }, {}, 2.0),
			  "noteheadBlackOnLine");
}

TEST(BuildPredictedPage, AnnotationsCarryOriginAndCategories)
{
	omr::PrimitiveSplit split = quarter_note_split();
	omr::NotationGraph graph = finish_graph(split);
	const omr::PageIO page =
		omr::build_predicted_page(image(), { { &split, &graph, { 100, 200 } } },
								  { { 10.0, 20.0, 30.0, 40.0, 50.0 } }, 2.0);

	auto head = std::find_if(page.annotations.begin(), page.annotations.end(),
							 [](const omr::Annotation& ann) {
								 return ann.cat_name.rfind("notehead", 0) == 0;
							 });
	ASSERT_NE(head, page.annotations.end());
	EXPECT_DOUBLE_EQ(head->bbox.x0, 110.0);
	EXPECT_DOUBLE_EQ(head->bbox.y0, 246.0);
	EXPECT_DOUBLE_EQ(head->bbox.x1, 124.0);
	EXPECT_DOUBLE_EQ(head->bbox.y1, 254.0);
	EXPECT_EQ(head->ann_id, "pred-0-0");

	auto stem = std::find_if(
		page.annotations.begin(), page.annotations.end(),
		[](const omr::Annotation& ann) { return ann.cat_name == "stem"; });
	ASSERT_NE(stem, page.annotations.end());
	EXPECT_DOUBLE_EQ(stem->bbox.x0, 122.0);
	EXPECT_DOUBLE_EQ(stem->bbox.y0, 220.0);
	EXPECT_EQ(stem->ann_id, "pred-0-1");
}

TEST(BuildPredictedPage, DurationPitchAndEdgesMatchReference)
{
	omr::PrimitiveSplit split = beamed_pair_split();
	omr::NotationGraph graph = finish_graph(split, true);
	const omr::PageIO page =
		omr::build_predicted_page(image(), { { &split, &graph, { 0, 0 } } },
								  { { 10.0, 20.0, 30.0, 40.0, 50.0 } }, 2.0);

	EXPECT_EQ(page.duration_sequence,
			  (std::vector<std::string>{ "eighth", "eighth" }));
	ASSERT_EQ(page.pitch_sequence.size(), 2u);
	for (const auto& pitch : page.pitch_sequence)
		EXPECT_NE(std::string("CDEFGAB").find(pitch[0]), std::string::npos);

	std::vector<std::string> relations;
	for (const auto& edge : page.graph_edges)
		relations.push_back(edge.relation);
	std::sort(relations.begin(), relations.end());
	EXPECT_EQ(relations,
			  (std::vector<std::string>{ "head_to_stem", "head_to_stem",
										 "stem_to_beam", "stem_to_beam" }));
}

TEST(BuildPredictedPage, MultipleCropsUseDistinctIds)
{
	omr::PrimitiveSplit split = quarter_note_split();
	omr::NotationGraph graph = finish_graph(split);
	const omr::PageIO page = omr::build_predicted_page(
		image(),
		{ { &split, &graph, { 0, 0 } }, { &split, &graph, { 50, 60 } } },
		{ { 10.0, 20.0, 30.0, 40.0, 50.0 }, { 10.0, 20.0, 30.0, 40.0, 50.0 } },
		2.0);
	std::set<std::string> ids;
	for (const auto& ann : page.annotations)
		ids.insert(ann.ann_id);
	EXPECT_EQ(ids.size(), 4u);
	EXPECT_TRUE(ids.count("pred-0-0"));
	EXPECT_TRUE(ids.count("pred-1-1"));
	EXPECT_EQ(page.duration_sequence,
			  (std::vector<std::string>{ "quarter", "quarter" }));
}

TEST(WritePredictionsJson, EmitsDeepScoresSchema)
{
	omr::PageIO page;
	page.image_id = "42";
	page.filename = "page.png";
	page.width = 200;
	page.height = 200;
	page.annotations = {
		{ "pred-0-0", { 10, 20, 30, 40 }, "noteheadBlackOnLine", 120, 1.0 },
		{ "pred-0-1", { 22, 5, 24, 40 }, "stem", 70, 1.0 },
	};
	page.graph_edges = { { "head_to_stem", "pred-0-0", "pred-0-1" } };
	page.duration_sequence = { "quarter" };

	const auto out = std::filesystem::temp_directory_path()
		/ "omr_cpp_predictions_test" / "out" / "page_42.json";
	omr::write_predictions_json(page, fake_categories(), out);
	std::ifstream input(out);
	ASSERT_TRUE(input.good());
	nlohmann::json data;
	input >> data;

	EXPECT_EQ(data["info"]["version"], "1.0");
	EXPECT_EQ(data["annotation_sets"], nlohmann::json::array({ "deepscores" }));
	std::set<std::string> kept_names;
	for (const auto& [id, cat] : data["categories"].items())
	{
		(void)id;
		kept_names.insert(cat["name"].get<std::string>());
	}
	EXPECT_EQ(
		kept_names,
		(std::set<std::string>{ "noteheadBlackOnLine", "noteheadBlackInSpace",
								"stem", "beam" }));
	EXPECT_EQ(data["annotations"]["pred-0-0"]["cat_id"],
			  nlohmann::json::array({ "208" }));
	EXPECT_EQ(data["annotations"]["pred-0-0"]["a_bbox"],
			  nlohmann::json::array({ 10.0, 20.0, 30.0, 40.0 }));
	EXPECT_EQ(data["annotations"]["pred-0-0"]["o_bbox"].size(), 8u);
	EXPECT_EQ(data["annotations"]["pred-0-1"]["cat_id"],
			  nlohmann::json::array({ "135" }));
	EXPECT_EQ(data["graph"]["edges"].size(), 1u);
	EXPECT_EQ(data["duration_sequence"], nlohmann::json::array({ "quarter" }));
}

TEST(WritePredictionsJson, UnknownCategoryIsDropped)
{
	omr::PageIO page;
	page.image_id = "1";
	page.filename = "x.png";
	page.width = 10;
	page.height = 10;
	page.annotations = {
		{ "x", { 0, 0, 1, 1 }, "notInTaxonomy", 1, 1.0 },
	};
	const auto out = std::filesystem::temp_directory_path()
		/ "omr_cpp_predictions_test" / "unknown.json";
	omr::write_predictions_json(page, fake_categories(), out);
	std::ifstream input(out);
	nlohmann::json data;
	input >> data;
	EXPECT_TRUE(data["annotations"].empty());
}

TEST(DeriveGroundTruth, KeepsScopedCategoriesWhenDataPresent)
{
	const std::filesystem::path data_dir =
		std::filesystem::temp_directory_path() / "omr_cpp_gt_fixture";
	std::filesystem::create_directories(data_dir);
	const nlohmann::json payload = {
		{ "categories",
		  { { "1",
			  { { "name", "staff" },
				{ "annotation_set", "deepscores" },
				{ "color", 1 } } },
			{ "7",
			  { { "name", "stem" },
				{ "annotation_set", "deepscores" },
				{ "color", 7 } } },
			{ "8",
			  { { "name", "noteheadBlackOnLine" },
				{ "annotation_set", "deepscores" },
				{ "color", 8 } } },
			{ "9",
			  { { "name", "noteheadBlackInSpace" },
				{ "annotation_set", "deepscores" },
				{ "color", 9 } } },
			{ "11",
			  { { "name", "beam" },
				{ "annotation_set", "deepscores" },
				{ "color", 11 } } },
			{ "99",
			  { { "name", "clefG" },
				{ "annotation_set", "deepscores" },
				{ "color", 99 } } } } },
		{ "images",
		  nlohmann::json::array({ { { "id", 42 },
									{ "filename", "page.png" },
									{ "width", 80 },
									{ "height", 80 } } }) },
		{ "annotations",
		  { { "h",
			  { { "img_id", 42 },
				{ "cat_id", nlohmann::json::array({ "8" }) },
				{ "a_bbox", nlohmann::json::array({ 10, 40, 20, 50 }) },
				{ "area", 100 } } },
			{ "s",
			  { { "img_id", 42 },
				{ "cat_id", nlohmann::json::array({ "7" }) },
				{ "a_bbox", nlohmann::json::array({ 20, 20, 22, 50 }) },
				{ "area", 60 } } },
			{ "c",
			  { { "img_id", 42 },
				{ "cat_id", nlohmann::json::array({ "99" }) },
				{ "a_bbox", nlohmann::json::array({ 0, 0, 5, 5 }) },
				{ "area", 25 } } } } },
	};
	std::ofstream(data_dir / "deepscores_test.json") << payload.dump(2);

	const omr::DeepScoresDataset dataset(data_dir, "test");
	const omr::PageIO gt = omr::derive_ground_truth(dataset, "42");
	std::set<std::string> scoped(std::begin(omr::SCOPED_CATEGORY_NAMES),
								 std::end(omr::SCOPED_CATEGORY_NAMES));
	for (const auto& ann : gt.annotations)
		EXPECT_TRUE(scoped.count(ann.cat_name));
	ASSERT_EQ(gt.annotations.size(), 2u);
	for (const auto& token : gt.duration_sequence)
		EXPECT_TRUE(token == "quarter" || token == "eighth");
	std::set<std::string> ids;
	for (const auto& ann : gt.annotations)
		ids.insert(ann.ann_id);
	for (const auto& edge : gt.graph_edges)
	{
		EXPECT_TRUE(ids.count(edge.src_id));
		EXPECT_TRUE(ids.count(edge.dst_id));
	}
}
