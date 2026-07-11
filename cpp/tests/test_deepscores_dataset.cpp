#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "deepscores/dataset.h"

namespace
{
	struct AnnotationSpec
	{
		std::string cat_id;
		std::array<double, 4> bbox;
	};

	struct FixtureImage
	{
		std::string id;
		cv::Mat1b image;
		cv::Mat1b semantic;
		std::vector<AnnotationSpec> annotations;
	};

	nlohmann::json fixture_categories()
	{
		return {
			{ "1",
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
		};
	}

	void write_fixture_dataset(const std::filesystem::path& root,
							   const std::vector<FixtureImage>& images)
	{
		std::filesystem::create_directories(root / "images");
		std::filesystem::create_directories(root / "segmentation");

		nlohmann::json images_json = nlohmann::json::array();
		nlohmann::json annotations_json = nlohmann::json::object();
		int ann_counter = 0;

		for (const auto& img : images)
		{
			const std::string img_filename = "img_" + img.id + ".png";
			const std::string seg_filename = "img_" + img.id + "_seg.png";

			EXPECT_TRUE(cv::imwrite((root / "images" / img_filename).string(),
									img.image));
			EXPECT_TRUE(cv::imwrite(
				(root / "segmentation" / seg_filename).string(), img.semantic));

			images_json.push_back({ { "id", std::stoi(img.id) },
									{ "filename", img_filename },
									{ "width", img.image.cols },
									{ "height", img.image.rows } });

			for (const auto& ann : img.annotations)
			{
				const std::string ann_id = "a" + std::to_string(ann_counter++);
				annotations_json[ann_id] = {
					{ "img_id", std::stoi(img.id) },
					{ "cat_id", nlohmann::json::array({ ann.cat_id }) },
					{ "a_bbox",
					  nlohmann::json::array({ ann.bbox[0], ann.bbox[1],
											  ann.bbox[2], ann.bbox[3] }) },
					{ "area", 100 },
				};
			}
		}

		const nlohmann::json payload = {
			{ "categories", fixture_categories() },
			{ "images", images_json },
			{ "annotations", annotations_json },
		};
		std::ofstream(root / "deepscores_test.json") << payload.dump(2);
	}
} // namespace

TEST(DeepScoresDataset, FirstSupportedCropCanBeLoadedWhenDataPresent)
{
	const std::filesystem::path root =
		std::filesystem::temp_directory_path() / "omr_cpp_dataset_fixture";
	std::filesystem::create_directories(root / "images");
	std::filesystem::create_directories(root / "segmentation");

	cv::Mat1b image = cv::Mat1b::ones(80, 80) * 255;
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 20; ++x)
			image(y, x) = 0;
	for (int y = 20; y < 50; ++y)
		for (int x = 20; x < 22; ++x)
			image(y, x) = 0;
	ASSERT_TRUE(cv::imwrite((root / "images" / "page.png").string(), image));

	cv::Mat1b semantic = cv::Mat1b::zeros(80, 80);
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 20; ++x)
			semantic(y, x) = 8;
	for (int y = 20; y < 50; ++y)
		for (int x = 20; x < 22; ++x)
			semantic(y, x) = 7;
	ASSERT_TRUE(cv::imwrite((root / "segmentation" / "page_seg.png").string(),
							semantic));

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
				{ "color", 11 } } } } },
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
				{ "area", 60 } } } } },
	};
	std::ofstream(root / "deepscores_test.json") << payload.dump(2);

	const omr::DeepScoresDataset dataset(root, "test");
	omr::CropDiagnosticsByImage diagnostics;
	const auto samples = dataset.iter_supported_crops(
		1, std::nullopt, std::nullopt, std::nullopt, &diagnostics);
	ASSERT_FALSE(samples.empty());
	const omr::CropSample& sample = samples[0];
	EXPECT_FALSE(sample.image_id.empty());
	EXPECT_FALSE(sample.annotation_ids.empty());
	EXPECT_GT(sample.staff_space, 0.0f);
	EXPECT_GT(sample.staff_line, 0.0f);
	EXPECT_TRUE(sample.original_mask.any());
	EXPECT_TRUE(sample.staffless_mask.any());
	ASSERT_TRUE(sample.truth_masks.count("notehead"));
	ASSERT_TRUE(sample.truth_masks.count("stem"));
	ASSERT_TRUE(sample.truth_masks.count("beam"));
	EXPECT_TRUE(
		sample.truth_masks.at("notehead").same_shape(sample.staffless_mask));
	EXPECT_TRUE(
		sample.truth_masks.at("stem").same_shape(sample.staffless_mask));
	EXPECT_TRUE(
		sample.truth_masks.at("beam").same_shape(sample.staffless_mask));
	EXPECT_GE(sample.origin.first, 0);
	EXPECT_GE(sample.origin.second, 0);
	const auto& page_diagnostics = diagnostics.at("42");
	EXPECT_EQ(page_diagnostics.candidate_target_annotations, 2);
	EXPECT_EQ(page_diagnostics.components_considered, 1);
	EXPECT_EQ(page_diagnostics.yielded_crops, 1);
	EXPECT_TRUE(page_diagnostics.balances());
}

TEST(CropDiagnostics, AccountsForEveryOutcome)
{
	omr::CropDiagnostics diagnostics;
	diagnostics.components_considered = 5;
	diagnostics.rejected_no_supported_notehead = 1;
	diagnostics.rejected_stacked_head = 1;
	diagnostics.rejected_semantic_union_empty = 1;
	diagnostics.rejected_no_ink_touched = 1;
	diagnostics.yielded_crops = 1;
	EXPECT_TRUE(diagnostics.balances());

	omr::CropDiagnostics second;
	second.candidate_target_annotations = 3;
	second.components_considered = 1;
	second.yielded_crops = 1;
	diagnostics += second;
	EXPECT_EQ(diagnostics.candidate_target_annotations, 3);
	EXPECT_EQ(diagnostics.components_considered, 6);
	EXPECT_EQ(diagnostics.yielded_crops, 2);
	EXPECT_TRUE(diagnostics.balances());
}

TEST(CropDiagnostics, ExercisesEveryRejectionBucket)
{
	const std::filesystem::path root =
		std::filesystem::temp_directory_path() / "omr_cpp_rejection_fixture";
	std::filesystem::remove_all(root);

	// image "1": notehead + far stem -> both yield (head_count==0 is not
	// a rejection in the frozen Python reference, so C++ must not reject
	// either)
	cv::Mat1b img1 = cv::Mat1b::ones(80, 80) * 255;
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 20; ++x)
			img1(y, x) = 0;
	for (int y = 20; y < 50; ++y)
		for (int x = 60; x < 62; ++x)
			img1(y, x) = 0;
	cv::Mat1b sem1 = cv::Mat1b::zeros(80, 80);
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 20; ++x)
			sem1(y, x) = 8;
	for (int y = 20; y < 50; ++y)
		for (int x = 60; x < 62; ++x)
			sem1(y, x) = 7;
	std::vector<AnnotationSpec> anns1 = {
		{ "8", { 10, 40, 20, 50 } }, // notehead (far from stem)
		{ "7", { 60, 20, 62, 50 } }, // stem-only component -> yields
	};

	// image "2": two overlapping noteheads -> rejected_stacked_head
	cv::Mat1b img2 = cv::Mat1b::ones(80, 80) * 255;
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 25; ++x)
			img2(y, x) = 0;
	cv::Mat1b sem2 = cv::Mat1b::zeros(80, 80);
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 25; ++x)
			sem2(y, x) = 8;
	std::vector<AnnotationSpec> anns2 = {
		{ "8", { 10, 40, 20, 50 } }, // center=15, width=10
		{ "8", { 11, 40, 21, 50 } }, // center=16, width=10
		// |15-16|=1 < 0.6*min(10,10)=6 -> stacked
	};

	// image "3": notehead but semantic has no matching color in bbox
	// -> rejected_semantic_union_empty
	cv::Mat1b img3 = cv::Mat1b::ones(80, 80) * 255;
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 20; ++x)
			img3(y, x) = 0;
	cv::Mat1b sem3 = cv::Mat1b::zeros(80, 80);
	// intentionally no color 8 in the bbox region
	std::vector<AnnotationSpec> anns3 = {
		{ "8", { 10, 40, 20, 50 } },
	};

	// image "4": notehead, semantic non-empty, but no ink at union
	// -> rejected_no_ink_touched
	cv::Mat1b img4 = cv::Mat1b::ones(80, 80) * 255;
	img4(5, 5) = 0; // tiny ink dot far from bbox so Otsu works
	cv::Mat1b sem4 = cv::Mat1b::zeros(80, 80);
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 20; ++x)
			sem4(y, x) =
				8; // semantic has notehead color, but ink is absent there
	std::vector<AnnotationSpec> anns4 = {
		{ "8", { 10, 40, 20, 50 } },
	};

	// image "5": valid notehead + stem -> yielded_crops
	cv::Mat1b img5 = cv::Mat1b::ones(80, 80) * 255;
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 20; ++x)
			img5(y, x) = 0;
	for (int y = 20; y < 50; ++y)
		for (int x = 20; x < 22; ++x)
			img5(y, x) = 0;
	cv::Mat1b sem5 = cv::Mat1b::zeros(80, 80);
	for (int y = 40; y < 50; ++y)
		for (int x = 10; x < 20; ++x)
			sem5(y, x) = 8;
	for (int y = 20; y < 50; ++y)
		for (int x = 20; x < 22; ++x)
			sem5(y, x) = 7;
	std::vector<AnnotationSpec> anns5 = {
		{ "8", { 10, 40, 20, 50 } }, // notehead
		{ "7", { 20, 20, 22, 50 } }, // stem
	};

	write_fixture_dataset(root,
						  {
							  { "1", img1, sem1, anns1 },
							  { "2", img2, sem2, anns2 },
							  { "3", img3, sem3, anns3 },
							  { "4", img4, sem4, anns4 },
							  { "5", img5, sem5, anns5 },
						  });

	const omr::DeepScoresDataset dataset(root, "test");

	omr::CropDiagnosticsByImage diag;
	const auto samples = dataset.iter_supported_crops(
		std::nullopt, std::nullopt, std::nullopt, std::nullopt, &diag);

	ASSERT_TRUE(diag.count("1"));
	EXPECT_EQ(diag.at("1").components_considered, 2);
	EXPECT_EQ(diag.at("1").rejected_no_supported_notehead, 0);
	EXPECT_EQ(diag.at("1").yielded_crops, 2);
	EXPECT_TRUE(diag.at("1").balances());

	ASSERT_TRUE(diag.count("2"));
	EXPECT_EQ(diag.at("2").components_considered, 1);
	EXPECT_EQ(diag.at("2").rejected_stacked_head, 1);
	EXPECT_TRUE(diag.at("2").balances());

	ASSERT_TRUE(diag.count("3"));
	EXPECT_EQ(diag.at("3").components_considered, 1);
	EXPECT_EQ(diag.at("3").rejected_semantic_union_empty, 1);
	EXPECT_TRUE(diag.at("3").balances());

	ASSERT_TRUE(diag.count("4"));
	EXPECT_EQ(diag.at("4").components_considered, 1);
	EXPECT_EQ(diag.at("4").rejected_no_ink_touched, 1);
	EXPECT_TRUE(diag.at("4").balances());

	ASSERT_TRUE(diag.count("5"));
	EXPECT_EQ(diag.at("5").components_considered, 1);
	EXPECT_EQ(diag.at("5").yielded_crops, 1);
	EXPECT_TRUE(diag.at("5").balances());

	EXPECT_EQ(samples.size(), 3u);

	// diagnostics must not change crop counts
	omr::CropDiagnosticsByImage diag_off;
	const auto samples_off = dataset.iter_supported_crops(
		std::nullopt, std::nullopt, std::nullopt, std::nullopt, nullptr);
	EXPECT_EQ(samples_off.size(), samples.size());
}

TEST(DeepScoresDataset, DiscoversCropsWithoutAnnotationsOrSegmentation)
{
	const auto root = std::filesystem::temp_directory_path()
		/ "omr_cpp_inference_crop_fixture";
	std::filesystem::remove_all(root);
	cv::Mat1b image = cv::Mat1b::ones(140, 120) * 255;
	for (int y : { 20, 30, 40, 50, 60 })
		for (int x = 0; x < image.cols; ++x)
			image(y, x) = 0;
	for (int y = 42; y < 50; ++y)
		for (int x = 20; x < 30; ++x)
			image(y, x) = 0;
	for (int y = 25; y < 50; ++y)
		image(y, 31) = 0;
	image(100, 105) = 0; // filtered noise
	for (int y = 115; y < 125; ++y)
		for (int x = 60; x < 90; ++x)
			image(y, x) = 0; // large title-like component outside staff band
	write_fixture_dataset(
		root,
		{
			{ "1", image, cv::Mat1b::zeros(image.size()), {} },
		});
	std::filesystem::remove(root / "segmentation" / "img_1_seg.png");

	const omr::DeepScoresDataset dataset(root, "test");
	const auto samples = dataset.discover_page_crops("1");
	ASSERT_EQ(samples.size(), 1u);
	EXPECT_TRUE(samples[0].truth_masks.empty());
	EXPECT_TRUE(samples[0].annotation_ids.empty());
	EXPECT_TRUE(samples[0].staffless_mask.any());
	EXPECT_GT(samples[0].staff_space, 0.0f);
	EXPECT_LT(samples[0].origin.second, 40);
	EXPECT_LT(samples[0].staffless_mask.cols(), 50);
}
