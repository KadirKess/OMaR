#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "deepscores/dataset.h"

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
	const auto samples = dataset.iter_supported_crops(1);
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
}
