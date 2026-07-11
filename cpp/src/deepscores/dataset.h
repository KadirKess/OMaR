#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/mask.h"

namespace omr
{

	struct BBoxD
	{
		double x0 = 0.0;
		double y0 = 0.0;
		double x1 = 0.0;
		double y1 = 0.0;
	};

	struct CropSample
	{
		Mask original_mask;
		Mask staffless_mask;
		std::unordered_map<std::string, Mask> truth_masks;
		float staff_space = 0.0f;
		float staff_line = 0.0f;
		std::string image_id;
		std::vector<std::string> annotation_ids;
		std::pair<int, int> origin{ 0, 0 };
	};

	struct CropDiagnostics
	{
		int candidate_target_annotations = 0;
		int components_considered = 0;
		int rejected_no_supported_notehead = 0;
		int rejected_stacked_head = 0;
		int rejected_semantic_union_empty = 0;
		int rejected_no_ink_touched = 0;
		int yielded_crops = 0;
		int images_without_noteheads = 0;

		bool balances() const noexcept;
		CropDiagnostics& operator+=(const CropDiagnostics& other) noexcept;
	};

	using CropDiagnosticsByImage =
		std::unordered_map<std::string, CropDiagnostics>;

	struct DatasetCategory
	{
		std::string id;
		std::string name;
		std::string annotation_set;
		int color = 0;
		nlohmann::json raw;
	};

	struct DatasetImage
	{
		std::string id;
		std::string filename;
		int width = 0;
		int height = 0;
		nlohmann::json raw;
	};

	struct DatasetAnnotation
	{
		std::string id;
		std::string img_id;
		std::vector<std::string> cat_id;
		BBoxD a_bbox;
		int area = 0;
		nlohmann::json raw;
	};

	class DeepScoresDataset
	{
	public:
		DeepScoresDataset(std::filesystem::path root,
						  const std::string& split = "train");

		const std::filesystem::path& root() const noexcept
		{
			return m_root;
		}

		std::tuple<Mask, Mask, DatasetImage>
		load_page(const std::string& image_id) const;

		// Inference-only crop discovery. This path reads page pixels only.
		std::vector<CropSample>
		discover_page_crops(const std::string& image_id,
							double link_scale = 0.2,
							double margin_scale = 0.6) const;

		std::vector<CropSample> iter_supported_crops(
			std::optional<int> limit = std::nullopt,
			std::optional<unsigned int> shuffle_seed = std::nullopt,
			std::optional<int> max_pages = std::nullopt,
			std::optional<std::unordered_set<std::string>> image_id_filter =
				std::nullopt,
			CropDiagnosticsByImage* diagnostics = nullptr) const;

		cv::Mat semantic_mask(const DatasetImage& image) const;
		Mask ink_mask(const DatasetImage& image) const;

		std::unordered_map<std::string, DatasetCategory> categories;
		std::unordered_map<std::string, DatasetAnnotation> annotations;
		std::unordered_map<std::string, DatasetImage> images;
		std::unordered_map<std::string, std::string> category_ids;
		std::unordered_map<std::string, std::vector<std::string>> by_image;

	private:
		std::tuple<Mask, Mask, float, float>
		staffless_ink(const DatasetImage& image) const;
		Mask category_mask(const cv::Mat& semantic,
						   const DatasetAnnotation& annotation) const;

		std::filesystem::path m_root;
	};

	double bbox_distance(const BBoxD& a, const BBoxD& b);

} // namespace omr
