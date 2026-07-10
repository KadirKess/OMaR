#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "deepscores/dataset.h"
#include "models/graph.h"
#include "models/primitives.h"

namespace omr
{

	inline constexpr const char* SCOPED_CATEGORY_NAMES[] = {
		"noteheadBlackOnLine",
		"noteheadBlackInSpace",
		"stem",
		"beam",
	};

	struct Annotation
	{
		std::string ann_id;
		BBoxD bbox;
		std::string cat_name;
		int area = 0;
		double confidence = 1.0;
	};

	struct GraphEdgeSpec
	{
		std::string relation;
		std::string src_id;
		std::string dst_id;
	};

	struct PageIO
	{
		std::string image_id;
		std::string filename;
		int width = 0;
		int height = 0;
		std::vector<Annotation> annotations;
		std::vector<GraphEdgeSpec> graph_edges;
		std::vector<std::string> duration_sequence;
		std::vector<std::string> pitch_sequence;
	};

	struct PredictedCropResult
	{
		const PrimitiveSplit* split = nullptr;
		const NotationGraph* graph = nullptr;
		std::pair<int, int> origin{ 0, 0 };
	};

	std::string notehead_cat_name(const BBoxD& head_bbox,
								  const std::vector<double>& staff_lines,
								  double staff_line);

	PageIO build_predicted_page(
		const DatasetImage& image,
		const std::vector<PredictedCropResult>& crop_results,
		const std::vector<std::vector<double>>& staff_lines_per_crop = {},
		double staff_line = 2.0);

	PageIO
	derive_ground_truth(const DeepScoresDataset& dataset,
						const std::string& image_id,
						std::optional<double> staff_space = std::nullopt,
						std::function<bool(const BBoxD&)> bbox_filter = nullptr,
						const std::vector<double>& staff_lines = {});

	void write_predictions_json(
		const PageIO& page,
		const std::unordered_map<std::string, DatasetCategory>&
			source_categories,
		const std::filesystem::path& out_path);

} // namespace omr
