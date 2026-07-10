#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "core/bbox.h"
#include "deepscores/dataset.h"
#include "deepscores/io.h"
#include "models/graph.h"
#include "models/primitives.h"
#include "symbols/splitter.h"

namespace omr::pipeline
{

	struct CropResult
	{
		PrimitiveSplit split;
		NotationGraph graph;
		std::vector<double> system_lines;
	};

	std::vector<double>
	recover_staff_rows(const std::filesystem::path& image_path,
					   double frac = 0.5);

	std::vector<double> system_for_crop(std::vector<double> rows,
										double crop_y_center);

	CropResult process_crop(const PrimitiveSplitter& splitter,
							const CropSample& sample,
							const std::vector<double>& staff_rows,
							double staff_line);

	std::function<bool(const BBoxD&)>
	make_crop_bbox_filter(const std::vector<CropSample>& samples);

	double compute_staff_line(const std::vector<CropSample>& samples);

} // namespace omr::pipeline
