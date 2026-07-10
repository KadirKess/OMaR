#include "pipeline_common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "preprocessing/preprocessor.h"
#include "reconstruction/assembly.h"
#include "reconstruction/semantics.h"

namespace omr::pipeline
{

	std::vector<double>
	recover_staff_rows(const std::filesystem::path& image_path, double frac)
	{
		cv::Mat gray = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
		if (gray.empty())
			return {};
		cv::Mat1b ink;
		cv::threshold(gray, ink, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		Preprocessor pre(ink);
		const auto rows_int = pre.staffline_detection_projection(frac);
		std::vector<double> rows;
		rows.reserve(rows_int.size());
		for (int r : rows_int)
			rows.push_back(static_cast<double>(r));
		return rows;
	}

	std::vector<double> system_for_crop(std::vector<double> rows,
										double crop_y_center)
	{
		std::sort(rows.begin(), rows.end());
		if (rows.size() <= 5)
			return rows;
		std::vector<double> best_window;
		double best_dist = std::numeric_limits<double>::infinity();
		for (std::size_t i = 0; i + 4 < rows.size(); ++i)
		{
			const double midpoint = (rows[i] + rows[i + 4]) / 2.0;
			const double d = std::abs(midpoint - crop_y_center);
			if (d < best_dist)
			{
				best_dist = d;
				best_window.assign(rows.begin() + i, rows.begin() + i + 5);
			}
		}
		return best_window;
	}

	CropResult process_crop(const PrimitiveSplitter& splitter,
							const CropSample& sample,
							const std::vector<double>& staff_rows,
							double staff_line)
	{
		(void)staff_line;
		PrimitiveSplit split =
			splitter.split(sample.staffless_mask, sample.staff_space,
						   sample.staff_line, &sample.original_mask);

		const double crop_y_center = static_cast<double>(sample.origin.second)
			+ static_cast<double>(sample.staffless_mask.rows()) / 2.0;
		std::vector<double> system_lines =
			system_for_crop(staff_rows, crop_y_center);

		NotationGraph graph = graph_from_split(split);
		std::vector<double> graph_staff_lines;
		graph_staff_lines.reserve(system_lines.size());
		for (double y : system_lines)
			graph_staff_lines.push_back(y - sample.origin.second);
		graph.staff_lines = std::move(graph_staff_lines);
		graph.staff_space = static_cast<double>(sample.staff_space);

		compute_staff_positions(graph);
		run_rewrite(graph);
		assign_durations(graph);
		assign_pitches(graph);

		return CropResult{ std::move(split), std::move(graph),
						   std::move(system_lines) };
	}

	std::function<bool(const BBoxD&)>
	make_crop_bbox_filter(const std::vector<CropSample>& samples)
	{
		if (samples.empty())
			return [](const BBoxD&) { return false; };

		struct Bounds
		{
			double x0, y0, x1, y1;
		};
		std::vector<Bounds> bounds;
		bounds.reserve(samples.size());
		for (const auto& s : samples)
		{
			const int ox = s.origin.first;
			const int oy = s.origin.second;
			const int w = s.staffless_mask.cols();
			const int h = s.staffless_mask.rows();
			bounds.push_back({ static_cast<double>(ox), static_cast<double>(oy),
							   static_cast<double>(ox + w),
							   static_cast<double>(oy + h) });
		}
		return [bounds](const BBoxD& bbox) {
			const double cx = (bbox.x0 + bbox.x1) / 2.0;
			const double cy = (bbox.y0 + bbox.y1) / 2.0;
			for (const auto& b : bounds)
				if (b.x0 <= cx && cx <= b.x1 && b.y0 <= cy && cy <= b.y1)
					return true;
			return false;
		};
	}

	double compute_staff_line(const std::vector<CropSample>& samples)
	{
		if (samples.empty())
			return 2.0;
		std::vector<double> lines;
		lines.reserve(samples.size());
		for (const auto& s : samples)
			lines.push_back(static_cast<double>(s.staff_line));
		std::sort(lines.begin(), lines.end());
		const std::size_t mid = lines.size() / 2;
		double median_val;
		if (lines.size() % 2 == 1)
			median_val = lines[mid];
		else
			median_val = 0.5 * (lines[mid - 1] + lines[mid]);
		return std::max(1.0, std::round(0.15 * median_val));
	}

} // namespace omr::pipeline
