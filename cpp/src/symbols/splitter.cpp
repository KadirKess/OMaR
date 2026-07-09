#include "symbols/splitter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace omr
{
	namespace
	{
		struct Region
		{
			int label = 0;
			int x0 = 0;
			int y0 = 0;
			int x1 = 0;
			int y1 = 0;
			int area = 0;
		};

		std::vector<Region> connected_regions(const Mask& mask)
		{
			cv::Mat labels;
			cv::Mat stats;
			cv::Mat centroids;
			const int n = cv::connectedComponentsWithStats(
				mask.mat(), labels, stats, centroids, 8, CV_32S);
			std::vector<Region> regions;
			regions.reserve(std::max(0, n - 1));
			for (int label = 1; label < n; ++label)
			{
				const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
				const int y = stats.at<int>(label, cv::CC_STAT_TOP);
				const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
				const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
				regions.push_back(
					Region{ label, x, y, x + w, y + h,
							stats.at<int>(label, cv::CC_STAT_AREA) });
			}
			return regions;
		}

		std::vector<cv::Point> label_points(const Mask& mask, int label)
		{
			cv::Mat labels;
			cv::Mat stats;
			cv::Mat centroids;
			cv::connectedComponentsWithStats(mask.mat(), labels, stats,
											 centroids, 8, CV_32S);
			std::vector<cv::Point> points;
			for (int y = 0; y < labels.rows; ++y)
			{
				const int* row = labels.ptr<int>(y);
				for (int x = 0; x < labels.cols; ++x)
					if (row[x] == label)
						points.emplace_back(x, y);
			}
			return points;
		}

		bool fit_line_y_on_x(const std::vector<cv::Point>& points,
							 double& slope, double& intercept)
		{
			if (points.empty())
				return false;
			double sum_x = 0.0;
			double sum_y = 0.0;
			for (const auto& p : points)
			{
				sum_x += p.x;
				sum_y += p.y;
			}
			const double mean_x = sum_x / static_cast<double>(points.size());
			const double mean_y = sum_y / static_cast<double>(points.size());
			double numerator = 0.0;
			double denominator = 0.0;
			for (const auto& p : points)
			{
				const double dx = p.x - mean_x;
				numerator += dx * (p.y - mean_y);
				denominator += dx * dx;
			}
			if (denominator == 0.0)
				return false;
			slope = numerator / denominator;
			intercept = mean_y - slope * mean_x;
			return true;
		}

		double percentile(std::vector<double> values, double q)
		{
			if (values.empty())
				return 0.0;
			std::sort(values.begin(), values.end());
			const double pos = std::clamp(q, 0.0, 1.0)
				* static_cast<double>(values.size() - 1);
			const auto lo = static_cast<std::size_t>(std::floor(pos));
			const auto hi = static_cast<std::size_t>(std::ceil(pos));
			if (lo == hi)
				return values[lo];
			const double t = pos - static_cast<double>(lo);
			return values[lo] * (1.0 - t) + values[hi] * t;
		}

		double median(std::vector<int> values)
		{
			if (values.empty())
				return 0.0;
			std::sort(values.begin(), values.end());
			const std::size_t mid = values.size() / 2;
			if (values.size() % 2 == 1)
				return values[mid];
			return 0.5 * (values[mid - 1] + values[mid]);
		}

		Mask mask_and_not(const Mask& a, const Mask& b)
		{
			return a & ~b;
		}

		int count_rows_with_ink(const Mask& mask)
		{
			int count = 0;
			for (int y = 0; y < mask.rows(); ++y)
			{
				bool any = false;
				const std::uint8_t* row = mask.mat().ptr<std::uint8_t>(y);
				for (int x = 0; x < mask.cols(); ++x)
				{
					if (row[x])
					{
						any = true;
						break;
					}
				}
				if (any)
					++count;
			}
			return count;
		}

		int count_cols_with_ink(const Mask& mask)
		{
			int count = 0;
			for (int x = 0; x < mask.cols(); ++x)
			{
				bool any = false;
				for (int y = 0; y < mask.rows(); ++y)
				{
					if (mask.at(y, x))
					{
						any = true;
						break;
					}
				}
				if (any)
					++count;
			}
			return count;
		}

		std::vector<std::pair<int, int>> active_column_regions(const Mask& mask)
		{
			std::vector<std::pair<int, int>> regions;
			bool in_region = false;
			int x0 = 0;
			for (int x = 0; x < mask.cols(); ++x)
			{
				bool active = false;
				for (int y = 0; y < mask.rows(); ++y)
				{
					if (mask.at(y, x))
					{
						active = true;
						break;
					}
				}
				if (active && !in_region)
				{
					x0 = x;
					in_region = true;
				}
				else if (!active && in_region)
				{
					regions.emplace_back(x0, x);
					in_region = false;
				}
			}
			if (in_region)
				regions.emplace_back(x0, mask.cols());
			return regions;
		}

		Mask mask_from_label(const Mask& mask, int label)
		{
			cv::Mat labels;
			cv::Mat stats;
			cv::Mat centroids;
			cv::connectedComponentsWithStats(mask.mat(), labels, stats,
											 centroids, 8, CV_32S);
			cv::Mat1b out = cv::Mat1b::zeros(mask.size());
			for (int y = 0; y < labels.rows; ++y)
			{
				const int* in_row = labels.ptr<int>(y);
				std::uint8_t* out_row = out.ptr<std::uint8_t>(y);
				for (int x = 0; x < labels.cols; ++x)
					if (in_row[x] == label)
						out_row[x] = 255;
			}
			return Mask(std::move(out));
		}
	} // namespace

	cv::Mat line_footprint(int length, int thickness, double angle_degrees)
	{
		const int half = std::max(1, length / 2);
		const int dy = static_cast<int>(
			std::lround(std::tan(angle_degrees * M_PI / 180.0) * half));
		const int height = 2 * std::abs(dy) + 2 * thickness + 3;
		const int width = 2 * half + 1;

		cv::Mat foot = cv::Mat::zeros(height, width, CV_8U);
		const int cy = height / 2;
		cv::line(foot, cv::Point(0, cy - dy), cv::Point(width - 1, cy + dy),
				 255, 1, cv::LINE_8);
		if (thickness > 1)
		{
			cv::Mat out;
			cv::dilate(foot, out, disk_kernel(std::max(1, thickness / 2)));
			foot = out;
		}
		return foot;
	}

	Mask ellipse_mask(cv::Size shape, int cx, int cy, int rx, int ry)
	{
		cv::Mat1b m = cv::Mat1b::zeros(shape);
		if (rx <= 0 || ry <= 0)
			return Mask(std::move(m));
		const int x_lo = std::max(0, cx - rx);
		const int x_hi = std::min(shape.width - 1, cx + rx);
		const int y_lo = std::max(0, cy - ry);
		const int y_hi = std::min(shape.height - 1, cy + ry);
		const double rx2 = static_cast<double>(rx) * rx;
		const double ry2 = static_cast<double>(ry) * ry;
		for (int y = y_lo; y <= y_hi; ++y)
		{
			const double dy = static_cast<double>(y - cy);
			std::uint8_t* row = m.ptr<std::uint8_t>(y);
			for (int x = x_lo; x <= x_hi; ++x)
			{
				const double dx = static_cast<double>(x - cx);
				if ((dx * dx) / rx2 + (dy * dy) / ry2 <= 1.0)
					row[x] = 255;
			}
		}
		return Mask(std::move(m));
	}

	bool mask_distance(const Mask& a, const Mask& b, int radius)
	{
		return (a.dilate(radius) & b).any();
	}

	std::optional<PrimitiveSplitter::Candidate>
	PrimitiveSplitter::detect_beam(const Mask& ink, float staff_space,
								   float staff_line) const
	{
		const auto& cfg = m_config;
		const int length =
			std::max(5,
					 static_cast<int>(std::lround(cfg.beam_min_length_spaces
												  * staff_space)));
		const int thickness =
			std::max(1, static_cast<int>(std::lround(staff_line)));
		std::optional<Candidate> best;

		for (int i = 0; i < 11; ++i)
		{
			const double angle = -cfg.beam_max_angle_degrees
				+ (2.0 * cfg.beam_max_angle_degrees * i) / 10.0;
			const cv::Mat footprint = line_footprint(length, thickness, angle);
			const Mask opened = ink.open(footprint);
			for (const auto& region : connected_regions(opened))
			{
				if (region.x1 - region.x0 < length)
					continue;
				const auto coords = label_points(opened, region.label);
				double min_x = std::numeric_limits<double>::infinity();
				double max_x = -std::numeric_limits<double>::infinity();
				for (const auto& p : coords)
				{
					min_x = std::min(min_x, static_cast<double>(p.x));
					max_x = std::max(max_x, static_cast<double>(p.x));
				}
				if (max_x - min_x == 0.0)
					continue;

				double slope = 0.0;
				double intercept = 0.0;
				if (!fit_line_y_on_x(coords, slope, intercept))
					continue;
				const double detected_angle =
					std::abs(std::atan(slope) * 180.0 / M_PI);
				if (detected_angle > cfg.beam_max_angle_degrees + 2.0)
					continue;

				std::vector<double> residuals;
				residuals.reserve(coords.size());
				for (const auto& p : coords)
					residuals.push_back(std::abs(
						p.y - (slope * static_cast<double>(p.x) + intercept)));
				double band_half =
					std::max(1.0, percentile(residuals, 0.95) + 0.75);
				band_half =
					std::min(0.4 * static_cast<double>(staff_space), band_half);

				cv::Mat1b candidate_mat = cv::Mat1b::zeros(ink.size());
				for (int y = 0; y < ink.rows(); ++y)
				{
					const std::uint8_t* ink_row =
						ink.mat().ptr<std::uint8_t>(y);
					std::uint8_t* out_row = candidate_mat.ptr<std::uint8_t>(y);
					for (int x = region.x0; x < region.x1; ++x)
					{
						const double line_y =
							slope * static_cast<double>(x) + intercept;
						if (ink_row[x]
							&& std::abs(static_cast<double>(y) - line_y)
								<= band_half)
							out_row[x] = 255;
					}
				}
				Mask candidate_mask(std::move(candidate_mat));
				const int area = candidate_mask.count_nonzero();
				if (area == 0)
					continue;
				const double span_score =
					std::min(1.0,
							 (region.x1 - region.x0)
								 / (3.5 * static_cast<double>(staff_space)));
				const double fill_score = std::min(
					1.0,
					area
						/ std::max(1.0,
								   (region.x1 - region.x0)
									   * static_cast<double>(staff_line)
									   * 1.5));
				const float confidence =
					static_cast<float>(0.65 * span_score + 0.35 * fill_score);
				if (!best || confidence > best->confidence)
					best = Candidate{ candidate_mask, confidence };
			}
		}

		if (!best || best->confidence < cfg.beam_threshold)
			return std::nullopt;
		return best;
	}

	std::vector<PrimitiveSplitter::Candidate>
	PrimitiveSplitter::detect_stems(const Mask& ink, const Mask& beam_mask,
									float staff_space, float staff_line) const
	{
		const auto& cfg = m_config;
		const int beam_radius =
			std::max(1, static_cast<int>(std::lround(staff_line)));
		const Mask work = mask_and_not(ink, beam_mask.dilate(beam_radius));
		const int length =
			std::max(5,
					 static_cast<int>(std::lround(cfg.stem_min_length_spaces
												  * staff_space)));
		const cv::Mat footprint = cv::Mat::ones(length, 1, CV_8U);
		const Mask opened = work.open(footprint);
		std::vector<Candidate> candidates;

		for (const auto& [x0, x1] : active_column_regions(opened))
		{
			const int expand = static_cast<int>(std::lround(staff_line));
			const int expanded_x0 = std::max(0, x0 - expand);
			const int expanded_x1 = std::min(ink.cols(), x1 + expand);
			int y0 = ink.rows();
			int y1 = 0;
			for (int y = 0; y < opened.rows(); ++y)
			{
				for (int x = x0; x < x1; ++x)
				{
					if (opened.at(y, x))
					{
						y0 = std::min(y0, y);
						y1 = std::max(y1, y + 1);
					}
				}
			}
			if (y0 >= y1)
				continue;

			cv::Mat1b stem_mat = cv::Mat1b::zeros(ink.size());
			for (int y = y0; y < y1; ++y)
			{
				const std::uint8_t* work_row = work.mat().ptr<std::uint8_t>(y);
				std::uint8_t* out_row = stem_mat.ptr<std::uint8_t>(y);
				for (int x = expanded_x0; x < expanded_x1; ++x)
					out_row[x] = work_row[x];
			}
			Mask mask(std::move(stem_mat));
			const int height = y1 - y0;
			const int width = std::max(1, expanded_x1 - expanded_x0);
			const double length_score = std::min(
				1.0, height / (2.8 * static_cast<double>(staff_space)));
			const double narrow_score =
				std::min(1.0, (2.5 * static_cast<double>(staff_line)) / width);
			const float confidence =
				static_cast<float>(0.7 * length_score + 0.3 * narrow_score);
			if (confidence >= cfg.stem_threshold && mask.any())
				candidates.push_back(Candidate{ mask, confidence });
		}
		return candidates;
	}

	std::optional<PrimitiveSplitter::Candidate>
	PrimitiveSplitter::best_head_for_stem(const Mask& ink,
										  const Candidate& stem,
										  const Mask& beam_mask,
										  float staff_space,
										  float staff_line) const
	{
		std::vector<int> xs;
		int y_min = ink.rows();
		int y_max = -1;
		for (int y = 0; y < stem.mask.rows(); ++y)
		{
			const std::uint8_t* row = stem.mask.mat().ptr<std::uint8_t>(y);
			for (int x = 0; x < stem.mask.cols(); ++x)
			{
				if (row[x])
				{
					xs.push_back(x);
					y_min = std::min(y_min, y);
					y_max = std::max(y_max, y);
				}
			}
		}
		if (xs.empty())
			return std::nullopt;

		const double stem_x = median(xs);
		const int rx =
			std::max(2,
					 static_cast<int>(std::lround(m_config.head_width_spaces
												  * staff_space / 2.0)));
		const int ry =
			std::max(2,
					 static_cast<int>(std::lround(m_config.head_height_spaces
												  * staff_space / 2.0)));
		const Mask repaired = mask_and_not(ink, beam_mask).close(1);
		const int x_step = std::max(1, rx / 3);
		const int y_step = std::max(1, ry / 3);

		double best_score = -1.0;
		int best_cx = 0;
		int best_cy = 0;
		bool found = false;
		const int endpoints[2] = { y_min, y_max };
		for (int endpoint : endpoints)
		{
			for (int dx = -rx; dx <= rx; dx += x_step)
			{
				for (int dy = -ry; dy <= ry; dy += y_step)
				{
					const int cx = static_cast<int>(std::lround(stem_x + dx));
					const int cy = endpoint + dy;
					if (cx < 0 || cx >= ink.cols() || cy < 0
						|| cy >= ink.rows())
						continue;
					const Mask ellipse =
						ellipse_mask(ink.size(), cx, cy, rx, ry);
					const int inside = ellipse.count_nonzero();
					if (inside == 0)
						continue;
					const Mask local = repaired & ellipse;
					const double occupancy =
						static_cast<double>(local.count_nonzero()) / inside;
					if (occupancy < m_config.filled_head_min_occupancy)
						continue;
					const double row_spread =
						static_cast<double>(count_rows_with_ink(local))
						/ std::max(1, 2 * ry);
					const double col_spread =
						static_cast<double>(count_cols_with_ink(local))
						/ std::max(1, 2 * rx);
					const double score = 0.65 * occupancy
						+ 0.175 * std::min(1.0, row_spread)
						+ 0.175 * std::min(1.0, col_spread);
					if (!found || score > best_score)
					{
						found = true;
						best_score = score;
						best_cx = cx;
						best_cy = cy;
					}
				}
			}
		}

		if (!found || best_score < m_config.head_threshold)
			return std::nullopt;
		const int growth =
			std::max(1, static_cast<int>(std::lround(staff_line / 3.0f)));
		const Mask support = ellipse_mask(ink.size(), best_cx, best_cy,
										  rx + growth, ry + growth);
		Mask mask = ink & support;
		if (!mask.any())
			return std::nullopt;
		return Candidate{ mask, static_cast<float>(std::min(1.0, best_score)) };
	}

	PrimitiveSplit
	PrimitiveSplitter::split(const Mask& staffless_mask, float staff_space,
							 float staff_line, const Mask* original_mask,
							 const std::vector<float>& staff_lines) const
	{
		if (staffless_mask.empty())
			throw std::invalid_argument(
				"staffless_mask must be a two-dimensional image");
		if (original_mask != nullptr
			&& original_mask->size() != staffless_mask.size())
		{
			throw std::invalid_argument(
				"original_mask and staffless_mask must have the same shape");
		}
		if (staff_space <= 0.0f || staff_line <= 0.0f)
			throw std::invalid_argument("staff dimensions must be positive");

		const Mask empty(staffless_mask.rows(), staffless_mask.cols());
		const auto beam = detect_beam(staffless_mask, staff_space, staff_line);
		Mask beam_mask = beam ? beam->mask.clone() : empty.clone();
		const auto stems =
			detect_stems(staffless_mask, beam_mask, staff_space, staff_line);

		std::vector<std::pair<int, Candidate>> head_candidates;
		for (int stem_index = 0; stem_index < static_cast<int>(stems.size());
			 ++stem_index)
		{
			auto head = best_head_for_stem(staffless_mask, stems[stem_index],
										   beam_mask, staff_space, staff_line);
			if (head)
				head_candidates.emplace_back(stem_index, *head);
		}

		std::sort(head_candidates.begin(), head_candidates.end(),
				  [](const auto& a, const auto& b) {
					  return a.second.confidence > b.second.confidence;
				  });
		std::vector<std::pair<int, Candidate>> accepted_heads;
		Mask claimed(staffless_mask.rows(), staffless_mask.cols());
		for (const auto& item : head_candidates)
		{
			const Mask overlap_mask = claimed & item.second.mask;
			const double overlap =
				static_cast<double>(overlap_mask.count_nonzero())
				/ std::max(1, item.second.mask.count_nonzero());
			if (overlap < 0.5)
			{
				accepted_heads.push_back(item);
				claimed = claimed | item.second.mask;
			}
		}

		Mask head_union(staffless_mask.rows(), staffless_mask.cols());
		for (const auto& [stem_index, head] : accepted_heads)
		{
			(void)stem_index;
			head_union = head_union | head.mask;
		}
		beam_mask = mask_and_not(beam_mask, head_union);

		std::vector<Mask> stem_masks;
		stem_masks.reserve(stems.size());
		for (const auto& stem : stems)
			stem_masks.push_back(
				mask_and_not(mask_and_not(stem.mask, head_union), beam_mask));

		PrimitiveSplit result;
		result.staff_space = staff_space;
		result.staff_line = staff_line;
		result.staff_lines = staff_lines;

		std::vector<std::optional<int>> stem_primitive_indices;
		stem_primitive_indices.reserve(stems.size());
		std::unordered_set<int> associated_stems;
		for (const auto& [stem_index, head] : accepted_heads)
		{
			(void)head;
			associated_stems.insert(stem_index);
		}

		for (int i = 0; i < static_cast<int>(stems.size()); ++i)
		{
			if (associated_stems.count(i) && stem_masks[i].any())
			{
				stem_primitive_indices.push_back(
					static_cast<int>(result.primitives.size()));
				result.primitives.emplace_back(
					PrimitiveKind::STEM, stem_masks[i], stems[i].confidence);
			}
			else
			{
				stem_primitive_indices.push_back(std::nullopt);
			}
		}

		for (const auto& [stem_index, head] : accepted_heads)
		{
			const int head_index = static_cast<int>(result.primitives.size());
			result.primitives.emplace_back(PrimitiveKind::NOTEHEAD, head.mask,
										   head.confidence);
			const auto target_stem = stem_primitive_indices[stem_index];
			if (target_stem)
				result.head_to_stem.emplace_back(head_index, *target_stem);
		}

		std::optional<int> beam_index;
		if (beam && beam_mask.any())
		{
			beam_index = static_cast<int>(result.primitives.size());
			result.primitives.emplace_back(PrimitiveKind::BEAM, beam_mask,
										   beam->confidence);
		}

		if (beam_index)
		{
			const int radius =
				std::max(1,
						 static_cast<int>(std::lround(std::max(
							 static_cast<double>(staff_line) + 1.0,
							 0.35 * static_cast<double>(staff_space)))));
			for (const auto& stem_index : stem_primitive_indices)
			{
				if (stem_index
					&& mask_distance(result.primitives[*stem_index].mask,
									 beam_mask, radius))
					result.stem_to_beam.emplace_back(*stem_index, *beam_index);
			}
		}

		Mask assigned(staffless_mask.rows(), staffless_mask.cols());
		for (const auto& primitive : result.primitives)
			assigned = assigned | primitive.mask;
		const Mask residual = mask_and_not(staffless_mask, assigned);
		const int min_unknown_area = std::max(
			2,
			static_cast<int>(std::lround(m_config.unknown_min_area_spaces
										 * staff_space * staff_space)));
		int unknown_count = 0;
		for (const auto& region : connected_regions(residual))
		{
			if (region.area < min_unknown_area)
				continue;
			Mask unknown_mask = mask_from_label(residual, region.label);
			result.primitives.emplace_back(PrimitiveKind::UNKNOWN, unknown_mask,
										   1.0f);
			++unknown_count;
		}

		int head_count = 0;
		int stem_count = 0;
		int beam_count = 0;
		for (const auto& primitive : result.primitives)
		{
			if (primitive.kind == PrimitiveKind::NOTEHEAD)
				++head_count;
			else if (primitive.kind == PrimitiveKind::STEM)
				++stem_count;
			else if (primitive.kind == PrimitiveKind::BEAM)
				++beam_count;
		}

		bool topology_valid = head_count == stem_count
			&& head_count == static_cast<int>(result.head_to_stem.size())
			&& head_count > 0;
		if (beam_count)
		{
			topology_valid = topology_valid && beam_count == 1
				&& static_cast<int>(result.stem_to_beam.size()) == stem_count
				&& stem_count >= 2;
		}
		else
		{
			topology_valid = topology_valid && head_count == 1;
		}
		if (!topology_valid)
			result.reasons.push_back("unsupported_or_incomplete_topology");
		if (unknown_count)
			result.reasons.push_back("unexplained_ink");

		if (!topology_valid)
			result.status = SplitStatus::REJECTED;
		else if (unknown_count)
			result.status = SplitStatus::UNCERTAIN;
		else
			result.status = SplitStatus::ACCEPTED;

		return result;
	}

} // namespace omr
