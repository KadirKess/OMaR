#pragma once

#include <optional>

#include "core/mask.h"
#include "models/primitives.h"

namespace omr
{

	struct SplitterConfig
	{
		double beam_min_length_spaces = 2.2;
		double beam_max_angle_degrees = 25.0;
		double stem_min_length_spaces = 1.6;
		double head_width_spaces = 1.4;
		double head_height_spaces = 1.05;
		double head_threshold = 0.56;
		double filled_head_min_occupancy = 0.68;
		double stem_threshold = 0.55;
		double beam_threshold = 0.55;
		double unknown_min_area_spaces = 0.12;
	};

	cv::Mat line_footprint(int length, int thickness, double angle_degrees);

	Mask ellipse_mask(cv::Size shape, int cx, int cy, int rx, int ry);

	bool mask_distance(const Mask& a, const Mask& b, int radius);

	class PrimitiveSplitter
	{
	public:
		PrimitiveSplitter() = default;
		explicit PrimitiveSplitter(SplitterConfig config)
			: m_config(config)
		{}

		const SplitterConfig& config() const noexcept
		{
			return m_config;
		}

		PrimitiveSplit split(const Mask& staffless_mask, float staff_space,
							 float staff_line,
							 const Mask* original_mask = nullptr,
							 const std::vector<float>& staff_lines = {}) const;

	private:
		struct Candidate
		{
			Mask mask;
			float confidence = 0.0f;
		};

		std::optional<Candidate> detect_beam(const Mask& ink, float staff_space,
											 float staff_line) const;
		std::vector<Candidate> detect_stems(const Mask& ink,
											const Mask& beam_mask,
											float staff_space,
											float staff_line) const;
		std::optional<Candidate> best_head_for_stem(const Mask& ink,
													const Candidate& stem,
													const Mask& beam_mask,
													float staff_space,
													float staff_line) const;

		SplitterConfig m_config;
	};

} // namespace omr
