#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace omr
{

	struct StaffDimensions
	{
		int staff_line_height = 0;
		int staff_space_height = 0;
	};

	class Preprocessor
	{
	public:
		explicit Preprocessor(cv::Mat1b image);

		const cv::Mat1b& image() const noexcept
		{
			return m_image;
		}

		cv::Mat1b& image() noexcept
		{
			return m_image;
		}

		StaffDimensions compute_staff_space_height() const;
		std::vector<int>
		staffline_detection_projection(double frac = 0.5) const;
		const cv::Mat1b& remove_stafflines(
			const std::optional<std::vector<int>>& rows = std::nullopt,
			std::optional<int> max_thickness = std::nullopt);

	private:
		cv::Mat1b m_image;
	};

} // namespace omr
