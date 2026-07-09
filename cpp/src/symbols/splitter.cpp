#include "symbols/splitter.h"

#include <cmath>

namespace omr
{

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

} // namespace omr
