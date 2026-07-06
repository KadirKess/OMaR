#include "core/mask.h"

#include <stdexcept>

namespace omr
{

	Mask::Mask(int rows, int cols)
		: m_mat(cv::Mat1b::zeros(rows, cols))
	{}

	Mask Mask::like(const Mask& template_)
	{
		return Mask(cv::Mat1b::zeros(template_.m_mat.size()));
	}

	Mask Mask::clone() const
	{
		return Mask(m_mat.clone());
	}

	bool Mask::any() const
	{
		if (m_mat.empty())
			return false;
		return cv::countNonZero(m_mat) > 0;
	}

	int Mask::count_nonzero() const
	{
		if (m_mat.empty())
			return 0;
		return cv::countNonZero(m_mat);
	}

	std::optional<BBox> Mask::bbox() const
	{
		if (m_mat.empty())
			return std::nullopt;
		std::vector<cv::Point> idx;
		cv::findNonZero(m_mat, idx);
		if (idx.empty())
			return std::nullopt;
		cv::Rect r = cv::boundingRect(idx);
		return BBox{ r.x, r.y, r.x + r.width,
					 r.y + r.height }; // exclusive lower-right
	}

	Mask Mask::dilate(int disk_radius) const
	{
		cv::Mat1b out;
		cv::dilate(m_mat, out, disk_kernel(disk_radius));
		return Mask(std::move(out));
	}

	Mask Mask::open(const cv::Mat& kernel) const
	{
		cv::Mat1b out;
		cv::morphologyEx(m_mat, out, cv::MORPH_OPEN, kernel);
		return Mask(std::move(out));
	}

	Mask Mask::close(int disk_radius) const
	{
		cv::Mat1b out;
		cv::morphologyEx(m_mat, out, cv::MORPH_CLOSE, disk_kernel(disk_radius));
		return Mask(std::move(out));
	}

	Mask Mask::operator&(const Mask& other) const
	{
		cv::Mat1b out;
		cv::bitwise_and(m_mat, other.m_mat, out);
		return Mask(std::move(out));
	}

	Mask Mask::operator|(const Mask& other) const
	{
		cv::Mat1b out;
		cv::bitwise_or(m_mat, other.m_mat, out);
		return Mask(std::move(out));
	}

	Mask Mask::operator~() const
	{
		cv::Mat1b out;
		cv::bitwise_not(m_mat, out);
		return Mask(std::move(out));
	}

	bool Mask::operator==(const Mask& other) const
	{
		if (m_mat.size() != other.m_mat.size())
			return false;
		if (m_mat.empty() && other.m_mat.empty())
			return true;
		return cv::countNonZero(m_mat != other.m_mat) == 0;
	}

	cv::Mat disk_kernel(int radius)
	{
		const int r = radius < 0 ? 0 : radius;
		return cv::getStructuringElement(cv::MORPH_ELLIPSE,
										 cv::Size(2 * r + 1, 2 * r + 1));
	}

	Mask is_in(const cv::Mat& labels, const std::unordered_set<int>& needles)
	{
		if (labels.type() != CV_32S && labels.type() != CV_16U)
		{
			throw std::invalid_argument("is_in: labels must be CV_32S or "
										"CV_16U (connectedComponents output)");
		}
		cv::Mat1b out = cv::Mat1b::zeros(labels.size());
		const int rows = labels.rows;
		const int cols = labels.cols;
		if (labels.type() == CV_32S)
		{
			for (int y = 0; y < rows; ++y)
			{
				const int* in_row = labels.ptr<int>(y);
				std::uint8_t* out_row = out.ptr<std::uint8_t>(y);
				for (int x = 0; x < cols; ++x)
					out_row[x] = needles.count(in_row[x]) ? 255 : 0;
			}
		}
		else
		{
			for (int y = 0; y < rows; ++y)
			{
				const std::uint16_t* in_row = labels.ptr<std::uint16_t>(y);
				std::uint8_t* out_row = out.ptr<std::uint8_t>(y);
				for (int x = 0; x < cols; ++x)
					out_row[x] = needles.count(in_row[x]) ? 255 : 0;
			}
		}
		return Mask(std::move(out));
	}

} // namespace omr
