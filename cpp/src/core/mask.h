#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <unordered_set>

#include "core/bbox.h"

namespace omr
{

	class Mask
	{
	public:
		Mask() = default;

		explicit Mask(cv::Mat1b mat)
			: m_mat(std::move(mat))
		{}

		Mask(int rows, int cols);

		static Mask like(const Mask& template_);

		int rows() const noexcept
		{
			return m_mat.rows;
		}
		int cols() const noexcept
		{
			return m_mat.cols;
		}
		cv::Size size() const noexcept
		{
			return m_mat.size();
		}
		bool empty() const noexcept
		{
			return m_mat.empty();
		}
		bool same_shape(const Mask& other) const noexcept
		{
			return m_mat.size() == other.m_mat.size();
		}

		bool any() const;
		int count_nonzero() const;
		std::optional<BBox> bbox() const;

		Mask clone() const;

		std::uint8_t at(int y, int x) const
		{
			return m_mat(y, x);
		}
		void set(int y, int x, std::uint8_t v)
		{
			m_mat(y, x) = v;
		}

		const cv::Mat1b& mat() const noexcept
		{
			return m_mat;
		}
		cv::Mat1b& mat() noexcept
		{
			return m_mat;
		}

		Mask dilate(int disk_radius) const;
		Mask open(const cv::Mat& kernel) const;
		Mask close(int disk_radius) const;

		Mask operator&(const Mask& other) const;
		Mask operator|(const Mask& other) const;
		Mask operator~() const;

		bool operator==(const Mask& other) const;

	private:
		cv::Mat1b m_mat;
	};

	cv::Mat disk_kernel(int radius);

	Mask is_in(const cv::Mat& labels, const std::unordered_set<int>& needles);

} // namespace omr
