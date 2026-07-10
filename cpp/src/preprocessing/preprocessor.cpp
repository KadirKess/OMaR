#include "preprocessing/preprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace omr
{
	namespace
	{
		void add_run(std::unordered_map<int, int>& runs, int length)
		{
			if (length > 0)
				++runs[length];
		}

		int mode_run(const std::unordered_map<int, int>& runs)
		{
			if (runs.empty())
				return 0;
			auto best = runs.begin();
			for (auto it = runs.begin(); it != runs.end(); ++it)
			{
				if (it->second > best->second
					|| (it->second == best->second && it->first < best->first))
				{
					best = it;
				}
			}
			return best->first;
		}

	} // namespace

	Preprocessor::Preprocessor(cv::Mat1b image)
		: m_image(std::move(image))
	{
		if (m_image.empty())
			throw std::invalid_argument("Preprocessor image must not be empty");
	}

	StaffDimensions Preprocessor::compute_staff_space_height() const
	{
		std::unordered_map<int, int> black_runs;
		std::unordered_map<int, int> white_runs;

		for (int col = 0; col < m_image.cols; ++col)
		{
			int cur_black_run = 0;
			int cur_white_run = 0;
			for (int row = 0; row < m_image.rows; ++row)
			{
				if (m_image(row, col) == 0)
				{
					++cur_black_run;
					add_run(white_runs, cur_white_run);
					cur_white_run = 0;
				}
				else
				{
					++cur_white_run;
					add_run(black_runs, cur_black_run);
					cur_black_run = 0;
				}
			}
			add_run(black_runs, cur_black_run);
			add_run(white_runs, cur_white_run);
		}

		return StaffDimensions{ mode_run(black_runs), mode_run(white_runs) };
	}

	std::vector<int>
	Preprocessor::staffline_detection_projection(double frac) const
	{
		const StaffDimensions dims = compute_staff_space_height();
		const int min_distance = std::max(1, dims.staff_space_height / 2);
		const int threshold = static_cast<int>(
			std::ceil(frac * static_cast<double>(m_image.cols)));

		std::vector<int> profile(m_image.rows, 0);
		for (int row = 0; row < m_image.rows; ++row)
		{
			int count = 0;
			for (int col = 0; col < m_image.cols; ++col)
				if (m_image(row, col) == 0)
					++count;
			profile[row] = count;
		}

		std::vector<int> candidates;
		int row = 0;
		while (row < m_image.rows)
		{
			if (profile[row] < threshold)
			{
				++row;
				continue;
			}

			const int start = row;
			while (row + 1 < m_image.rows && profile[row + 1] == profile[start])
				++row;
			const int end = row;
			const int prev = start > 0 ? profile[start - 1] : -1;
			const int next = end + 1 < m_image.rows ? profile[end + 1] : -1;
			if (profile[start] > prev && profile[start] > next)
				candidates.push_back((start + end) / 2);
			++row;
		}

		std::stable_sort(candidates.begin(), candidates.end(),
						 [&profile](int a, int b) {
							 if (profile[a] != profile[b])
								 return profile[a] > profile[b];
							 return a < b;
						 });

		std::vector<int> selected;
		for (const int row : candidates)
		{
			const bool too_close =
				std::any_of(selected.begin(), selected.end(),
							[row, min_distance](int kept) {
								return std::abs(row - kept) < min_distance;
							});
			if (!too_close)
				selected.push_back(row);
		}
		std::sort(selected.begin(), selected.end());
		return selected;
	}

	const cv::Mat1b&
	Preprocessor::remove_stafflines(const std::optional<std::vector<int>>& rows,
									std::optional<int> max_thickness)
	{
		const std::vector<int> detected_rows =
			rows.has_value() ? *rows : staffline_detection_projection();
		if (!max_thickness)
			max_thickness = 2 * compute_staff_space_height().staff_line_height;

		for (const int raw_row : detected_rows)
		{
			if (raw_row < 0 || raw_row >= m_image.rows)
				continue;
			for (int col = 0; col < m_image.cols; ++col)
			{
				if (m_image(raw_row, col) != 0)
					continue;
				int top = raw_row;
				while (top - 1 >= 0 && m_image(top - 1, col) == 0)
					--top;
				int bottom = raw_row;
				while (bottom + 1 < m_image.rows
					   && m_image(bottom + 1, col) == 0)
					++bottom;
				if (bottom - top + 1 <= *max_thickness)
				{
					for (int row = top; row <= bottom; ++row)
						m_image(row, col) = 255;
				}
			}
		}
		return m_image;
	}

} // namespace omr
