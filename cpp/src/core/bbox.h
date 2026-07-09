#pragma once

namespace omr
{

	struct BBox
	{
		int x0 = 0;
		int y0 = 0;
		int x1 = 0;
		int y1 = 0;

		int width() const noexcept
		{
			return x1 - x0;
		}
		int height() const noexcept
		{
			return y1 - y0;
		}
		bool empty() const noexcept
		{
			return width() <= 0 || height() <= 0;
		}

		bool operator==(const BBox&) const = default;
	};

} // namespace omr
