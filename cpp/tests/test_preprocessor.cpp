#include <gtest/gtest.h>

#include "preprocessing/preprocessor.h"

namespace
{

	cv::Mat1b white_image(int rows, int cols)
	{
		return cv::Mat1b(rows, cols, std::uint8_t{ 255 });
	}

	void draw_staff(cv::Mat1b& image, const std::vector<int>& rows,
					int thickness = 1)
	{
		for (const int row : rows)
		{
			for (int dy = 0; dy < thickness; ++dy)
			{
				const int y = row + dy;
				if (y < 0 || y >= image.rows)
					continue;
				for (int x = 0; x < image.cols; ++x)
					image(y, x) = 0;
			}
		}
	}

	int count_ink(const cv::Mat1b& image)
	{
		int count = 0;
		for (int y = 0; y < image.rows; ++y)
			for (int x = 0; x < image.cols; ++x)
				if (image(y, x) == 0)
					++count;
		return count;
	}

} // namespace

TEST(PreprocessorStaffDimensions, FindsLineAndSpaceModes)
{
	cv::Mat1b image = white_image(64, 80);
	draw_staff(image, { 10, 20, 30, 40, 50 }, 2);

	const omr::Preprocessor preprocessor(image);
	const omr::StaffDimensions dims = preprocessor.compute_staff_space_height();

	EXPECT_EQ(dims.staff_line_height, 2);
	EXPECT_EQ(dims.staff_space_height, 8);
}

TEST(PreprocessorProjection, DetectsFiveFullWidthStaffRows)
{
	cv::Mat1b image = white_image(70, 90);
	const std::vector<int> staff_rows{ 12, 22, 32, 42, 52 };
	draw_staff(image, staff_rows);
	for (int y = 8; y < 18; ++y)
		for (int x = 70; x < 74; ++x)
			image(y, x) = 0;

	const omr::Preprocessor preprocessor(image);
	const std::vector<int> rows = preprocessor.staffline_detection_projection();

	EXPECT_EQ(rows, staff_rows);
}

TEST(PreprocessorProjection, KeepsOnePeakFromThickStaffLine)
{
	cv::Mat1b image = white_image(70, 90);
	draw_staff(image, { 12, 22, 32, 42, 52 }, 3);

	const omr::Preprocessor preprocessor(image);
	const std::vector<int> rows = preprocessor.staffline_detection_projection();

	ASSERT_EQ(rows.size(), 5);
	EXPECT_NEAR(rows[0], 13, 1);
	EXPECT_NEAR(rows[1], 23, 1);
	EXPECT_NEAR(rows[2], 33, 1);
	EXPECT_NEAR(rows[3], 43, 1);
	EXPECT_NEAR(rows[4], 53, 1);
}

TEST(PreprocessorRemoveStafflines, BlanksThinHorizontalLines)
{
	cv::Mat1b image = white_image(64, 80);
	const std::vector<int> staff_rows{ 10, 20, 30, 40, 50 };
	draw_staff(image, staff_rows);

	omr::Preprocessor preprocessor(image);
	const cv::Mat1b& out = preprocessor.remove_stafflines(staff_rows);

	EXPECT_EQ(count_ink(out), 0);
}

TEST(PreprocessorRemoveStafflines, PreservesVerticalStemsCrossingStaff)
{
	cv::Mat1b image = white_image(64, 80);
	const std::vector<int> staff_rows{ 10, 20, 30, 40, 50 };
	draw_staff(image, staff_rows);
	for (int y = 6; y < 46; ++y)
		for (int x = 30; x < 32; ++x)
			image(y, x) = 0;

	omr::Preprocessor preprocessor(image);
	const cv::Mat1b& out = preprocessor.remove_stafflines(staff_rows);

	for (int y = 6; y < 46; ++y)
	{
		EXPECT_EQ(out(y, 30), 0);
		EXPECT_EQ(out(y, 31), 0);
	}
	EXPECT_EQ(out(10, 0), 255);
	EXPECT_EQ(out(20, 79), 255);
}
