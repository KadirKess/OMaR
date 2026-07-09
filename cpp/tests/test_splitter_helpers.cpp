#include <gtest/gtest.h>

#include "core/mask.h"
#include "symbols/splitter.h"

namespace
{

int ink(const cv::Mat& m)
{
	return cv::countNonZero(m);
}

void add_head(omr::Mask& image, int x, int y, int rx = 7, int ry = 5)
{
	omr::Mask head = omr::ellipse_mask(image.size(), x, y, rx, ry);
	image = image | head;
}

void fill_rect(omr::Mask& image, int y0, int y1, int x0, int x1)
{
	for (int y = y0; y < y1; ++y)
		for (int x = x0; x < x1; ++x)
			image.set(y, x, 255);
}

int count_kind(const omr::PrimitiveSplit& split, omr::PrimitiveKind kind)
{
	return static_cast<int>(split.of_kind(kind).size());
}

} // namespace

TEST(LineFootprint, HorizontalLineHasFormulaDimensionsAndMiddleRowOnly)
{
	cv::Mat f = omr::line_footprint(9, 1, 0.0);
	ASSERT_EQ(f.rows, 5);
	ASSERT_EQ(f.cols, 9);
	EXPECT_EQ(f.type(), CV_8U);
	EXPECT_EQ(ink(f), 9);
	for (int x = 0; x < 9; ++x)
	{
		EXPECT_GT(f.at<std::uint8_t>(2, x), 0);
		EXPECT_EQ(f.at<std::uint8_t>(0, x), 0);
		EXPECT_EQ(f.at<std::uint8_t>(4, x), 0);
	}
}

TEST(LineFootprint, ThicknessGtOneGrowsFrameAndInk)
{
	cv::Mat thin = omr::line_footprint(9, 1, 0.0);
	cv::Mat thick = omr::line_footprint(9, 2, 0.0);
	ASSERT_EQ(thin.rows, 5);
	ASSERT_EQ(thick.rows, 7);
	ASSERT_EQ(thin.cols, 9);
	ASSERT_EQ(thick.cols, 9);
	EXPECT_GT(ink(thick), ink(thin));
	for (int x = 0; x < thick.cols; ++x)
		EXPECT_GT(thick.at<std::uint8_t>(thick.rows / 2, x), 0);
}

TEST(LineFootprint, SlantedLineMatchesExpectedSlope)
{
	cv::Mat f = omr::line_footprint(9, 1, 45.0);
	ASSERT_EQ(f.rows, 13);
	ASSERT_EQ(f.cols, 9);
	EXPECT_GT(f.at<std::uint8_t>(2, 0), 0);
	EXPECT_GT(f.at<std::uint8_t>(10, 8), 0);
	EXPECT_GT(f.at<std::uint8_t>(6, 4), 0);
	// cv::LINE_8 vs skimage Bresenham differ on anti-diagonal steps.
	EXPECT_GE(ink(f), 9);
	EXPECT_LE(ink(f), 13);
}

TEST(LineFootprint, NegativeAngleMatchesPositiveAbsSlope)
{
	cv::Mat pos = omr::line_footprint(9, 1, 30.0);
	cv::Mat neg = omr::line_footprint(9, 1, -30.0);
	ASSERT_EQ(pos.rows, neg.rows);
	ASSERT_EQ(pos.cols, neg.cols);
	EXPECT_EQ(ink(pos), ink(neg));
}

TEST(EllipseMask, CenterAndBoundaryExtremesFilledCornerOutside)
{
	omr::Mask m = omr::ellipse_mask(cv::Size(21, 21), 10, 10, 5, 3);
	ASSERT_EQ(m.rows(), 21);
	ASSERT_EQ(m.cols(), 21);
	EXPECT_GT(m.at(10, 10), 0);
	EXPECT_GT(m.at(10, 5), 0);
	EXPECT_GT(m.at(10, 15), 0);
	EXPECT_GT(m.at(7, 10), 0);
	EXPECT_GT(m.at(13, 10), 0);
	EXPECT_EQ(m.at(7, 5), 0);
	EXPECT_EQ(m.at(13, 15), 0);
}

TEST(EllipseMask, BboxMatchesRadii)
{
	omr::Mask m = omr::ellipse_mask(cv::Size(21, 21), 10, 10, 5, 3);
	auto b = m.bbox();
	ASSERT_TRUE(b.has_value());
	EXPECT_EQ(b->x0, 5);
	EXPECT_EQ(b->x1, 16); // exclusive lower-right
	EXPECT_EQ(b->y0, 7);
	EXPECT_EQ(b->y1, 14);
}

TEST(EllipseMask, IsSymmetricAboutBothAxes)
{
	omr::Mask m = omr::ellipse_mask(cv::Size(21, 21), 10, 10, 5, 3);
	for (int dy = -3; dy <= 3; ++dy)
		for (int dx = -5; dx <= 5; ++dx)
		{
			unsigned char c = m.at(10 + dy, 10 + dx);
			EXPECT_EQ(m.at(10 - dy, 10 - dx), c);
			EXPECT_EQ(m.at(10 - dy, 10 + dx), c);
		}
}

TEST(EllipseMask, ZeroRadiiYieldEmptyMask)
{
	omr::Mask m = omr::ellipse_mask(cv::Size(7, 7), 3, 3, 0, 0);
	EXPECT_EQ(m.count_nonzero(), 0);
}

TEST(MaskDistance, TrueWhenWithinRadiusFalseWhenOutside)
{
	omr::Mask a(11, 11), b(11, 11);
	a.set(5, 5, 255);
	b.set(5, 8, 255);
	EXPECT_FALSE(omr::mask_distance(a, b, 2));
	EXPECT_TRUE(omr::mask_distance(a, b, 3));
}

TEST(MaskDistance, OverlappingOrTouchingMasksReturnTrue)
{
	omr::Mask a(7, 7), b(7, 7);
	a.set(3, 3, 255);
	b.set(3, 3, 255);
	EXPECT_TRUE(omr::mask_distance(a, b, 0));
	EXPECT_TRUE(omr::mask_distance(a, b, 1));
}

TEST(MaskDistance, FarApartMasksReturnFalseForModerateRadius)
{
	omr::Mask a(20, 20), b(20, 20);
	a.set(2, 2, 255);
	b.set(18, 18, 255);
	EXPECT_FALSE(omr::mask_distance(a, b, 5));
	EXPECT_FALSE(omr::mask_distance(a, b, 11));
}

TEST(SplitterConfig, DefaultsMatchPythonFrozenDataclass)
{
	omr::SplitterConfig c;
	EXPECT_DOUBLE_EQ(c.beam_min_length_spaces, 2.2);
	EXPECT_DOUBLE_EQ(c.beam_max_angle_degrees, 25.0);
	EXPECT_DOUBLE_EQ(c.stem_min_length_spaces, 1.6);
	EXPECT_DOUBLE_EQ(c.head_width_spaces, 1.4);
	EXPECT_DOUBLE_EQ(c.head_height_spaces, 1.05);
	EXPECT_DOUBLE_EQ(c.head_threshold, 0.56);
	EXPECT_DOUBLE_EQ(c.filled_head_min_occupancy, 0.68);
	EXPECT_DOUBLE_EQ(c.stem_threshold, 0.55);
	EXPECT_DOUBLE_EQ(c.beam_threshold, 0.55);
	EXPECT_DOUBLE_EQ(c.unknown_min_area_spaces, 0.12);
}

TEST(PrimitiveSplitter, DefaultConstructedUsesDefaultConfig)
{
	omr::PrimitiveSplitter s;
	EXPECT_DOUBLE_EQ(s.config().beam_threshold, 0.55);
	EXPECT_DOUBLE_EQ(s.config().head_threshold, 0.56);
}

TEST(PrimitiveSplitter, CustomConfigPassedThrough)
{
	omr::SplitterConfig c;
	c.beam_threshold = 0.9;
	c.head_threshold = 0.4;
	omr::PrimitiveSplitter s(c);
	EXPECT_DOUBLE_EQ(s.config().beam_threshold, 0.9);
	EXPECT_DOUBLE_EQ(s.config().head_threshold, 0.4);
}

TEST(PrimitiveSplitter, UpstemQuarterNote)
{
	omr::Mask image(60, 50);
	add_head(image, 20, 40);
	fill_rect(image, 12, 41, 26, 28);

	const omr::PrimitiveSplitter splitter;
	const omr::PrimitiveSplit result = splitter.split(image, 10.0f, 2.0f);

	EXPECT_EQ(result.status, omr::SplitStatus::ACCEPTED);
	EXPECT_EQ(count_kind(result, omr::PrimitiveKind::NOTEHEAD), 1);
	EXPECT_EQ(count_kind(result, omr::PrimitiveKind::STEM), 1);
	EXPECT_EQ(count_kind(result, omr::PrimitiveKind::BEAM), 0);
	ASSERT_EQ(result.head_to_stem.size(), 1);
	EXPECT_EQ(result.head_to_stem[0], std::make_pair(1, 0));
}

TEST(PrimitiveSplitter, SingleBeamGroup)
{
	omr::Mask image(70, 90);
	for (const auto& [x, y] :
		 std::vector<std::pair<int, int>>{ { 20, 45 }, { 45, 48 }, { 70, 42 } })
	{
		add_head(image, x, y);
		fill_rect(image, 12, y + 1, x + 6, x + 8);
	}
	fill_rect(image, 10, 15, 26, 78);

	const omr::PrimitiveSplitter splitter;
	const omr::PrimitiveSplit result = splitter.split(image, 10.0f, 2.0f);

	EXPECT_EQ(result.status, omr::SplitStatus::ACCEPTED);
	EXPECT_EQ(count_kind(result, omr::PrimitiveKind::NOTEHEAD), 3);
	EXPECT_EQ(count_kind(result, omr::PrimitiveKind::STEM), 3);
	EXPECT_EQ(count_kind(result, omr::PrimitiveKind::BEAM), 1);
	EXPECT_EQ(result.head_to_stem.size(), 3);
	EXPECT_EQ(result.stem_to_beam.size(), 3);
}

TEST(PrimitiveSplitter, UnexplainedRegionMakesResultUncertain)
{
	omr::Mask image(70, 70);
	add_head(image, 20, 45);
	fill_rect(image, 15, 46, 26, 28);
	fill_rect(image, 8, 20, 50, 62);

	const omr::PrimitiveSplitter splitter;
	const omr::PrimitiveSplit result = splitter.split(image, 10.0f, 2.0f);

	EXPECT_EQ(result.status, omr::SplitStatus::UNCERTAIN);
	EXPECT_EQ(count_kind(result, omr::PrimitiveKind::UNKNOWN), 1);
	EXPECT_NE(std::find(result.reasons.begin(), result.reasons.end(),
						"unexplained_ink"),
			  result.reasons.end());
}

TEST(PrimitiveSplitter, OpenHeadIsNotAcceptedAsSupported)
{
	omr::Mask image(60, 50);
	cv::ellipse(image.mat(), cv::Point(20, 40), cv::Size(7, 5), 0.0, 0.0,
				360.0, cv::Scalar(255), 1);
	fill_rect(image, 12, 41, 26, 28);

	const omr::PrimitiveSplitter splitter;
	const omr::PrimitiveSplit result = splitter.split(image, 10.0f, 2.0f);

	EXPECT_NE(result.status, omr::SplitStatus::ACCEPTED);
}

TEST(PrimitiveSplitter, InvalidDimensions)
{
	const omr::PrimitiveSplitter splitter;
	const omr::Mask image(3, 3);

	EXPECT_THROW(splitter.split(image, 0.0f, 1.0f), std::invalid_argument);
	EXPECT_THROW(splitter.split(image, 1.0f, 0.0f), std::invalid_argument);
}
