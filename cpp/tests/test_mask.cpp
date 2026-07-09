#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <unordered_set>
#include <vector>

#include "core/mask.h"

namespace
{

	// Build a 0/255 mask from a 2D grid of 0/1 ints (row-major: outer = rows).
	omr::Mask from_grid(const std::vector<std::vector<int>>& rows)
	{
		const int h = static_cast<int>(rows.size());
		const int w = h > 0 ? static_cast<int>(rows[0].size()) : 0;
		cv::Mat1b m(h, w);
		for (int y = 0; y < h; ++y)
			for (int x = 0; x < w; ++x)
				m(y, x) = rows[y][x] ? 255 : 0;
		return omr::Mask(std::move(m));
	}

} // namespace

TEST(MaskCtor, DefaultIsEmptyAndQueriesReflectThat)
{
	omr::Mask m;
	EXPECT_TRUE(m.empty());
	EXPECT_FALSE(m.any());
	EXPECT_EQ(m.count_nonzero(), 0);
	EXPECT_FALSE(m.bbox().has_value());
}

TEST(MaskCtor, ZerosOfSizeHasRightShapeAndNoInk)
{
	omr::Mask m(5, 7);
	EXPECT_FALSE(m.empty());
	EXPECT_EQ(m.rows(), 5);
	EXPECT_EQ(m.cols(), 7);
	EXPECT_FALSE(m.any());
	EXPECT_EQ(m.count_nonzero(), 0);
}

TEST(MaskCtor, LikeProducesZerosOfSameShape)
{
	omr::Mask a = from_grid({ { 1, 0, 1 }, { 0, 1, 0 } });
	omr::Mask b = omr::Mask::like(a);
	EXPECT_TRUE(b.same_shape(a));
	EXPECT_FALSE(b.any());
}

TEST(MaskQueries, AnyCountAndBboxExclusiveLowerRight)
{
	omr::Mask m = from_grid({
		{ 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 1, 0, 0 },
		{ 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 0 },
	});
	// pixels at (y=1,x=3), (y=2,x=2), (y=2,x=3) -> 3 ink pixels
	EXPECT_TRUE(m.any());
	EXPECT_EQ(m.count_nonzero(), 3);
	auto b = m.bbox();
	ASSERT_TRUE(b.has_value());
	// x in {2,3} -> x0=2, x1=3+1=4 ; y in {1,2} -> y0=1, y1=2+1=3
	EXPECT_EQ(b->x0, 2);
	EXPECT_EQ(b->y0, 1);
	EXPECT_EQ(b->x1, 4); // exclusive
	EXPECT_EQ(b->y1, 3); // exclusive
	EXPECT_EQ(b->width(), 2);
	EXPECT_EQ(b->height(), 2);
}

TEST(MaskQueries, CloneIsIndependent)
{
	omr::Mask a = from_grid({ { 1, 0 }, { 0, 1 } });
	omr::Mask b = a.clone();
	EXPECT_TRUE(a == b);
	b.set(0, 1, 255);
	EXPECT_FALSE(a == b); // original untouched
	EXPECT_EQ(a.at(0, 1), 0);
	EXPECT_EQ(b.at(0, 1), 255);
}

TEST(MaskOps, DoubleNotIsIdentityProvingZero255Invariant)
{
	omr::Mask m = from_grid({ { 1, 0, 1 }, { 0, 1, 0 }, { 1, 0, 1 } });
	omr::Mask n = ~m;
	// ~255 == 0 and ~0 == 255 (would break with 0/1 storage).
	EXPECT_EQ(n.at(0, 0), 0);
	EXPECT_EQ(n.at(0, 1), 255);
	EXPECT_EQ(n.count_nonzero(), 4);
	EXPECT_TRUE(~n == m);
}

TEST(MaskOps, AndOrAndXorFreeEquivalent)
{
	omr::Mask a = from_grid({ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } });
	omr::Mask b = from_grid({ { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 } });
	EXPECT_EQ((a & b).count_nonzero(), 1);
	EXPECT_EQ((a | b).count_nonzero(), 5);
	// De Morgan: ~a | ~b == ~(a & b), sanity check the 0/255 algebra.
	EXPECT_TRUE((~a | ~b) == ~(a & b));
}

TEST(MaskOps, DilateGrowsSinglePixelByDiskKernel)
{
	omr::Mask m(11, 11);
	m.set(5, 5, 255);
	omr::Mask d = m.dilate(1);
	auto b = d.bbox();
	ASSERT_TRUE(b.has_value());
	EXPECT_EQ(b->x0, 4);
	EXPECT_EQ(b->y0, 4);
	EXPECT_EQ(b->x1, 7);
	EXPECT_EQ(b->y1, 7);
	// Disk(1) is the 3x3 ellipse structuring element; whatever its exact pixel
	// set, dilation stamps a copy of it, so ink count == kernel ink count.
	EXPECT_EQ(d.count_nonzero(), cv::countNonZero(omr::disk_kernel(1)));
}

TEST(MaskOps, OpenRemovesThinShapeAndPreservesBlock)
{
	cv::Mat rect3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	// Single pixel -> removed by opening.
	omr::Mask single(7, 7);
	single.set(3, 3, 255);
	EXPECT_FALSE(single.open(rect3).any());

	// 7x7 solid block in a 15x15 image -> preserved (erode->5x5, dilate->7x7).
	omr::Mask block(15, 15);
	for (int y = 4; y <= 10; ++y)
		for (int x = 4; x <= 10; ++x)
			block.set(y, x, 255);
	omr::Mask opened = block.open(rect3);
	EXPECT_TRUE(opened.any());
	EXPECT_EQ(opened.count_nonzero(), 49);
}

TEST(MaskOps, CloseFillsSinglePixelHole)
{
	omr::Mask m(15, 15);
	for (int y = 4; y <= 10; ++y)
		for (int x = 4; x <= 10; ++x)
			m.set(y, x, 255);
	m.set(7, 7, 0); // punch a hole
	EXPECT_EQ(m.count_nonzero(), 48);
	omr::Mask c = m.close(1);
	EXPECT_EQ(c.at(7, 7), 255);
	EXPECT_GE(c.count_nonzero(), 49);
}

TEST(MaskOps, DiskKernelHasRightSizeAndCenter)
{
	cv::Mat k = omr::disk_kernel(2);
	EXPECT_EQ(k.rows, 5);
	EXPECT_EQ(k.cols, 5);
	EXPECT_GT(k.at<unsigned char>(2, 2), 0);
}

TEST(MaskOps, DiskKernelZeroIsOneByOneToMatchSkimage)
{
	// skimage.morphology.disk(0) returns a 1x1 SE; dilation with it is
	// identity.
	cv::Mat k = omr::disk_kernel(0);
	EXPECT_EQ(k.rows, 1);
	EXPECT_EQ(k.cols, 1);
	EXPECT_GT(k.at<unsigned char>(0, 0), 0);
	omr::Mask m(5, 5);
	m.set(2, 2, 255);
	EXPECT_TRUE(m.dilate(0) == m);
}

TEST(IsIn, MatchesNumpyIsinForCV32S)
{
	const int rows = 4, cols = 4;
	cv::Mat labels(rows, cols, CV_32S);
	int vals[4][4] = {
		{ 0, 0, 0, 0 }, { 1, 1, 2, 2 }, { 3, 3, 0, 0 }, { 1, 2, 3, 0 }
	};
	for (int y = 0; y < rows; ++y)
		for (int x = 0; x < cols; ++x)
			labels.at<int>(y, x) = vals[y][x];
	std::unordered_set<int> needles{ 1, 3 };
	omr::Mask out = omr::is_in(labels, needles);
	EXPECT_EQ(out.rows(), rows);
	EXPECT_EQ(out.cols(), cols);
	EXPECT_EQ(out.at(0, 0), 0);
	EXPECT_EQ(out.at(1, 0), 255); // label 1
	EXPECT_EQ(out.at(1, 2), 0); // label 2
	EXPECT_EQ(out.at(2, 0), 255); // label 3
	EXPECT_EQ(out.at(3, 0), 255); // label 1
	EXPECT_EQ(out.at(3, 1), 0); // label 2
	EXPECT_EQ(out.at(3, 2), 255); // label 3
	EXPECT_EQ(out.count_nonzero(), 6);
}

TEST(IsIn, WorksForCV16U)
{
	cv::Mat labels(2, 2, CV_16U);
	labels.at<std::uint16_t>(0, 0) = 7;
	labels.at<std::uint16_t>(0, 1) = 8;
	labels.at<std::uint16_t>(1, 0) = 9;
	labels.at<std::uint16_t>(1, 1) = 7;
	std::unordered_set<int> needles{ 7, 9 };
	omr::Mask out = omr::is_in(labels, needles);
	EXPECT_EQ(out.at(0, 0), 255); // 7
	EXPECT_EQ(out.at(0, 1), 0); // 8
	EXPECT_EQ(out.at(1, 0), 255); // 9
	EXPECT_EQ(out.at(1, 1), 255); // 7
}

TEST(IsIn, RejectsBadLabelType)
{
	cv::Mat labels(2, 2, CV_8U);
	std::unordered_set<int> needles{ 1 };
	EXPECT_THROW(omr::is_in(labels, needles), std::invalid_argument);
}
