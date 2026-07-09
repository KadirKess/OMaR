#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "models/primitives.h"

namespace
{

	omr::Mask single_pixel_mask(int rows, int cols, int y, int x)
	{
		omr::Mask m(rows, cols);
		m.set(y, x, 255);
		return m;
	}

} // namespace

TEST(PrimitiveKinds, StringValuesMatchPythonStrEnum)
{
	EXPECT_STREQ(omr::to_string(omr::PrimitiveKind::NOTEHEAD), "notehead");
	EXPECT_STREQ(omr::to_string(omr::PrimitiveKind::STEM), "stem");
	EXPECT_STREQ(omr::to_string(omr::PrimitiveKind::BEAM), "beam");
	EXPECT_STREQ(omr::to_string(omr::PrimitiveKind::UNKNOWN), "unknown");
	EXPECT_STREQ(omr::to_string(omr::SplitStatus::ACCEPTED), "accepted");
	EXPECT_STREQ(omr::to_string(omr::SplitStatus::UNCERTAIN), "uncertain");
	EXPECT_STREQ(omr::to_string(omr::SplitStatus::REJECTED), "rejected");
}

TEST(Primitive, ConstructorComputesBboxAndClipsConfidence)
{
	// 6x6 mask with ink at (row=2,col=3) and (row=4,col=5).
	omr::Mask m(6, 6);
	m.set(2, 3, 255);
	m.set(4, 5, 255);
	omr::Primitive p(omr::PrimitiveKind::NOTEHEAD, std::move(m), 1.5f);
	EXPECT_EQ(p.kind, omr::PrimitiveKind::NOTEHEAD);
	EXPECT_FLOAT_EQ(p.confidence, 1.0f); // clipped
	EXPECT_EQ(p.bbox.x0, 3);
	EXPECT_EQ(p.bbox.y0, 2);
	EXPECT_EQ(p.bbox.x1, 6); // exclusive: 5 + 1
	EXPECT_EQ(p.bbox.y1, 5); // exclusive: 4 + 1
}

TEST(Primitive, NegativeConfidenceClampedToZero)
{
	omr::Primitive p(omr::PrimitiveKind::STEM, single_pixel_mask(4, 4, 1, 1),
					 -3.0f);
	EXPECT_FLOAT_EQ(p.confidence, 0.0f);
}

TEST(Primitive, EmptyMaskRejected)
{
	omr::Mask empty(4, 4);
	EXPECT_THROW(
		omr::Primitive(omr::PrimitiveKind::BEAM, std::move(empty), 1.0f),
		std::invalid_argument);
}

TEST(PrimitiveSplit, DefaultsMatchPythonDataclass)
{
	omr::PrimitiveSplit s;
	EXPECT_TRUE(s.primitives.empty());
	EXPECT_TRUE(s.head_to_stem.empty());
	EXPECT_TRUE(s.stem_to_beam.empty());
	EXPECT_TRUE(s.reasons.empty());
	EXPECT_EQ(s.status, omr::SplitStatus::REJECTED);
	EXPECT_FLOAT_EQ(s.staff_space, 0.0f);
	EXPECT_FLOAT_EQ(s.staff_line, 0.0f);
	EXPECT_TRUE(s.staff_lines.empty());
	EXPECT_TRUE(s.of_kind(omr::PrimitiveKind::NOTEHEAD).empty());
}

TEST(PrimitiveSplit, OfKindReturnsPointersInOrder)
{
	omr::PrimitiveSplit s;
	s.primitives.emplace_back(omr::PrimitiveKind::NOTEHEAD,
							  single_pixel_mask(4, 4, 0, 0), 1.0f);
	s.primitives.emplace_back(omr::PrimitiveKind::STEM,
							  single_pixel_mask(4, 4, 0, 1), 1.0f);
	s.primitives.emplace_back(omr::PrimitiveKind::NOTEHEAD,
							  single_pixel_mask(4, 4, 1, 0), 1.0f);
	auto heads = s.of_kind(omr::PrimitiveKind::NOTEHEAD);
	ASSERT_EQ(heads.size(), 2u);
	EXPECT_EQ(heads[0], &s.primitives[0]);
	EXPECT_EQ(heads[1], &s.primitives[2]);
}

TEST(PrimitiveSplit, ClassMaskUnionsSameKindAndMatchesShape)
{
	// Two noteheads at disjoint positions, one stem elsewhere.
	omr::Mask head_a(10, 10);
	head_a.set(1, 1, 255);
	head_a.set(1, 2, 255);
	omr::Mask head_b(10, 10);
	head_b.set(5, 6, 255);
	omr::Mask stem(10, 10);
	stem.set(2, 3, 255);

	omr::PrimitiveSplit s;
	s.primitives.emplace_back(omr::PrimitiveKind::NOTEHEAD, std::move(head_a),
							  0.9f);
	s.primitives.emplace_back(omr::PrimitiveKind::STEM, std::move(stem), 0.8f);
	s.primitives.emplace_back(omr::PrimitiveKind::NOTEHEAD, std::move(head_b),
							  0.7f);

	omr::Mask nh = s.class_mask(omr::PrimitiveKind::NOTEHEAD);
	EXPECT_EQ(nh.rows(), 10);
	EXPECT_EQ(nh.cols(), 10);
	EXPECT_EQ(nh.count_nonzero(), 3); // (1,1),(1,2),(5,6)
	auto b = nh.bbox();
	ASSERT_TRUE(b.has_value());
	EXPECT_EQ(b->x0, 1);
	EXPECT_EQ(b->x1, 7);
	EXPECT_EQ(b->y0, 1);
	EXPECT_EQ(b->y1, 6);

	// Empty kind returns a zero mask of the right shape, not a throw.
	omr::Mask beams = s.class_mask(omr::PrimitiveKind::BEAM);
	EXPECT_FALSE(beams.any());
	EXPECT_TRUE(beams.same_shape(nh));
}

TEST(PrimitiveSplit, ClassMaskThrowsWhenPrimitivesEmpty)
{
	omr::PrimitiveSplit s;
	EXPECT_THROW(s.class_mask(omr::PrimitiveKind::NOTEHEAD),
				 std::invalid_argument);
}
