#include <gtest/gtest.h>

#include "symbols/evaluation.h"

namespace
{

	omr::Mask mask(std::initializer_list<std::initializer_list<int>> rows)
	{
		const int height = static_cast<int>(rows.size());
		const int width = static_cast<int>(rows.begin()->size());
		omr::Mask out(height, width);
		int y = 0;
		for (const auto& row : rows)
		{
			int x = 0;
			for (const int value : row)
			{
				if (value != 0)
					out.set(y, x, 255);
				++x;
			}
			++y;
		}
		return out;
	}

	omr::Primitive primitive(omr::PrimitiveKind kind, omr::Mask value)
	{
		return omr::Primitive(kind, std::move(value), 1.0f);
	}

} // namespace

TEST(ScoreMask, PartialOverlapMatchesPythonReference)
{
	const omr::Mask truth = mask({ { 1, 1 }, { 0, 0 } });
	const omr::Mask predicted = mask({ { 1, 0 }, { 1, 0 } });
	const omr::MaskScores score = omr::score_mask(predicted, truth);

	EXPECT_DOUBLE_EQ(score.precision, 0.5);
	EXPECT_DOUBLE_EQ(score.recall, 0.5);
	EXPECT_DOUBLE_EQ(score.f1, 0.5);
	EXPECT_DOUBLE_EQ(score.iou, 1.0 / 3.0);
}

TEST(ScoreMask, EmptyMasksMatchPythonEdgeCases)
{
	const omr::Mask empty(2, 2);
	const omr::Mask ink = mask({ { 1, 0 }, { 0, 0 } });

	EXPECT_EQ(omr::score_mask(empty, empty),
			  (omr::MaskScores{ 1.0, 1.0, 1.0, 1.0 }));
	EXPECT_EQ(omr::score_mask(ink, empty),
			  (omr::MaskScores{ 0.0, 0.0, 0.0, 0.0 }));
	EXPECT_EQ(omr::score_mask(empty, ink),
			  (omr::MaskScores{ 0.0, 0.0, 0.0, 0.0 }));
}

TEST(ScoreMask, RejectsMismatchedShapes)
{
	EXPECT_THROW(omr::score_mask(omr::Mask(2, 2), omr::Mask(3, 2)),
				 std::invalid_argument);
}

TEST(EvaluateSplit, ScoresSupportedAndUnknownMasks)
{
	omr::PrimitiveSplit split;
	split.primitives.push_back(primitive(omr::PrimitiveKind::NOTEHEAD,
										 mask({ { 1, 0, 0 }, { 0, 0, 0 } })));
	split.primitives.push_back(primitive(omr::PrimitiveKind::STEM,
										 mask({ { 0, 1, 0 }, { 0, 1, 0 } })));
	split.primitives.push_back(primitive(omr::PrimitiveKind::UNKNOWN,
										 mask({ { 0, 0, 0 }, { 0, 0, 1 } })));

	const std::map<std::string, omr::Mask> truth{
		{ "notehead", mask({ { 1, 0, 0 }, { 0, 0, 0 } }) },
		{ "stem", mask({ { 0, 1, 0 }, { 0, 1, 0 } }) },
		{ "beam", omr::Mask(2, 3) },
	};
	const omr::Mask foreground = mask({ { 1, 1, 0 }, { 0, 1, 1 } });

	const auto scores = omr::evaluate_split(split, truth, &foreground);
	ASSERT_EQ(scores.size(), 4u);
	EXPECT_DOUBLE_EQ(scores.at("notehead").f1, 1.0);
	EXPECT_DOUBLE_EQ(scores.at("stem").f1, 1.0);
	EXPECT_DOUBLE_EQ(scores.at("beam").f1, 1.0);
	EXPECT_DOUBLE_EQ(scores.at("unknown").f1, 1.0);
}

TEST(EvaluateSplit, RequiresEverySupportedTruthMask)
{
	omr::PrimitiveSplit split;
	split.primitives.push_back(
		primitive(omr::PrimitiveKind::NOTEHEAD, mask({ { 1, 0 }, { 0, 0 } })));

	const std::map<std::string, omr::Mask> incomplete{
		{ "notehead", mask({ { 1, 0 }, { 0, 0 } }) },
		{ "stem", omr::Mask(2, 2) },
	};
	EXPECT_THROW(omr::evaluate_split(split, incomplete), std::invalid_argument);
}
