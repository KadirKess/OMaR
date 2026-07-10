#include <gtest/gtest.h>

#include "benchmarks/metrics.h"

namespace
{

	omr::Annotation ann(const std::string& ann_id, omr::BBoxD bbox,
						const std::string& cat_name, int area = 10,
						double confidence = 1.0)
	{
		return omr::Annotation{ ann_id, bbox, cat_name, area, confidence };
	}

} // namespace

TEST(IouBox, BasicCasesMatchPython)
{
	EXPECT_DOUBLE_EQ(omr::iou_box({ 0, 0, 4, 4 }, { 0, 0, 4, 4 }), 1.0);
	EXPECT_DOUBLE_EQ(omr::iou_box({ 0, 0, 2, 2 }, { 5, 5, 7, 7 }), 0.0);
	EXPECT_DOUBLE_EQ(omr::iou_box({ 0, 0, 4, 4 }, { 2, 2, 6, 6 }), 4.0 / 28.0);
	EXPECT_DOUBLE_EQ(omr::iou_box({ 0, 0, 10, 10 }, { 2, 2, 4, 4 }),
					 4.0 / 100.0);
	EXPECT_DOUBLE_EQ(omr::iou_box({ 0, 0, 0, 0 }, { 0, 0, 4, 4 }), 0.0);
}

TEST(MatchPairs, GreedyAndCrossClassBehavior)
{
	std::vector<omr::Annotation> preds{
		ann("p_best", { 0, 0, 10, 10 }, "stem"),
		ann("p_overlap", { 4, 4, 14, 14 }, "stem"),
	};
	std::vector<omr::Annotation> gts{ ann("g1", { 0, 0, 10, 10 }, "stem") };
	auto matches = omr::match_pairs(preds, gts);
	ASSERT_EQ(matches.size(), 1u);
	EXPECT_EQ(matches[0].pred_id, "p_best");

	preds = { ann("p1", { 0, 0, 10, 10 }, "noteheadBlackOnLine") };
	gts = { ann("g1", { 0, 0, 10, 10 }, "noteheadBlackInSpace") };
	matches = omr::match_pairs(preds, gts);
	ASSERT_EQ(matches.size(), 1u);
	EXPECT_NE(matches[0].pred_label, matches[0].gt_label);

	EXPECT_TRUE(omr::match_pairs({ ann("p1", { 0, 0, 2, 2 }, "stem") },
								 { ann("g1", { 10, 10, 12, 12 }, "stem") }, 0.5)
					.empty());
}

TEST(AveragePrecision, MatchesReferenceCases)
{
	std::vector<omr::Annotation> preds{
		ann("p1", { 0, 0, 10, 10 }, "stem", 10, 0.9),
		ann("p2", { 20, 0, 30, 10 }, "stem", 10, 0.8),
	};
	std::vector<omr::Annotation> gts{
		ann("g1", { 0, 0, 10, 10 }, "stem"),
		ann("g2", { 20, 0, 30, 10 }, "stem"),
	};
	EXPECT_DOUBLE_EQ(omr::average_precision(preds, gts, "stem").value(), 1.0);

	EXPECT_DOUBLE_EQ(omr::average_precision(
						 { ann("p1", { 0, 0, 10, 10 }, "stem", 10, 0.9) },
						 { ann("g1", { 100, 100, 110, 110 }, "stem") }, "stem")
						 .value(),
					 0.0);
	EXPECT_DOUBLE_EQ(omr::average_precision(
						 {}, { ann("g1", { 0, 0, 10, 10 }, "stem") }, "stem")
						 .value(),
					 0.0);
	EXPECT_FALSE(omr::average_precision(preds, {}, "stem").has_value());

	auto [mAP, per_class] =
		omr::mean_average_precision({ ann("p1", { 0, 0, 10, 10 }, "stem"),
									  ann("p2", { 0, 0, 10, 10 }, "beam") },
									{ ann("g1", { 0, 0, 10, 10 }, "stem") },
									{ "stem", "beam", "noteheadBlackOnLine" });
	EXPECT_TRUE(per_class.count("stem"));
	EXPECT_FALSE(per_class.count("beam"));
	EXPECT_DOUBLE_EQ(mAP, per_class["stem"]);
}

TEST(ConfusionMatrix, DiagonalMisclassificationAndEmpty)
{
	const std::vector<std::string> classes{ "noteheadBlackOnLine", "stem",
											"beam" };
	auto [matrix, accuracy] =
		omr::confusion_matrix({ { "p1", "g1", 0.9, "stem", "stem" },
								{ "p2", "g2", 0.9, "stem", "stem" },
								{ "p3", "g3", 0.9, "beam", "beam" } },
							  classes);
	EXPECT_DOUBLE_EQ(accuracy, 1.0);
	EXPECT_EQ(matrix[1][1], 2);
	EXPECT_EQ(matrix[2][2], 1);

	std::tie(matrix, accuracy) =
		omr::confusion_matrix({ { "p1", "g1", 0.9, "stem", "beam" },
								{ "p2", "g2", 0.9, "stem", "stem" } },
							  classes);
	EXPECT_DOUBLE_EQ(accuracy, 0.5);
	EXPECT_EQ(matrix[2][1], 1);
	EXPECT_EQ(matrix[1][1], 1);

	std::tie(matrix, accuracy) = omr::confusion_matrix({}, classes);
	EXPECT_DOUBLE_EQ(accuracy, 0.0);
}

TEST(EdgePrf1, TranslationAndUnmatchedEndpoints)
{
	std::vector<omr::PairMatch> matches{
		{ "p1", "g1", 0.9, "stem", "stem" },
		{ "p2", "g2", 0.9, "stem", "stem" },
	};
	auto [p, r, f1] =
		omr::edge_prf1({ { "head_to_stem", "p1", "p2" } },
					   { { "head_to_stem", "g1", "g2" } }, matches);
	EXPECT_DOUBLE_EQ(p, 1.0);
	EXPECT_DOUBLE_EQ(r, 1.0);
	EXPECT_DOUBLE_EQ(f1, 1.0);

	std::tie(p, r, f1) = omr::edge_prf1(
		{ { "head_to_stem", "p1", "p2" }, { "stem_to_beam", "p1", "p2" } },
		{ { "head_to_stem", "g1", "g2" } }, matches);
	EXPECT_DOUBLE_EQ(p, 0.5);

	std::tie(p, r, f1) =
		omr::edge_prf1({ { "head_to_stem", "p1", "p_unmatched" } }, {},
					   { { "p1", "g1", 0.9, "notehead", "notehead" } });
	EXPECT_DOUBLE_EQ(p, 0.0);
	EXPECT_DOUBLE_EQ(r, 0.0);
	EXPECT_DOUBLE_EQ(f1, 0.0);
}

TEST(SymbolErrorRate, ReferenceCases)
{
	EXPECT_DOUBLE_EQ(omr::symbol_error_rate({ "quarter", "eighth", "eighth" },
											{ "quarter", "eighth", "eighth" }),
					 0.0);
	EXPECT_DOUBLE_EQ(
		omr::symbol_error_rate({ "quarter", "eighth", "eighth", "eighth" },
							   { "quarter", "quarter", "eighth", "eighth" }),
		0.25);
	EXPECT_DOUBLE_EQ(omr::symbol_error_rate({ "quarter", "eighth", "eighth" },
											{ "quarter", "eighth" }),
					 0.5);
	EXPECT_DOUBLE_EQ(
		omr::symbol_error_rate({ "quarter" }, { "quarter", "eighth" }), 0.5);
	EXPECT_DOUBLE_EQ(omr::symbol_error_rate({ "quarter", "quarter" },
											{ "eighth", "eighth" }),
					 1.0);
	EXPECT_DOUBLE_EQ(omr::symbol_error_rate({ "quarter" }, {}), 1.0);
	EXPECT_DOUBLE_EQ(omr::symbol_error_rate({}, {}), 0.0);
}
