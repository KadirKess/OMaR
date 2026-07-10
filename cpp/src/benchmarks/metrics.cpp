#include "benchmarks/metrics.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_set>

namespace omr
{

	double iou_box(const BBoxD& a, const BBoxD& b)
	{
		const double ix0 = std::max(a.x0, b.x0);
		const double iy0 = std::max(a.y0, b.y0);
		const double ix1 = std::min(a.x1, b.x1);
		const double iy1 = std::min(a.y1, b.y1);
		const double iw = std::max(0.0, ix1 - ix0);
		const double ih = std::max(0.0, iy1 - iy0);
		const double inter = iw * ih;
		const double area_a = std::max(0.0, (a.x1 - a.x0) * (a.y1 - a.y0));
		const double area_b = std::max(0.0, (b.x1 - b.x0) * (b.y1 - b.y0));
		const double uni = area_a + area_b - inter;
		return uni > 0.0 ? inter / uni : 0.0;
	}

	std::vector<PairMatch> match_pairs(const std::vector<Annotation>& pred_anns,
									   const std::vector<Annotation>& gt_anns,
									   double iou_threshold)
	{
		struct Scored
		{
			double iou = 0.0;
			const Annotation* pred = nullptr;
			const Annotation* gt = nullptr;
		};
		std::vector<Scored> scored;
		for (const auto& pred : pred_anns)
		{
			for (const auto& gt : gt_anns)
			{
				const double iou = iou_box(pred.bbox, gt.bbox);
				if (iou >= iou_threshold)
					scored.push_back({ iou, &pred, &gt });
			}
		}
		std::sort(
			scored.begin(), scored.end(),
			[](const Scored& a, const Scored& b) { return a.iou > b.iou; });

		std::unordered_set<std::string> used_pred;
		std::unordered_set<std::string> used_gt;
		std::vector<PairMatch> matches;
		for (const auto& item : scored)
		{
			if (used_pred.count(item.pred->ann_id)
				|| used_gt.count(item.gt->ann_id))
				continue;
			used_pred.insert(item.pred->ann_id);
			used_gt.insert(item.gt->ann_id);
			matches.push_back(PairMatch{ item.pred->ann_id, item.gt->ann_id,
										 item.iou, item.pred->cat_name,
										 item.gt->cat_name });
		}
		return matches;
	}

	std::optional<double>
	average_precision(const std::vector<Annotation>& pred_anns,
					  const std::vector<Annotation>& gt_anns,
					  const std::string& class_name, double iou_threshold)
	{
		std::vector<const Annotation*> preds;
		std::vector<const Annotation*> gts;
		for (const auto& ann : pred_anns)
			if (ann.cat_name == class_name)
				preds.push_back(&ann);
		for (const auto& ann : gt_anns)
			if (ann.cat_name == class_name)
				gts.push_back(&ann);
		if (gts.empty())
			return std::nullopt;
		if (preds.empty())
			return 0.0;

		std::sort(preds.begin(), preds.end(),
				  [](const Annotation* a, const Annotation* b) {
					  return a->confidence > b->confidence;
				  });
		std::vector<double> tp(preds.size(), 0.0);
		std::vector<double> fp(preds.size(), 0.0);
		std::unordered_set<int> used_gt;
		for (int i = 0; i < static_cast<int>(preds.size()); ++i)
		{
			int best_j = -1;
			double best_iou = iou_threshold;
			for (int j = 0; j < static_cast<int>(gts.size()); ++j)
			{
				if (used_gt.count(j))
					continue;
				const double iou = iou_box(preds[i]->bbox, gts[j]->bbox);
				if (iou >= best_iou)
				{
					best_iou = iou;
					best_j = j;
				}
			}
			if (best_j >= 0)
			{
				tp[i] = 1.0;
				used_gt.insert(best_j);
			}
			else
			{
				fp[i] = 1.0;
			}
		}

		std::vector<double> recall;
		std::vector<double> precision;
		recall.reserve(preds.size());
		precision.reserve(preds.size());
		double tp_cum = 0.0;
		double fp_cum = 0.0;
		for (std::size_t i = 0; i < preds.size(); ++i)
		{
			tp_cum += tp[i];
			fp_cum += fp[i];
			recall.push_back(tp_cum / static_cast<double>(gts.size()));
			precision.push_back(tp_cum / std::max(tp_cum + fp_cum, 1e-12));
		}

		std::vector<double> rec{ 0.0 };
		rec.insert(rec.end(), recall.begin(), recall.end());
		rec.push_back(1.0);
		std::vector<double> pre{ 1.0 };
		pre.insert(pre.end(), precision.begin(), precision.end());
		pre.push_back(0.0);
		for (int i = static_cast<int>(pre.size()) - 2; i >= 0; --i)
			pre[i] = std::max(pre[i], pre[i + 1]);

		double ap = 0.0;
		for (std::size_t i = 0; i + 1 < rec.size(); ++i)
		{
			if (rec[i + 1] != rec[i])
				ap += (rec[i + 1] - rec[i]) * pre[i + 1];
		}
		return ap;
	}

	std::pair<double, std::unordered_map<std::string, double>>
	mean_average_precision(const std::vector<Annotation>& pred_anns,
						   const std::vector<Annotation>& gt_anns,
						   const std::vector<std::string>& classes,
						   double iou_threshold)
	{
		std::unordered_map<std::string, double> per_class;
		for (const auto& class_name : classes)
		{
			auto ap = average_precision(pred_anns, gt_anns, class_name,
										iou_threshold);
			if (ap)
				per_class[class_name] = *ap;
		}
		if (per_class.empty())
			return { 0.0, per_class };
		double total = 0.0;
		for (const auto& [name, ap] : per_class)
		{
			(void)name;
			total += ap;
		}
		return { total / static_cast<double>(per_class.size()), per_class };
	}

	std::pair<std::vector<std::vector<int>>, double>
	confusion_matrix(const std::vector<PairMatch>& matches,
					 const std::vector<std::string>& classes)
	{
		std::unordered_map<std::string, int> idx;
		for (int i = 0; i < static_cast<int>(classes.size()); ++i)
			idx[classes[i]] = i;
		std::vector<std::vector<int>> matrix(
			classes.size(), std::vector<int>(classes.size(), 0));
		int correct = 0;
		int counted = 0;
		for (const auto& match : matches)
		{
			if (idx.count(match.gt_label) && idx.count(match.pred_label))
			{
				++matrix[idx[match.gt_label]][idx[match.pred_label]];
				++counted;
				if (match.gt_label == match.pred_label)
					++correct;
			}
		}
		(void)counted;
		const double accuracy = matches.empty() ? 0.0
												: static_cast<double>(correct)
				/ static_cast<double>(matches.size());
		return { matrix, accuracy };
	}

	std::tuple<double, double, double>
	edge_prf1(const std::vector<GraphEdgeSpec>& pred_edges,
			  const std::vector<GraphEdgeSpec>& gt_edges,
			  const std::vector<PairMatch>& matches)
	{
		std::unordered_map<std::string, std::string> pred_to_gt;
		for (const auto& match : matches)
			pred_to_gt[match.pred_id] = match.gt_id;

		std::set<std::tuple<std::string, std::string, std::string>> translated;
		int fp_unmatched = 0;
		for (const auto& edge : pred_edges)
		{
			if (pred_to_gt.count(edge.src_id) && pred_to_gt.count(edge.dst_id))
			{
				translated.insert({ edge.relation, pred_to_gt[edge.src_id],
									pred_to_gt[edge.dst_id] });
			}
			else
			{
				++fp_unmatched;
			}
		}

		std::set<std::tuple<std::string, std::string, std::string>> gt_set;
		for (const auto& edge : gt_edges)
			gt_set.insert({ edge.relation, edge.src_id, edge.dst_id });

		int tp = 0;
		for (const auto& edge : translated)
			if (gt_set.count(edge))
				++tp;
		const int fp = static_cast<int>(translated.size()) - tp + fp_unmatched;
		const int fn = static_cast<int>(gt_set.size()) - tp;
		const double p = (tp + fp) ? static_cast<double>(tp) / (tp + fp) : 0.0;
		const double r = (tp + fn) ? static_cast<double>(tp) / (tp + fn) : 0.0;
		const double f1 = (p + r) ? 2.0 * p * r / (p + r) : 0.0;
		return { p, r, f1 };
	}

	double symbol_error_rate(const std::vector<std::string>& hyp,
							 const std::vector<std::string>& ref)
	{
		if (ref.empty())
			return hyp.empty() ? 0.0 : 1.0;
		const int n = static_cast<int>(ref.size());
		std::vector<int> prev(n + 1);
		std::iota(prev.begin(), prev.end(), 0);
		for (int i = 1; i <= static_cast<int>(hyp.size()); ++i)
		{
			std::vector<int> cur(n + 1, 0);
			cur[0] = i;
			for (int j = 1; j <= n; ++j)
			{
				const int cost = hyp[i - 1] == ref[j - 1] ? 0 : 1;
				cur[j] = std::min(
					{ prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost });
			}
			prev = std::move(cur);
		}
		return static_cast<double>(prev[n]) / std::max(1, n);
	}

} // namespace omr
