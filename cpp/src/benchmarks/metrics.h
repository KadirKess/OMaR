#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "deepscores/io.h"

namespace omr
{

	struct PairMatch
	{
		std::string pred_id;
		std::string gt_id;
		double iou = 0.0;
		std::string pred_label;
		std::string gt_label;
	};

	double iou_box(const BBoxD& a, const BBoxD& b);

	std::vector<PairMatch> match_pairs(const std::vector<Annotation>& pred_anns,
									   const std::vector<Annotation>& gt_anns,
									   double iou_threshold = 0.5);

	std::optional<double>
	average_precision(const std::vector<Annotation>& pred_anns,
					  const std::vector<Annotation>& gt_anns,
					  const std::string& class_name,
					  double iou_threshold = 0.5);

	std::pair<double, std::unordered_map<std::string, double>>
	mean_average_precision(const std::vector<Annotation>& pred_anns,
						   const std::vector<Annotation>& gt_anns,
						   const std::vector<std::string>& classes,
						   double iou_threshold = 0.5);

	std::pair<std::vector<std::vector<int>>, double>
	confusion_matrix(const std::vector<PairMatch>& matches,
					 const std::vector<std::string>& classes);

	std::tuple<double, double, double>
	edge_prf1(const std::vector<GraphEdgeSpec>& pred_edges,
			  const std::vector<GraphEdgeSpec>& gt_edges,
			  const std::vector<PairMatch>& matches);

	double symbol_error_rate(const std::vector<std::string>& hyp,
							 const std::vector<std::string>& ref);

} // namespace omr
