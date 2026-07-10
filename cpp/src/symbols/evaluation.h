#pragma once

#include <map>
#include <string>

#include "models/primitives.h"

namespace omr
{

	struct MaskScores
	{
		double precision = 0.0;
		double recall = 0.0;
		double f1 = 0.0;
		double iou = 0.0;

		bool operator==(const MaskScores&) const = default;
	};

	MaskScores score_mask(const Mask& predicted, const Mask& truth);

	std::map<std::string, MaskScores>
	evaluate_split(const PrimitiveSplit& result,
				   const std::map<std::string, Mask>& truth_masks,
				   const Mask* foreground_mask = nullptr);

} // namespace omr
