#include "symbols/evaluation.h"

#include <array>
#include <stdexcept>

namespace omr
{

	MaskScores score_mask(const Mask& predicted, const Mask& truth)
	{
		if (!predicted.same_shape(truth))
			throw std::invalid_argument(
				"predicted and truth masks must have the same shape");

		const int tp = (predicted & truth).count_nonzero();
		const int fp = predicted.count_nonzero() - tp;
		const int fn = truth.count_nonzero() - tp;

		const double precision = tp + fp > 0
			? static_cast<double>(tp) / static_cast<double>(tp + fp)
			: static_cast<double>(!truth.any());
		const double recall = tp + fn > 0
			? static_cast<double>(tp) / static_cast<double>(tp + fn)
			: static_cast<double>(!predicted.any());
		const double f1 = precision + recall > 0.0
			? 2.0 * precision * recall / (precision + recall)
			: 0.0;
		const int union_count = tp + fp + fn;
		const double iou = union_count > 0
			? static_cast<double>(tp) / static_cast<double>(union_count)
			: 1.0;

		return { precision, recall, f1, iou };
	}

	std::map<std::string, MaskScores>
	evaluate_split(const PrimitiveSplit& result,
				   const std::map<std::string, Mask>& truth_masks,
				   const Mask* foreground_mask)
	{
		static constexpr std::array mapping{
			std::pair{ "notehead", PrimitiveKind::NOTEHEAD },
			std::pair{ "stem", PrimitiveKind::STEM },
			std::pair{ "beam", PrimitiveKind::BEAM },
		};

		std::map<std::string, MaskScores> scores;
		for (const auto& [name, kind] : mapping)
		{
			const auto truth = truth_masks.find(name);
			if (truth == truth_masks.end())
				throw std::invalid_argument("missing truth mask: "
											+ std::string(name));
			scores.emplace(name,
						   score_mask(result.class_mask(kind), truth->second));
		}

		if (foreground_mask != nullptr)
		{
			Mask known_truth = Mask::like(truth_masks.at("notehead"));
			for (const auto& [name, kind] : mapping)
			{
				(void)kind;
				if (!known_truth.same_shape(truth_masks.at(name)))
					throw std::invalid_argument(
						"truth masks must have the same shape");
				known_truth = known_truth | truth_masks.at(name);
			}
			if (!known_truth.same_shape(*foreground_mask))
				throw std::invalid_argument(
					"foreground and truth masks must have the same shape");

			const Mask unknown_truth = *foreground_mask & ~known_truth;
			scores.emplace("unknown",
						   score_mask(result.class_mask(PrimitiveKind::UNKNOWN),
									  unknown_truth));
		}

		return scores;
	}

} // namespace omr
