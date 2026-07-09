#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "core/mask.h"

namespace omr
{

	enum class PrimitiveKind
	{
		NOTEHEAD,
		STEM,
		BEAM,
		UNKNOWN,
	};

	enum class SplitStatus
	{
		ACCEPTED,
		UNCERTAIN,
		REJECTED,
	};

	inline const char* to_string(PrimitiveKind k)
	{
		switch (k)
		{
		case PrimitiveKind::NOTEHEAD:
			return "notehead";
		case PrimitiveKind::STEM:
			return "stem";
		case PrimitiveKind::BEAM:
			return "beam";
		case PrimitiveKind::UNKNOWN:
			return "unknown";
		}
		return "unknown";
	}

	inline const char* to_string(SplitStatus s)
	{
		switch (s)
		{
		case SplitStatus::ACCEPTED:
			return "accepted";
		case SplitStatus::UNCERTAIN:
			return "uncertain";
		case SplitStatus::REJECTED:
			return "rejected";
		}
		return "rejected";
	}

	struct Primitive
	{
		PrimitiveKind kind = PrimitiveKind::UNKNOWN;
		Mask mask;
		float confidence = 0.0f;
		BBox bbox;

		Primitive() = default;
		Primitive(PrimitiveKind kind, Mask mask, float confidence);

		bool operator==(const Primitive&) const = default;
	};

	struct PrimitiveSplit
	{
		std::vector<Primitive> primitives;
		std::vector<std::pair<int, int>> head_to_stem;
		std::vector<std::pair<int, int>> stem_to_beam;
		SplitStatus status = SplitStatus::REJECTED;
		std::vector<std::string> reasons;
		float staff_space = 0.0f;
		float staff_line = 0.0f;
		std::vector<float> staff_lines;

		std::vector<const Primitive*> of_kind(PrimitiveKind kind) const;
		Mask class_mask(PrimitiveKind kind) const;
	};

} // namespace omr
