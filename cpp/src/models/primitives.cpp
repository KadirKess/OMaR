#include "models/primitives.h"

#include <algorithm>
#include <stdexcept>

namespace omr
{

	Primitive::Primitive(PrimitiveKind kind, Mask mask, float confidence)
		: kind(kind)
		, mask(std::move(mask))
	{
		auto b = this->mask.bbox();
		if (!b)
			throw std::invalid_argument("a primitive mask cannot be empty");
		bbox = *b;
		this->confidence = std::clamp(confidence, 0.0f, 1.0f);
	}

	std::vector<const Primitive*>
	PrimitiveSplit::of_kind(PrimitiveKind kind) const
	{
		std::vector<const Primitive*> out;
		for (const auto& p : primitives)
			if (p.kind == kind)
				out.push_back(&p);
		return out;
	}

	Mask PrimitiveSplit::class_mask(PrimitiveKind kind) const
	{
		if (primitives.empty())
			throw std::invalid_argument(
				"cannot infer mask shape from an empty result");
		Mask result = Mask::like(primitives[0].mask);
		for (const auto& p : primitives)
			if (p.kind == kind)
				result = result | p.mask;
		return result;
	}

} // namespace omr
