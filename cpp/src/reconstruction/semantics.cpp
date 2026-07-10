#include "reconstruction/semantics.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "models/primitives.h"

namespace omr
{
	namespace
	{
		struct NoteLike
		{
			NodePtr node;
			bool is_beamed = false;
		};

		std::vector<NoteLike> iter_note_like(const NotationGraph& graph)
		{
			std::vector<NoteLike> out;
			for (const auto& [id, node] : graph.nodes)
			{
				(void)id;
				if (node->kind == NodeKind::NOTE
					|| node->kind == NodeKind::CHORD)
				{
					out.push_back({ node, false });
				}
				else if (node->kind == NodeKind::BEAMED_GROUP)
				{
					if (auto members =
							node->get<std::vector<NodePtr>>("members"))
					{
						for (const auto& member : *members)
							out.push_back({ member, true });
					}
				}
			}
			return out;
		}

		const Primitive* first_notehead(const NodePtr& node)
		{
			if (auto head = node->get<const Primitive*>("notehead"))
				return *head;
			if (auto heads =
					node->get<std::vector<const Primitive*>>("noteheads"))
			{
				if (!heads->empty())
					return (*heads)[0];
			}
			return nullptr;
		}

		bool is_filled(const Primitive* primitive, double threshold = 0.5)
		{
			if (primitive == nullptr)
				return false;
			const int area = primitive->bbox.width() * primitive->bbox.height();
			if (area <= 0)
				return false;
			return static_cast<double>(primitive->mask.count_nonzero()) / area
				>= threshold;
		}

		std::pair<char, int> clef_base(const std::string& clef)
		{
			if (clef == "treble")
				return { 'E', 4 };
			if (clef == "bass")
				return { 'G', 2 };
			if (clef == "alto")
				return { 'F', 3 };
			throw std::invalid_argument("unknown clef: " + clef);
		}

		std::string step_to_pitch(int step, char base_letter, int base_octave)
		{
			const std::string letters = "CDEFGAB";
			const int base = static_cast<int>(letters.find(base_letter));
			const int total = base + step;
			const int letter_index = ((total % 7) + 7) % 7;
			const int octave = base_octave
				+ static_cast<int>(std::floor(static_cast<double>(total)
											  / 7.0));
			return std::string(1, letters[letter_index])
				+ std::to_string(octave);
		}

		double event_duration(const NodePtr& node)
		{
			if (node->kind == NodeKind::BEAMED_GROUP)
			{
				double total = 0.0;
				if (auto members = node->get<std::vector<NodePtr>>("members"))
				{
					for (const auto& member : *members)
						total += member->get_or<double>("duration", 0.0);
				}
				return total;
			}
			return node->get_or<double>("duration", 0.0);
		}
	} // namespace

	void assign_durations(NotationGraph& graph)
	{
		for (const auto& [node, is_beamed] : iter_note_like(graph))
		{
			const Primitive* head = first_notehead(node);
			const auto stem = node->get<const Primitive*>("stem");

			double duration = QUARTER;
			if (!stem)
				duration = WHOLE;
			else if (head != nullptr && !is_filled(head))
				duration = HALF;
			else if (is_beamed)
				duration = EIGHTH;
			else
				duration = QUARTER;

			node->attrs["duration"] = duration;
		}
	}

	void assign_pitches(NotationGraph& graph, const std::string& clef)
	{
		const auto [base_letter, base_octave] = clef_base(clef);

		for (const auto& [node, is_beamed] : iter_note_like(graph))
		{
			(void)is_beamed;
			if (node->kind == NodeKind::CHORD)
			{
				std::vector<std::string> pitches;
				if (auto positions =
						node->get<std::vector<double>>("staff_positions"))
				{
					pitches.reserve(positions->size());
					for (const double pos : *positions)
					{
						pitches.push_back(
							step_to_pitch(static_cast<int>(std::lround(pos)),
										  base_letter, base_octave));
					}
				}
				node->attrs["pitches"] = std::move(pitches);
			}
			else if (auto staff_pos = node->get<double>("staff_pos"))
			{
				node->attrs["pitch"] =
					step_to_pitch(static_cast<int>(std::lround(*staff_pos)),
								  base_letter, base_octave);
			}
		}
	}

	MeasureCheck validate_measure_durations(const NotationGraph& graph,
											std::pair<int, int> time_signature,
											double tol)
	{
		const double expected =
			static_cast<double>(time_signature.first) / time_signature.second;
		double actual = 0.0;
		for (const auto& [id, node] : graph.nodes)
		{
			(void)id;
			if (node->kind == NodeKind::NOTE || node->kind == NodeKind::CHORD
				|| node->kind == NodeKind::BEAMED_GROUP)
				actual += event_duration(node);
		}

		return MeasureCheck{ expected, actual,
							 std::abs(actual - expected) <= tol };
	}

} // namespace omr
