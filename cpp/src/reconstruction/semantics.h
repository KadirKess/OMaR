#pragma once

#include <string>
#include <utility>

#include "models/graph.h"

namespace omr
{

	inline constexpr double WHOLE = 1.0;
	inline constexpr double HALF = 0.5;
	inline constexpr double QUARTER = 0.25;
	inline constexpr double EIGHTH = 0.125;

	struct MeasureCheck
	{
		double expected = 0.0;
		double actual = 0.0;
		bool is_valid = false;
	};

	void assign_durations(NotationGraph& graph);

	void assign_pitches(NotationGraph& graph,
						const std::string& clef = "treble");

	MeasureCheck
	validate_measure_durations(const NotationGraph& graph,
							   std::pair<int, int> time_signature = { 4, 4 },
							   double tol = 1e-6);

} // namespace omr
