#pragma once

#include "models/graph.h"
#include "models/primitives.h"

namespace omr
{

	NotationGraph graph_from_split(const PrimitiveSplit& split);

	void compute_staff_positions(NotationGraph& graph);

	void build_spatial_edges(NotationGraph& graph);

	bool rule_note(NotationGraph& graph);
	bool rule_chord(NotationGraph& graph);
	bool rule_beamed_group(NotationGraph& graph);

	void run_rewrite(NotationGraph& graph);

} // namespace omr
