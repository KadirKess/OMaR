#include "reconstruction/assembly.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <tuple>

namespace omr
{
	namespace
	{
		NodeKind primitive_node_kind(PrimitiveKind kind)
		{
			switch (kind)
			{
			case PrimitiveKind::NOTEHEAD:
				return NodeKind::NOTEHEAD;
			case PrimitiveKind::STEM:
				return NodeKind::STEM;
			case PrimitiveKind::BEAM:
				return NodeKind::BEAM;
			case PrimitiveKind::UNKNOWN:
				return NodeKind::UNKNOWN;
			}
			return NodeKind::UNKNOWN;
		}

		std::pair<double, double> bbox_center(const BBox& bbox)
		{
			return { (bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0 };
		}

		bool x_overlaps(const BBox& a, const BBox& b)
		{
			return a.x0 < b.x1 && b.x0 < a.x1;
		}

		std::vector<NodePtr> heads_on_stem(const NotationGraph& graph,
										   const NodePtr& stem)
		{
			std::vector<NodePtr> heads;
			for (const auto& node :
				 graph.neighbors(stem->id, Relation::HEAD_TO_STEM))
			{
				if (node->kind == NodeKind::NOTEHEAD)
					heads.push_back(node);
			}
			return heads;
		}

		void absorb(NotationGraph& graph, const std::vector<int>& consumed_ids,
					int new_id)
		{
			const std::set<int> consumed(consumed_ids.begin(),
										 consumed_ids.end());
			std::set<std::tuple<int, int, Relation>> added;
			const std::vector<Edge> old_edges = graph.edges;
			for (const auto& edge : old_edges)
			{
				if (!consumed.count(edge.src) && !consumed.count(edge.dst))
					continue;
				const int src = consumed.count(edge.src) ? new_id : edge.src;
				const int dst = consumed.count(edge.dst) ? new_id : edge.dst;
				if (src == dst)
					continue;
				const auto key = std::make_tuple(src, dst, edge.relation);
				if (added.insert(key).second)
					graph.add_edge(src, dst, edge.relation);
			}
			for (const int node_id : consumed_ids)
				graph.remove_node(node_id);
		}
	} // namespace

	NotationGraph graph_from_split(const PrimitiveSplit& split)
	{
		NotationGraph graph;
		graph.staff_space = split.staff_space;
		graph.staff_lines.assign(split.staff_lines.begin(),
								 split.staff_lines.end());

		std::vector<int> node_ids;
		node_ids.reserve(split.primitives.size());
		for (const auto& primitive : split.primitives)
		{
			const auto node =
				graph.add_node(primitive_node_kind(primitive.kind), &primitive);
			node_ids.push_back(node->id);
		}

		for (const auto& [head_idx, stem_idx] : split.head_to_stem)
		{
			if (head_idx < 0 || stem_idx < 0
				|| head_idx >= static_cast<int>(node_ids.size())
				|| stem_idx >= static_cast<int>(node_ids.size()))
				throw std::out_of_range("head_to_stem index out of range");
			graph.add_edge(node_ids[head_idx], node_ids[stem_idx],
						   Relation::HEAD_TO_STEM);
		}

		for (const auto& [stem_idx, beam_idx] : split.stem_to_beam)
		{
			if (stem_idx < 0 || beam_idx < 0
				|| stem_idx >= static_cast<int>(node_ids.size())
				|| beam_idx >= static_cast<int>(node_ids.size()))
				throw std::out_of_range("stem_to_beam index out of range");
			graph.add_edge(node_ids[stem_idx], node_ids[beam_idx],
						   Relation::STEM_TO_BEAM);
		}

		return graph;
	}

	void compute_staff_positions(NotationGraph& graph)
	{
		if (graph.staff_lines.empty())
			return;

		const double ref_y = *std::max_element(graph.staff_lines.begin(),
											   graph.staff_lines.end());
		const double half_space = graph.staff_space / 2.0;
		if (half_space == 0.0)
			return;

		for (auto& [id, node] : graph.nodes)
		{
			(void)id;
			if (node->primitive == nullptr)
				continue;
			const auto [cx, cy] = bbox_center(node->primitive->bbox);
			(void)cx;
			node->attrs["staff_pos"] = (ref_y - cy) / half_space;
		}

		std::vector<NodePtr> ordered;
		ordered.reserve(graph.nodes.size());
		for (const auto& [id, node] : graph.nodes)
		{
			(void)id;
			if (node->primitive != nullptr)
				ordered.push_back(node);
		}
		std::sort(ordered.begin(), ordered.end(),
				  [](const NodePtr& a, const NodePtr& b) {
					  return a->primitive->bbox.x0 < b->primitive->bbox.x0;
				  });
		for (int i = 0; i < static_cast<int>(ordered.size()); ++i)
			ordered[i]->attrs["x_order"] = i;
	}

	void build_spatial_edges(NotationGraph& graph)
	{
		const double space = graph.staff_space;
		if (space <= 0.0)
			return;

		const double align_tol = 0.5 * space;
		const double above_gap = 0.5 * space;
		std::vector<NodePtr> nodes;
		for (const auto& [id, node] : graph.nodes)
		{
			(void)id;
			if (node->primitive != nullptr)
				nodes.push_back(node);
		}

		for (std::size_t i = 0; i < nodes.size(); ++i)
		{
			for (std::size_t j = i + 1; j < nodes.size(); ++j)
			{
				const auto& a = nodes[i];
				const auto& b = nodes[j];
				const BBox& a_box = a->primitive->bbox;
				const BBox& b_box = b->primitive->bbox;
				const auto [a_cx, a_cy] = bbox_center(a_box);
				const auto [b_cx, b_cy] = bbox_center(b_box);
				const bool same_column = x_overlaps(a_box, b_box);

				if (same_column || std::abs(a_cx - b_cx) <= align_tol)
					graph.add_edge(a->id, b->id, Relation::ALIGNED_VERTICAL);

				if (same_column)
				{
					if (b_cy - a_cy > above_gap)
						graph.add_edge(a->id, b->id, Relation::ABOVE);
					else if (a_cy - b_cy > above_gap)
						graph.add_edge(b->id, a->id, Relation::ABOVE);
				}
			}
		}
	}

	bool rule_note(NotationGraph& graph)
	{
		for (const auto& stem : graph.nodes_of_kind(NodeKind::STEM))
		{
			const auto heads = heads_on_stem(graph, stem);
			if (heads.size() != 1)
				continue;
			const auto& head = heads[0];

			const auto note = graph.add_node(NodeKind::NOTE);
			if (auto staff_pos = head->get<double>("staff_pos"))
				note->attrs["staff_pos"] = *staff_pos;
			if (auto x_order = head->get<int>("x_order"))
				note->attrs["x_order"] = *x_order;
			note->attrs["notehead"] = head->primitive;
			note->attrs["stem"] = stem->primitive;

			absorb(graph, { head->id, stem->id }, note->id);
			return true;
		}
		return false;
	}

	bool rule_chord(NotationGraph& graph)
	{
		for (const auto& stem : graph.nodes_of_kind(NodeKind::STEM))
		{
			auto heads = heads_on_stem(graph, stem);
			if (heads.size() < 2)
				continue;
			std::sort(heads.begin(), heads.end(),
					  [](const NodePtr& a, const NodePtr& b) {
						  return a->get_or<double>("staff_pos", 0.0)
							  < b->get_or<double>("staff_pos", 0.0);
					  });

			std::vector<const Primitive*> noteheads;
			std::vector<double> staff_positions;
			std::vector<int> consumed;
			noteheads.reserve(heads.size());
			staff_positions.reserve(heads.size());
			consumed.reserve(heads.size() + 1);
			std::optional<int> min_x_order;

			for (const auto& head : heads)
			{
				noteheads.push_back(head->primitive);
				staff_positions.push_back(
					head->get_or<double>("staff_pos", 0.0));
				if (auto x_order = head->get<int>("x_order"))
				{
					if (!min_x_order || *x_order < *min_x_order)
						min_x_order = *x_order;
				}
				consumed.push_back(head->id);
			}
			consumed.push_back(stem->id);

			const auto chord = graph.add_node(NodeKind::CHORD);
			chord->attrs["staff_positions"] = std::move(staff_positions);
			chord->attrs["noteheads"] = std::move(noteheads);
			chord->attrs["stem"] = stem->primitive;
			if (min_x_order)
				chord->attrs["x_order"] = *min_x_order;

			absorb(graph, consumed, chord->id);
			return true;
		}
		return false;
	}

	bool rule_beamed_group(NotationGraph& graph)
	{
		for (const auto& beam : graph.nodes_of_kind(NodeKind::BEAM))
		{
			std::vector<NodePtr> members;
			for (const auto& node :
				 graph.neighbors(beam->id, Relation::STEM_TO_BEAM))
			{
				if (node->kind == NodeKind::NOTE
					|| node->kind == NodeKind::CHORD)
					members.push_back(node);
			}
			if (members.size() < 2)
				continue;
			std::sort(members.begin(), members.end(),
					  [](const NodePtr& a, const NodePtr& b) {
						  return a->get_or<int>("x_order", 0)
							  < b->get_or<int>("x_order", 0);
					  });

			std::vector<int> consumed;
			consumed.reserve(members.size() + 1);
			for (const auto& member : members)
				consumed.push_back(member->id);
			consumed.push_back(beam->id);

			const auto group = graph.add_node(NodeKind::BEAMED_GROUP);
			group->attrs["members"] = members;
			group->attrs["beam"] = beam->primitive;
			if (auto x_order = members[0]->get<int>("x_order"))
				group->attrs["x_order"] = *x_order;

			absorb(graph, consumed, group->id);
			return true;
		}
		return false;
	}

	void run_rewrite(NotationGraph& graph)
	{
		const int max_iterations =
			std::max(1, static_cast<int>(graph.nodes.size()) * 10);
		for (int i = 0; i < max_iterations; ++i)
		{
			if (rule_chord(graph))
				continue;
			if (rule_note(graph))
				continue;
			if (rule_beamed_group(graph))
				continue;
			return;
		}
	}

} // namespace omr
