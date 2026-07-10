#include "models/graph.h"

#include <set>

namespace omr
{

	NodePtr NotationGraph::add_node(NodeKind kind, const Primitive* primitive)
	{
		const int node_id = next_node_id_++;
		auto node = std::make_shared<Node>();
		node->id = node_id;
		node->kind = kind;
		node->primitive = primitive;
		nodes.emplace(node_id, node);
		return node;
	}

	void NotationGraph::add_edge(int src, int dst, Relation relation)
	{
		edges.push_back(Edge{ src, dst, relation });
	}

	void NotationGraph::remove_node(int node_id)
	{
		auto it = nodes.find(node_id);
		if (it == nodes.end())
		{
			return;
		}
		nodes.erase(it);
		std::vector<Edge> kept;
		kept.reserve(edges.size());
		for (const Edge& e : edges)
		{
			if (e.src != node_id && e.dst != node_id)
			{
				kept.push_back(e);
			}
		}
		edges = std::move(kept);
	}

	std::vector<NodePtr> NotationGraph::nodes_of_kind(NodeKind kind) const
	{
		std::vector<NodePtr> result;
		for (const auto& [id, node] : nodes)
		{
			if (node->kind == kind)
			{
				result.push_back(node);
			}
		}
		return result;
	}

	static std::vector<NodePtr> collect_neighbors(const NotationGraph& graph,
												  int node_id,
												  const Relation* relation)
	{
		std::set<int> neighbor_ids;
		for (const Edge& e : graph.edges)
		{
			if (relation && e.relation != *relation)
			{
				continue;
			}
			if (e.src == node_id)
			{
				neighbor_ids.insert(e.dst);
			}
			else if (e.dst == node_id)
			{
				neighbor_ids.insert(e.src);
			}
		}
		std::vector<NodePtr> result;
		for (int nid : neighbor_ids)
		{
			auto it = graph.nodes.find(nid);
			if (it != graph.nodes.end())
			{
				result.push_back(it->second);
			}
		}
		return result;
	}

	std::vector<NodePtr> NotationGraph::neighbors(int node_id) const
	{
		return collect_neighbors(*this, node_id, nullptr);
	}

	std::vector<NodePtr> NotationGraph::neighbors(int node_id,
												  Relation relation) const
	{
		return collect_neighbors(*this, node_id, &relation);
	}

	std::vector<Edge> NotationGraph::edges_of(int node_id) const
	{
		std::vector<Edge> result;
		for (const Edge& e : edges)
		{
			if (e.src == node_id || e.dst == node_id)
			{
				result.push_back(e);
			}
		}
		return result;
	}

} // namespace omr
