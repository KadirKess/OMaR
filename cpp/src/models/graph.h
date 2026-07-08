#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace omr {

struct Primitive;

enum class Relation {
  HEAD_TO_STEM,
  STEM_TO_BEAM,
  ABOVE,
  ALIGNED_VERTICAL,
};

enum class NodeKind {
  NOTEHEAD,
  STEM,
  BEAM,
  UNKNOWN,
  NOTE,
  CHORD,
  BEAMED_GROUP,
  MEASURE,
};

struct Node;
using NodePtr = std::shared_ptr<Node>;

/// Heterogeneous attribute value. Alternatives match the attrs table above.
using AttrValue =
    std::variant<int, double, std::string, const Primitive *,
                 std::vector<const Primitive *>, std::vector<NodePtr>,
                 std::vector<double>, std::vector<std::string>>;

struct Node {
  int id = 0;
  NodeKind kind = NodeKind::UNKNOWN;
  const Primitive *primitive = nullptr; // nullptr for compound nodes
  std::map<std::string, AttrValue> attrs;

  bool has_attr(const std::string &key) const { return attrs.count(key) > 0; }

  /// attrs[key] as T, or std::nullopt when absent (mirrors `attrs.get(key)`).
  template <typename T> std::optional<T> get(const std::string &key) const {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return std::nullopt;
    }
    if (const T *value = std::get_if<T>(&it->second)) {
      return *value;
    }
    return std::nullopt;
  }

  /// attrs[key] as T, or `fallback` when absent (mirrors `attrs.get(key, x)`).
  template <typename T> T get_or(const std::string &key, T fallback) const {
    return get<T>(key).value_or(fallback);
  }
};

struct Edge {
  int src = 0;
  int dst = 0;
  Relation relation = Relation::HEAD_TO_STEM;
};

class NotationGraph {
public:
  std::map<int, NodePtr> nodes;
  std::vector<Edge> edges;
  double staff_space = 0.0;
  std::vector<double> staff_lines;

  NodePtr add_node(NodeKind kind, const Primitive *primitive = nullptr);

  void add_edge(int src, int dst, Relation relation);

  void remove_node(int node_id);

  std::vector<NodePtr> nodes_of_kind(NodeKind kind) const;

  std::vector<NodePtr> neighbors(int node_id) const;
  std::vector<NodePtr> neighbors(int node_id, Relation relation) const;

  std::vector<Edge> edges_of(int node_id) const;

private:
  int next_node_id_ = 0;
};

} // namespace omr