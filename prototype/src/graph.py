from enum import StrEnum
from dataclasses import dataclass, field
from src.primitives import PrimitiveKind, Primitive


class Relation(StrEnum):
    """Relation between primitives in a graph."""

    HEAD_TO_STEM = "head_to_stem"
    STEM_TO_BEAM = "stem_to_beam"
    ABOVE = "above"
    ALIGNED_VERTICAL = "aligned_vertical"


class NodeKind(StrEnum):
    """Kind of node in a graph. superset of PrimitiveKind."""

    NOTEHEAD = "notehead"
    STEM = "stem"
    BEAM = "beam"
    UNKNOWN = "unknown"
    NOTE = "note"
    CHORD = "chord"
    BEAMED_GROUP = "beamed_group"
    MEASURE = "measure"


@dataclass
class Node:
    """A node in the graph."""

    id: int
    kind: NodeKind
    primitive: Primitive | None
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """An edge in the graph."""

    src: int
    dst: int
    relation: Relation


@dataclass
class NotationGraph:
    """A graph of musical notation primitives."""

    nodes: dict[int, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    staff_space: float = 0.0
    staff_lines: tuple[float, ...] = field(default_factory=tuple)
    _next_node_id: int = 0

    def add_node(self, kind, primitive=None, **attrs) -> Node:
        node_id = self._next_node_id
        self._next_node_id += 1
        node = Node(id=node_id, kind=kind, primitive=primitive, attrs=attrs)
        self.nodes[node_id] = node
        return node

    def add_edge(self, src: int, dst: int, relation: Relation) -> None:
        edge = Edge(src=src, dst=dst, relation=relation)
        self.edges.append(edge)

    def remove_node(self, node_id: int) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.edges = [
                e for e in self.edges if e.src != node_id and e.dst != node_id
            ]

    def nodes_of_kind(self, kind: PrimitiveKind) -> list[Node]:
        return [node for node in self.nodes.values() if node.kind == kind]

    def neighbors(self, node_id: int, relation: Relation = None) -> list[Node]:
        if relation is None:
            neighbor_ids = {e.dst for e in self.edges if e.src == node_id} | {
                e.src for e in self.edges if e.dst == node_id
            }
        else:
            neighbor_ids = {
                e.dst for e in self.edges if e.src == node_id and e.relation == relation
            } | {
                e.src for e in self.edges if e.dst == node_id and e.relation == relation
            }
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def edges_of(self, node_id: int) -> list[Edge]:
        return [e for e in self.edges if e.src == node_id or e.dst == node_id]
