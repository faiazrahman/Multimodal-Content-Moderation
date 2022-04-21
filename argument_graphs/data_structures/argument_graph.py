import itertools
import functools
import operator
from collections import defaultdict
from typing import List, Dict

from .argumentative_unit_node import ArgumentativeUnitType, ArgumentativeUnitNode
from .relationship_type_edge import RelationshipType, RelationshipTypeEdge

class ArgumentGraph:

    def __init__(self):
        self.mapping: Dict[ArgumentativeUnitNode,
                           List[RelationshipTypeEdge]] = defaultdict(list)
        self.root: ArgumentativeUnitNode = None

    def __repr__(self):
        """ Representation of object """
        return f"ArgumentGraph({self.mapping}, {self.root})"

    def __str__(self):
        """ String casting of object """
        return self.linearize()

    def add_node(self, node: ArgumentativeUnitNode):
        """ Adds node to argument graph """
        if node not in self.mapping:
            self.mapping[node] = list()
        return

    def add_edge(self, edge: RelationshipTypeEdge):
        """
        Adds edge connecting source node and target node to the graph by
        1. Adding the source and target nodes if they are not already in the
           graph
        2. Adding a directed edge from the source node to the target node
        """

        if edge.source_node not in self.mapping:
            self.add_node(edge.source_node)

        if edge.target_node not in self.mapping:
            self.add_node(edge.target_node)

        self.mapping[edge.source_node].append(edge)
        return

    def node_has_no_outward_edges(self, node: ArgumentativeUnitNode) -> bool:
        """ Returns True if node has no directed edges outward from itself """
        return len(self.mapping[node]) == 0

    def linearize(self) -> str:
        """ Linearizes the argument graph into a single text string """
        raise NotImplementedError("TODO: Graph linearization algorithm")

    @property
    def all_nodes(self) -> List[ArgumentativeUnitNode]:
        """ Returns list of all nodes in the graph """
        return list(self.mapping.keys())

    @property
    def claim_nodes(self) -> List[ArgumentativeUnitNode]:
        """ Returns list of all claim nodes in the graph """
        return [node for node in list(self.mapping.keys())
                if node.classification == ArgumentativeUnitType.CLAIM]

    @property
    def premise_nodes(self) -> List[ArgumentativeUnitNode]:
        """ Returns list of all premise nodes in the graph """
        return [node for node in list(self.mapping.keys())
                if node.classification == ArgumentativeUnitType.PREMISE]

    @property
    def all_edges(self) -> List[RelationshipTypeEdge]:
        """
        Returns list of all edges in the graph

        Alternative Implementations
        - List comprehension
          `return [edge for edge_list in all_edge_lists for edge in edge_list]`
        - Using functools and operator
          `return list(functools.reduce(operator.iconcat, all_edge_lists, []))`
        - Using itertools
          `return list(itertools.chain.from_iterable(all_edge_lists))`
        """
        all_edge_lists = [edges for edges in self.mapping.values() if len(edges) > 0]
        return list(functools.reduce(operator.iconcat, all_edge_lists, []))
