import itertools
import functools
import operator
from collections import defaultdict
from typing import List, Dict

from ..utils import ArgumentGraphLinearizer
from .argumentative_unit_node import ArgumentativeUnitType, ArgumentativeUnitNode
from .relationship_type_edge import RelationshipType, RelationshipTypeEdge

class ArgumentGraph:

    def __init__(self):
        """
        Attributes
            mapping: Dict mapping Node -> List[Edge], where an Edge contains
                the source node, the target node, and the relationship
                type; maps child-to-parent relations (e.g. premise entails
                claim, claim entails claim, major claim to root)
            reverse_mapping: Dict mapping Node -> List[Edge], which is the
                reverse of the actual mapping; this is used for top-down
                traversal (e.g. to go from the root to its children, to go from
                a claim to the child claims and child premises that entail it)
            root: Node; root of the argument graph (tree), from which traversal
                will start
        """
        self.mapping: Dict[ArgumentativeUnitNode,
                           List[RelationshipTypeEdge]] = defaultdict(list)
        self.reverse_mapping: Dict[ArgumentativeUnitNode,
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

        # Create and add the reverse edge
        reverse_edge = RelationshipTypeEdge(
            source_node=edge.target_node,
            target_node=edge.source_node,
            relation=RelationshipType.TO_CHILD,
        )
        self.reverse_mapping[reverse_edge.source_node].append(reverse_edge)
        return

    def node_entails_no_edges(self, node: ArgumentativeUnitNode) -> bool:
        """ Returns True if node has no directed edges outward from itself """
        return len(self.mapping[node]) == 0

    def linearize(self) -> str:
        """ Linearizes the argument graph into a single text string """
        linearizer = ArgumentGraphLinearizer()
        return linearizer.linearize(self)

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

    def get_all_child_nodes(
        self,
        node: ArgumentativeUnitNode
    ) -> List[ArgumentativeUnitNode]:
        """ Gets all child nodes for a given node, using the reverse mapping """
        return [edge.target_node for edge in self.reverse_mapping[node]]

    def get_child_claim_nodes(
        self,
        node: ArgumentativeUnitNode
    ) -> List[ArgumentativeUnitNode]:
        """ Gets all child claim nodes for a given node """
        return [edge.target_node for edge in self.reverse_mapping[node]
                if edge.target_node.classification == ArgumentativeUnitType.CLAIM]

    def get_child_premise_nodes(
        self,
        node: ArgumentativeUnitNode
    ) -> List[ArgumentativeUnitNode]:
        """ Gets all child premise nodes for a given node """
        return [edge.target_node for edge in self.reverse_mapping[node]
                if edge.target_node.classification == ArgumentativeUnitType.PREMISE]

    def set_claim_child_subtree_sizes(self):
        """
        Recursively computes the child subtree sizes for every claim node in
        the tree, starting from the root node's child claims

        Note that this calls the helper `_compute_child_subtree_sizes(...)` on
        every single child claim node of the root (which can be thought of as
        k disjoint subtrees)
        """
        root_node_children = [edge.target_node for edge in self.reverse_mapping[self.root]]
        for root_child_claim in root_node_children:
            self._compute_child_subtree_sizes(root_child_claim)
        return

    def _compute_child_subtree_sizes(
        self,
        subtree_root_node: ArgumentativeUnitNode
    ):
        """
        Recursively computes the child subtree sizes for a single, connected
        subtree with the given root node

        Note that this only assigns values for claim nodes, since premise nodes
        have no children (so their subtree_size remains 0)
        """
        raise NotImplementedError("TODO: Subtree size recursive algorithm")
