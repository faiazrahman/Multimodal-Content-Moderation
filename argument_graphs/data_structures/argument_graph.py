import itertools
import functools
import operator
from collections import defaultdict, deque
from re import L
from typing import List, Dict, Set, Deque

# from ..modules import ArgumentGraphLinearizer # Circular dependency
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

    # def __str__(self):
    #     """ String casting of object """
    #     return self.linearize()

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

    def remove_edge_in_cycle(self, edge: RelationshipTypeEdge):
        """
        Removes an edge in a cycle (leaving nodes unaffected)

        Note that the following always holds
            Thm: If adding an edge to a graph creates a cycle, both its source
            and target nodes are not new nodes to the graph (i.e., they were
            already present in the connected graph before this edge was drawn)
        - This is why we can only remove the edge and not worry about whether
          we added a new node (and thus must remove it as well)
        """
        self.mapping[edge.source_node].remove(edge)

        # Note that since the mappings store a list of Edge objects, we cannot
        # create a new reverse edge and .remove() it, since that will have a
        # different memory address (despite being the same logical reverse
        # edge); thus, we search for the actual reverse edge object then remove
        # it from the reverse mapping
        reverse_edge_to_remove = None
        for reverse_edge in self.reverse_mapping[edge.target_node]:
            if reverse_edge.target_node == edge.source_node:
                reverse_edge_to_remove = reverse_edge
                break
        self.reverse_mapping[edge.target_node].remove(reverse_edge_to_remove)

    def node_entails_no_edges(self, node: ArgumentativeUnitNode) -> bool:
        """ Returns True if node has no directed edges outward from itself """
        return len(self.mapping[node]) == 0

    # def linearize(self) -> str:
    #     """ Linearizes the argument graph into a single text string """
    #     linearizer = ArgumentGraphLinearizer()
    #     return linearizer.linearize(self)

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

    def set_child_subtree_sizes(self):
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
        curr_node: ArgumentativeUnitNode
    ):
        """
        Recursively computes the child subtree sizes for a single, connected
        subtree with the given root node

        Note that all premise nodes will have subtree_size of 1 (just themself,
        since they have no children)

        Note: On large argument graphs, this recursive approach may exceed the
        call stack limit; as a result, an iterative approach with a stack data
        structure may need to be implemented instead
        """
        curr_node.subtree_size = 1
        for child_node in self.get_all_child_nodes(curr_node):
            self._compute_child_subtree_sizes(child_node)
            curr_node.subtree_size += child_node.subtree_size

    def has_cycle(self) -> bool:
        """
        Returns true if the graph contains a cycle; false if not

        Runs depth-first search, keeping track of visited nodes and marking
        nodes with colors (see CLRS text on algorithms)

        Uses colors to mark vertices
            white   Not visited
            grey    Visited, but all vertices reachable from this vertex have
                    not yet been visited
            black   Finished, i.e. this vertex and all vertices reachable from
                    it have been visited
        """
        color = { node: "white" for node in list(self.mapping.keys()) }
        # We will store whether or not we found a cycle as a list of a single
        # element (so that its value can be changed via call by reference in
        # the recursive calls)
        found_cycle = [False]

        for node in list(self.mapping.keys()):
            if color[node] == "white":
                self._find_cycle(node, color, found_cycle)
            if found_cycle[0] == True:
                break

        if found_cycle[0] == True:
            return True
        else:
            return False

    def _find_cycle(
        self,
        node: ArgumentativeUnitNode,
        color: Dict[ArgumentativeUnitNode, str],
        found_cycle: List[bool]
    ):
        """ Recursive helper for finding cycles in graph """
        if found_cycle[0] == True:
            return
        color[node] = "grey"
        for neighbor in [edge.target_node for edge in self.mapping[node]]:
            if color[neighbor] == "grey":
                found_cycle[0] = True
                return
            if color[neighbor] == "white":
                self._find_cycle(neighbor, color, found_cycle)
        color[node] = "black"

    def print_mapping(self):
        """ Prints the mapping to the terminal, for simple visuals """
        for node in list(self.mapping.keys()):
            print(f"{str(node.classification)}: {node.text}")
            for edge in self.mapping[node]:
                print(f"  {str(edge.target_node.classification)}: {edge.target_node.text}")

    def print_graph(self):
        """
        Prints the argument graph to the terminal, for simple visuals

        e.g.

        CLAIM: Free healthcare should be a human right
        -PREMISE: Free healthcare in Canada is well-received
        -PREMISE: Citizens like free healthcare
            CLAIM: I think free healthcare is great
            -PREMISE: My healthcare plan is free
        CLAIM: The president's healthcare policies are horrible
        -PREMISE: Did you see the awful healthcare policy?
        -PREMISE: Citizens are upset
        """

        TAB = " " * 4

        print(type(self))
        visited_nodes = set()
        stack = deque()

        root_child_claims = self.get_child_claim_nodes(self.root)
        root_child_claims.sort(key=lambda node: node.subtree_size, reverse=False)
        for child_claim in root_child_claims:
            stack.append((child_claim, TAB))

        while len(stack) != 0:
            curr_node, tabs = stack.pop()
            if curr_node in visited_nodes:
                continue
            visited_nodes.add(curr_node)

            # Print current claim node and its immediate child premises
            print(tabs + str(curr_node.classification) + ": " + curr_node.text)
            child_premises = self.get_child_premise_nodes(curr_node)
            for premise in child_premises:
                print(tabs + "-" + str(premise.classification) + ": " + premise.text)

            # Append the child claims (with one additional tab)
            child_claims = self.get_child_claim_nodes(curr_node)
            child_claims.sort(key=lambda node: node.subtree_size, reverse=False)
            for claim in child_claims:
                stack.append((claim, tabs + TAB))

        root_child_premises = self.get_child_premise_nodes(self.root)
        for child_premise in root_child_premises:
            print(TAB + str(child_premise.classification) + ": " + child_premise.text)

        print("")
