from collections import deque
from typing import List

from argument_graphs.data_structures import ArgumentGraph, ArgumentativeUnitNode

class ArgumentGraphLinearizer:

    def __init__(self, linearization_method: str = "greedy-heuristics"):
        self.linearization_method = linearization_method

    def linearize(self, graph: ArgumentGraph, separator: str = "\n") -> str:
        """
        Runs graph linearization algorithm, returning a linearized text string
        """
        if self.linearization_method == "greedy-heuristics":
            return self.run_top_down_greedy_heuristics_linearization(
                graph,
                separator=separator
            )
        else:
            raise NotImplementedError("Error: Given linearization_method is not implemented")

    def run_top_down_greedy_heuristics_linearization(
        self,
        graph: ArgumentGraph,
        separator: str = "\n"
    ) -> str:
        """
        Runs graph linearization algorithm using top-down traversal with
        greedy heuristics (semantic ordering, subtree size prioritization, and
        zero-degree premise tailing)
        """

        linearized_graph_components = list()

        # Pre-compute claim subtree sizes
        graph.set_claim_child_subtree_sizes()

        # Initialize stack for depth-first traversal
        # Note that the stack only contains claim nodes (since premises are
        # immediately added to the linearized_graph_components, as per the
        # Semantic Ordering Heuristic)
        visited_nodes = set()
        stack = deque()

        # Add the child claim nodes of the root to the stack s.t. the claim
        # with the largest subtree is at the top (and will be processed first,
        # as per the Subtree Size Prioritization Heuristic)
        root_child_claims: List[ArgumentativeUnitNode] = graph.get_child_claim_nodes(graph.root)
        root_child_claims.sort(key=lambda node: node.subtree_size, reverse=False)
        for child_claim in root_child_claims:
            stack.append(child_claim)

        # Run traversal
        while len(stack) != 0:
            curr_node: ArgumentativeUnitNode = stack.pop()
            if curr_node in visited_nodes:
                # Depending on how the graph is constructed, this may not
                # ever happen
                continue
            visited_nodes.add(curr_node)

            # Greedy Step: Add the current claim node immediately
            linearized_graph_components.append(curr_node.text)

            # Semantic Ordering Heuristic: Child premises before child claims
            child_premises = graph.get_child_premise_nodes(curr_node)
            linearized_graph_components.extend(
                [premise.text for premise in child_premises]
            )

            # Subtree Size Prioritization Heuristicc: Large subtrees before
            # small ones; this means we put them onto the stack in order of
            # smallest to largest (so that in our next iteration, we pop the
            # largest child subtree)
            child_claims = graph.get_child_claim_nodes(curr_node)
            child_claims.sort(key=lambda node: node.subtree_size, reverse=False)
            for child_claim in child_claims:
                stack.append(child_claim)

        # Zero-Degree Premise Tailing Heuristic (after traversal, add child
        # premises of the root to the linearized string, i.e. at the very end)
        root_child_premises = graph.get_child_premise_nodes(graph.root)
        for premise in root_child_premises:
            linearized_graph_components.append(premise.text)

        # Stringify the linearized graph
        linearized_graph_string = separator.join(linearized_graph_components)
        return linearized_graph_string
