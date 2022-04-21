from collections import defaultdict

class ArgumentGraph:

    def __init__(self):
        self.mapping = defaultdict(list)
        self.root = None

    def __repr__(self):
        """ Representation of object """
        return f"ArgumentGraph({self.mapping}, {self.root})"

    def add_node(self, node):
        pass

    def add_edge(self, edge):
        pass

    def node_has_no_outward_edges(self, node):
        pass

    @property
    def all_nodes(self):
        pass

    @property
    def claim_nodes(self):
        pass

    @property
    def premise_nodes(self):
        pass

    @property
    def all_edges(self):
        pass
