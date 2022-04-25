"""
Run (from root)

python -m scripts.test_argument_graph_linearization
"""

from argument_graphs.data_structures import ArgumentGraph, ArgumentativeUnitNode,\
    RelationshipTypeEdge, ArgumentativeUnitType, RelationshipType
from argument_graphs.modules import ArgumentGraphLinearizer

BERT = "bert-base-uncased"
ROBERTA = "roberta-base"

def test_no_root_premises():

    graph = ArgumentGraph()

    root = ArgumentativeUnitNode("", ArgumentativeUnitType.ROOT_NODE)
    graph.root = root
    graph.add_node(root)

    c1 = ArgumentativeUnitNode(
        "Free healthcare should be a human right",
        ArgumentativeUnitType.CLAIM
    )
    p1a = ArgumentativeUnitNode(
        "Free healthcare in Canada is well-received",
        ArgumentativeUnitType.PREMISE
    )
    p1b = ArgumentativeUnitNode(
        "Citizens like free healthcare",
        ArgumentativeUnitType.PREMISE
    )
    c1root = RelationshipTypeEdge(
        c1, root, RelationshipType.TO_ROOT
    )
    e1a = RelationshipTypeEdge(
        p1a, c1, RelationshipType.SUPPORTS
    )
    e1b = RelationshipTypeEdge(
        p1b, c1, RelationshipType.SUPPORTS
    )
    graph.add_edge(c1root)
    graph.add_edge(e1a)
    graph.add_edge(e1b)

    c2 = ArgumentativeUnitNode(
        "I think free healthcare is great",
        ArgumentativeUnitType.CLAIM
    )
    p2a = ArgumentativeUnitNode(
        "My healthcare plan is free",
        ArgumentativeUnitType.PREMISE
    )
    c2c1 = RelationshipTypeEdge(
        c2, c1, RelationshipType.SUPPORTS
    )
    e2a = RelationshipTypeEdge(
        p2a, c2, RelationshipType.SUPPORTS
    )
    graph.add_edge(c2c1)
    graph.add_edge(e2a)

    c3 = ArgumentativeUnitNode(
        "The president's healthcare policies are horrible",
        ArgumentativeUnitType.CLAIM
    )
    p3a = ArgumentativeUnitNode(
        "Did you see the awful healthcare policy?",
        ArgumentativeUnitType.PREMISE
    )
    p3b = ArgumentativeUnitNode(
        "Citizens are upset",
        ArgumentativeUnitType.PREMISE
    )
    c3root = RelationshipTypeEdge(
        c3, root, RelationshipType.TO_ROOT
    )
    e3a = RelationshipTypeEdge(
        p3a, c3, RelationshipType.SUPPORTS
    )
    e3b = RelationshipTypeEdge(
        p3b, c3, RelationshipType.SUPPORTS
    )
    graph.add_edge(c3root)
    graph.add_edge(e3a)
    graph.add_edge(e3b)
    graph.print_graph()

    linearizer = ArgumentGraphLinearizer()
    print(linearizer)

    linearized_graph = linearizer.linearize(graph)
    print(linearized_graph)

def test_with_root_premises():
    graph = ArgumentGraph()

    root = ArgumentativeUnitNode("", ArgumentativeUnitType.ROOT_NODE)
    graph.root = root
    graph.add_node(root)

    c1 = ArgumentativeUnitNode(
        "Free healthcare should be a human right",
        ArgumentativeUnitType.CLAIM
    )
    p1a = ArgumentativeUnitNode(
        "Free healthcare in Canada is well-received",
        ArgumentativeUnitType.PREMISE
    )
    p1b = ArgumentativeUnitNode(
        "Citizens like free healthcare",
        ArgumentativeUnitType.PREMISE
    )
    c1root = RelationshipTypeEdge(
        c1, root, RelationshipType.TO_ROOT
    )
    e1a = RelationshipTypeEdge(
        p1a, c1, RelationshipType.SUPPORTS
    )
    e1b = RelationshipTypeEdge(
        p1b, c1, RelationshipType.SUPPORTS
    )
    graph.add_edge(c1root)
    graph.add_edge(e1a)
    graph.add_edge(e1b)

    c2 = ArgumentativeUnitNode(
        "I think free healthcare is great",
        ArgumentativeUnitType.CLAIM
    )
    p2a = ArgumentativeUnitNode(
        "My healthcare plan is free",
        ArgumentativeUnitType.PREMISE
    )
    c2c1 = RelationshipTypeEdge(
        c2, c1, RelationshipType.SUPPORTS
    )
    e2a = RelationshipTypeEdge(
        p2a, c2, RelationshipType.SUPPORTS
    )
    graph.add_edge(c2c1)
    graph.add_edge(e2a)

    c3 = ArgumentativeUnitNode(
        "The president's healthcare policies are horrible",
        ArgumentativeUnitType.CLAIM
    )
    p3a = ArgumentativeUnitNode(
        "Did you see the awful healthcare policy?",
        ArgumentativeUnitType.PREMISE
    )
    p3b = ArgumentativeUnitNode(
        "Citizens are upset",
        ArgumentativeUnitType.PREMISE
    )
    c3root = RelationshipTypeEdge(
        c3, root, RelationshipType.TO_ROOT
    )
    e3a = RelationshipTypeEdge(
        p3a, c3, RelationshipType.SUPPORTS
    )
    e3b = RelationshipTypeEdge(
        p3b, c3, RelationshipType.SUPPORTS
    )
    graph.add_edge(c3root)
    graph.add_edge(e3a)
    graph.add_edge(e3b)

    pr1 = ArgumentativeUnitNode(
        "People in Santa Barbara are in support of free healthcare",
        ArgumentativeUnitType.PREMISE
    )
    pr2 = ArgumentativeUnitNode(
        "Last week this policy was popular",
        ArgumentativeUnitType.PREMISE
    )
    pr3 = ArgumentativeUnitNode(
        "Senators enjoy the weekend off",
        ArgumentativeUnitType.PREMISE
    )
    pr1e = RelationshipTypeEdge(
        pr1, root, RelationshipType.TO_ROOT
    )
    pr2e = RelationshipTypeEdge(
        pr2, root, RelationshipType.TO_ROOT
    )
    pr3e = RelationshipTypeEdge(
        pr3, root, RelationshipType.TO_ROOT
    )
    graph.add_edge(pr1e)
    graph.add_edge(pr2e)
    graph.add_edge(pr3e)
    graph.print_graph()

    linearizer = ArgumentGraphLinearizer()
    print(linearizer)

    linearized_graph = linearizer.linearize(graph)
    print(linearized_graph)

if __name__ == "__main__":
    test_no_root_premises()
    print("")
    test_with_root_premises()
