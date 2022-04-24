"""
Run (from root)
```
python -m scripts.test_argument_graph_cycle_detection
```
"""

from argument_graphs.data_structures import ArgumentGraph, ArgumentativeUnitNode,\
    RelationshipTypeEdge, ArgumentativeUnitType, RelationshipType

if __name__ == "__main__":

    # Should be false
    graph = ArgumentGraph()
    graph.print_graph()

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
    graph.print_graph()
    graph.add_edge(e1b)
    graph.print_graph()

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
    graph.print_graph()

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
    graph.print_graph()
    graph.add_edge(e3b)
    graph.print_graph()

    print(graph.has_cycle())
    assert(graph.has_cycle() == False)

    # Should be true
    graph = ArgumentGraph()
    graph.print_graph()

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
    graph.print_graph()
    graph.add_edge(e1b)
    graph.print_graph()

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
    graph.print_graph()

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
    c3c2 = RelationshipTypeEdge(
        c3, c2, RelationshipType.SUPPORTS
    )
    e3a = RelationshipTypeEdge(
        p3a, c3, RelationshipType.SUPPORTS
    )
    e3b = RelationshipTypeEdge(
        p3b, c3, RelationshipType.SUPPORTS
    )
    graph.add_edge(c3c2)
    graph.add_edge(e3a)
    graph.print_graph()
    graph.add_edge(e3b)
    graph.print_graph()

    # Induce cycle
    c1c3 = RelationshipTypeEdge(
        c1, c3, RelationshipType.SUPPORTS
    )
    graph.add_edge(c1c3)
    graph.print_graph()

    print(graph.has_cycle())
    assert(graph.has_cycle() == True)

    # Should be true
    graph = ArgumentGraph()
    graph.print_graph()

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
    graph.print_graph()
    graph.add_edge(e1b)
    graph.print_graph()

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
    graph.print_graph()

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
    c3c2 = RelationshipTypeEdge(
        c3, c2, RelationshipType.SUPPORTS
    )
    e3a = RelationshipTypeEdge(
        p3a, c3, RelationshipType.SUPPORTS
    )
    e3b = RelationshipTypeEdge(
        p3b, c3, RelationshipType.SUPPORTS
    )
    graph.add_edge(c3c2)
    graph.add_edge(e3a)
    graph.print_graph()
    graph.add_edge(e3b)
    graph.print_graph()

    # Induce cycle
    c1c2 = RelationshipTypeEdge(
        c1, c2, RelationshipType.SUPPORTS
    )
    graph.add_edge(c1c2)
    graph.print_graph()

    print(graph.has_cycle())
    assert(graph.has_cycle() == True)
