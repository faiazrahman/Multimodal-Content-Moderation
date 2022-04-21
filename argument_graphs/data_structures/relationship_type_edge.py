import enum

from argumentative_unit_node import ArgumentativeUnitNode

class RelationshipType(enum.Enum):
    """ Represents a classification of an entailment relationship """
    NEUTRAL = 0
    SUPPORTS = 1 # Entails
    CONTRADICTS = 2
    TO_ROOT = 3

    def __str__(self):
        return str(self.name)

    def __int__(self):
        return int(self.value)

class RelationshipTypeEdge:

    def __init__(
        self,
        source_node: ArgumentativeUnitNode = None,
        target_node: ArgumentativeUnitNode = None,
        relation: RelationshipType = None,
    ):
        self.source_node = source_node
        self.target_node = target_node
        self.relation = relation
