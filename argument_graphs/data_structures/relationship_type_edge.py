import enum

from .argumentative_unit_node import ArgumentativeUnitNode

class RelationshipType(enum.Enum):
    """ Represents a classification of an entailment relationship """
    NEUTRAL = 0
    SUPPORTS = 1 # Entails
    CONTRADICTS = 2
    TO_ROOT = 3

    def __str__(self):
        """ Call as `str(...)` """
        return str(self.name)

    def __int__(self):
        """ Call as `int(...)` """
        return int(self.value)

    @classmethod
    def from_label(cls, label: int) -> 'RelationshipType':
        if label == cls.NEUTRAL.value:
            return cls.NEUTRAL
        elif label == cls.SUPPORTS.value:
            return cls.SUPPORTS
        elif label == cls.CONTRADICTS.value:
            return cls.CONTRADICTS
        elif label == cls.TO_ROOT.value:
            return cls.TO_ROOT
        else:
            return None

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
