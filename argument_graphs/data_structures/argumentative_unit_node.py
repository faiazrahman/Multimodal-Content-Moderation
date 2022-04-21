import enum

class ArgumentativeUnitType(enum.Enum):
    """ Represents a classification of an argumentative unit """
    NON_ARGUMENTATIVE_UNIT = 0
    CLAIM = 1
    PREMISE = 2
    ROOT_NODE = 3

    def __str__(self):
        """ Call as `str(...)` """
        return str(self.name)

    def __int__(self):
        """ Call as `int(...)` """
        return int(self.value)

    @classmethod
    def from_label(cls, label: int) -> 'ArgumentativeUnitType':
        """ For convenience in instantiation """
        if label == cls.NON_ARGUMENTATIVE_UNIT.value:
            return cls.NON_ARGUMENTATIVE_UNIT
        elif label == cls.CLAIM.value:
            return cls.CLAIM
        elif label == cls.PREMISE.value:
            return cls.PREMISE
        elif label == cls.ROOT_NODE.value:
            return cls.ROOT_NODE
        else:
            return None

class ArgumentativeUnitNode:

    def __init__(
        self,
        text: str = "",
        classification: ArgumentativeUnitType = None,
    ):
        self.text = text
        self.classification = classification
