import enum

class ArgumentativeUnitType(enum.Enum):
    """ Represents a classification of an argumentative unit """
    NON_ARGUMENTATIVE_UNIT = 0
    CLAIM = 1
    PREMISE = 2
    ROOT_NODE = 3

    def __str__(self):
        return str(self.name)

    def __int__(self):
        return int(self.value)

class ArgumentativeUnitNode:

    def __init__(
        self,
        text: str = "",
        classification: ArgumentativeUnitType = None
    ):
        self.text = text
        self.classification = classification
