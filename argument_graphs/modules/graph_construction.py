from .utterance_segmentation import UtteranceToArgumentativeUnitSegmenter

class ArgumentGraphConstructor:

    def __init__(self, segmentation_method: str = "sentence"):
        self.utterance_segmenter = UtteranceToArgumentativeUnitSegmenter(
            segmentation_method=segmentation_method
        )
