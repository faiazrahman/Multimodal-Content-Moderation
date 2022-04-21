from typing import List

class UtteranceToArgumentativeUnitSegmenter:

    def __init__(self, segmentation_method: str = "sentence"):
        """
        An instance of this class can only segment sentences via one
        segmentation method; to use a different segmentation method, another
        instance must be made
        """
        self.segmentation_method = segmentation_method

    def segment(self, text: str) -> List[str]:
        """
        Segments a single text string (utterance) into a list of its
        argumentative units
        """
        if self.segmentation_method == "sentence":
            return UtteranceToArgumentativeUnitSegmenter\
                .run_sentence_segmentation(text)
        else:
            raise NotImplementedError("Given segmentation_method is not implemented")

    @staticmethod
    def run_sentence_segmentation(text: str) -> List[str]:
        """
        Segments a text string into sentences
        """
        pass
