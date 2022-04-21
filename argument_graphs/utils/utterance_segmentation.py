import spacy
from typing import List

class UtteranceToArgumentativeUnitSegmenter:

    def __init__(self, segmentation_method: str = "sentence"):
        """
        An instance of this class can only segment sentences via one
        segmentation method; to use a different segmentation method, another
        instance must be made
        """
        self.segmentation_method = segmentation_method

        self.spacy_pipeline = None
        if segmentation_method == "sentence":
            self.spacy_pipeline = spacy.load("en_core_web_sm")

    def segment(self, text: str) -> List[str]:
        """
        Segments a single text string (utterance) into a list of its
        argumentative units
        """
        if self.segmentation_method == "sentence":
            return self.run_sentence_segmentation(text)
        else:
            raise NotImplementedError("Given segmentation_method is not implemented")

    def run_sentence_segmentation(self, text: str) -> List[str]:
        """
        Segments a text string into sentences
        """
        processed_text = self.spacy_pipeline(text)
        sentences = [sentence for sentence in processed_text.sents]
        return sentences
