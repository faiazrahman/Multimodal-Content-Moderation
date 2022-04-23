"""
Run (from root)
python -m scripts.test_sentence_level_utterance_segmentation
"""

from pprint import pprint

from argument_graphs.modules import UtteranceToArgumentativeUnitSegmenter

if __name__ == "__main__":
    segmenter = UtteranceToArgumentativeUnitSegmenter()
    print(segmenter)
    text = "Hello there! My name is Faiaz. Not to be confused with Mr. Rahman. Is that a bell?"
    segments = segmenter.segment(text)
    pprint(segments)

    assert(isinstance(segments, list))
    assert(isinstance(segments[0], str))
