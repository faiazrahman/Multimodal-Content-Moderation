import logging
from typing import List, Tuple

import transformers
from transformers import AutoTokenizer

from argument_graphs.modules import ArgumentGraphConstructor, \
    ArgumentGraphLinearizer
from argument_graphs.data_structures import ArgumentGraph
from argument_graphs.utils import encode_single_inputs

logging.basicConfig(level=logging.INFO)

ENTAILMENT_SCORE_MINIMUM_THRESHOLD = 0.2

class ArgSum:

    def __init__(
        self,
        # For utterance segmentation (within graph construction)
        segmentation_method: str = "sentence",
        # For argument graph construction
        auc_trained_model_version: int = None,
        rtc_trained_model_version: int = None,
        auc_tokenizer_model_name: str = "roberta-base",
        rtc_tokenizer_model_name: str = "bert-base-uncased",
        auc_model_batch_size: int = 16,
        rtc_model_batch_size: int = 32,
        entailment_score_minimum_threshold: float = ENTAILMENT_SCORE_MINIMUM_THRESHOLD,
        # For argument graph linearization
        linearization_method: str = "greedy-heuristics",
        # For linearized text summarization without fine-tuning (transformers)
        summarization_transformers_model_name: int = "facebook/bart-large-cnn",
        # For linearized text summarization with fine-tuning (DialogueSumm)
        # Note that `fine_tuned_for_dialogue` must be true to use fine-tuned model
        fine_tuned_for_dialogue: bool = False,
        dialogue_summarization_tokenizer_model_name: str = "facebook/bart-large-cnn",
        dialogue_summarization_model_version: int = None
    ):
        self.graph_constructor = ArgumentGraphConstructor(
            segmentation_method=segmentation_method,
            auc_trained_model_version=auc_trained_model_version,
            rtc_trained_model_version=rtc_trained_model_version,
            auc_tokenizer_model_name=auc_tokenizer_model_name,
            rtc_tokenizer_model_name=rtc_tokenizer_model_name,
            auc_model_batch_size=auc_model_batch_size,
            rtc_model_batch_size=rtc_model_batch_size,
            entailment_score_minimum_threshold=entailment_score_minimum_threshold
        )
        self.graph_linearizer = ArgumentGraphLinearizer(
            linearization_method=linearization_method
        )

        self.fine_tuned_for_dialogue = fine_tuned_for_dialogue
        self.tokenizer = None # If using fine-tuned DialogueSummarizationModel
        self.text_summarizer = None
        if fine_tuned_for_dialogue:
            self.tokenizer = AutoTokenizer.from_pretrained(dialogue_summarization_tokenizer_model_name)
            # TODO: We should have a DialogueSummarizationModel class that
            # can be fine-tuned on SamSum conversation data
            self.text_summarizer = None
            raise NotImplementedError("Need to implement DialogueSummarizationModel")
        else:
            # Load Hugging Face `transformers` summarization pipeline using the
            # specified base model
            self.text_summarizer = transformers.pipeline(
                "summarization",
                model=summarization_transformers_model_name
            )

    def summarize(self, dialogue_utterances: List[str]) -> str:
        """
        Summarizes the given list of dialogue utterances (strings) using the
        ArgSum algorithm

        Note that the configuration for the ArgSum algorithm is specified
        during instantiation of an `ArgSum(...)` object (i.e. the specific
        trained submodel versions, the segmentation method, the linearization
        method, etc.)
        """
        graph: ArgumentGraph = self.graph_constructor.construct_graph(dialogue_utterances)
        linearized_graph: str = self.graph_linearizer.linearize(graph)

        summary = "none"
        if self.fine_tuned_for_dialogue:
            # TODO: Tokenize the linearized_graph using self.tokenizer and
            # encode_single_inputs() (from argument_graphs.utils)
            # TODO: Look into how BART generates tensors with its tokenizer
            encoded_inputs = encode_single_inputs(linearized_graph)
            summary = None
            raise NotImplementedError("Need to implement DialogueSummarizationModel")
        else:
            # Using summarization pipeline without fine-tuning on dialogue
            # Note: We define the summary's max_length as max(min(75, num_words // 2), 5)
            # Note that num_words is calculated very roughly, splitting on whitespace
            num_words = len(linearized_graph.split())
            max_length = min(75, num_words // 2) # For short comment threads, it'll be <75
            max_length = max(max_length, 5) # Avoid 1-length maxes, which leads to unexpected behavior
            min_length = min(5, max_length - 1)
            summary = self.text_summarizer(
                linearized_graph,
                min_length=min_length,
                max_length=max_length,
                truncation=True
            )

            # Pipeline returns a list containing a dict
            # https://huggingface.co/docs/transformers/master/en/main_classes/pipelines
            summary = summary[0]['summary_text']

        return summary
