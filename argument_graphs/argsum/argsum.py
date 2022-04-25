import logging
from typing import List, Tuple

from transformers import AutoModel

from argument_graphs.modules import ArgumentGraphConstructor, \
    ArgumentGraphLinearizer

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

        self.text_summarizer = None
        if fine_tuned_for_dialogue:
            # TODO: We should have a DialogueSummarizationModel class that
            # can be fine-tuned on SamSum conversation data
            raise NotImplementedError("Need to implement DialogueSummarizationModel")
        else:
            # Load a base model from Hugging Face `transformers`
            self.text_summarizer = AutoModel.from_pretrained(
                summarization_transformers_model_name
            )

    def summarize(dialogue_utterances: List[str]) -> str:
        """
        Summarizes the given list of dialogue utterances (strings) using the
        ArgSum algorithm

        Note that the configuration for the ArgSum algorithm is specified
        during instantiation of an `ArgSum(...)` object (i.e. the specific
        trained submodel versions, the segmentation method, the linearization
        method, etc.)
        """
        raise NotImplementedError("ArgSum.summarize()")
