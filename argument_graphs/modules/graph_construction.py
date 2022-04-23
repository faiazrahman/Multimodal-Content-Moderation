import logging
from typing import List

from transformers import AutoTokenizer

from utils import get_checkpoint_path_from_trained_model_version
from argument_graphs.utils import generate_batches, encode_single_inputs, \
    encode_paired_inputs
from argument_graphs.data_structures import ArgumentGraph, ArgumentativeUnitNode, \
    RelationshipTypeEdge, ArgumentativeUnitType, RelationshipType
from argument_graphs.submodels import ArgumentativeUnitClassificationModel, \
    RelationshipTypeClassificationModel
from .utterance_segmentation import UtteranceToArgumentativeUnitSegmenter

logging.basicConfig(level=logging.INFO)

class ArgumentGraphConstructor:

    def __init__(
        self,
        segmentation_method: str = "sentence",
        auc_trained_model_version: int = None,
        rtc_trained_model_version: int = None,
        auc_tokenizer_model_name: str = "roberta-base",
        rtc_tokenizer_model_name: str = "bert-base-uncased",
        auc_model_batch_size: int = 16,
        rtc_model_batch_size: int = 32,
    ):
        logging.info("Initializing ArgumentGraphConstructor instance...")
        logging.info("NOTE: Make sure that the auc_tokenizer_model_name matches the base model of the given auc_trained_model_version (and the same for rtc_*)")
        logging.info("> You can check what base model was used to train each model by checking hparams.yaml files in the lightning_logs/ folders for your trained models")
        if not auc_trained_model_version or not rtc_trained_model_version:
            raise ValueError("ArgumentGraphConstructor must be passed both an auc_trained_model_version and a rtc_trained_model_version")

        self.utterance_segmenter = UtteranceToArgumentativeUnitSegmenter(
            segmentation_method=segmentation_method
        )

        self.auc_tokenizer = AutoTokenizer.from_pretrained(auc_tokenizer_model_name)
        self.rtc_tokenizer = AutoTokenizer.from_pretrained(rtc_tokenizer_model_name)
        self.auc_model = ArgumentativeUnitClassificationModel.load_from_checkpoint(
            get_checkpoint_path_from_trained_model_version(auc_trained_model_version)
        )
        self.rtc_model = RelationshipTypeClassificationModel.load_from_checkpoint(
            get_checkpoint_path_from_trained_model_version(rtc_trained_model_version)
        )

        self.auc_model_batch_size = auc_model_batch_size
        self.rtc_model_batch_size = rtc_model_batch_size

    def construct_graph(self, dialogue_utterances: List[str]) -> ArgumentGraph:
        # Utterance segmentation
        dialogue_argumentative_units = list()
        for utterance in dialogue_utterances:
            argumentative_units = self.utterance_segmenter.segment(utterance) # type: List[str]
            dialogue_argumentative_units.extend(argumentative_units)

        # Argumentative unit classification
        # Keep track of all the nodes both as a whole, and by their type (claim
        # or premise)
        all_nodes = list()
        all_nodes_by_type = {
            ArgumentativeUnitType.CLAIM: list(),
            ArgumentativeUnitType.PREMISE: list(),
            ArgumentativeUnitType.NON_ARGUMENTATIVE_UNIT: list(),
        }

        auc_batches = generate_batches(
            dialogue_argumentative_units,
            batch_size=self.auc_model_batch_size
        )
        for argumentative_unit_batch in auc_batches:
            # Pass through AUC model (in batches) to get classification prediction
            encoded_inputs = encode_single_inputs(
                argumentative_unit_batch,
                tokenizer=self.auc_tokenizer
            )
            preds, losses = self.auc_model(encoded_inputs)
            print(preds)

            # Create a node for each argumentative unit in the batch
            for i, argumentative_unit in enumerate(argumentative_unit_batch):
                classification = ArgumentativeUnitType.from_label(preds[i])
                node = ArgumentativeUnitNode(
                    text=argumentative_unit,
                    classification=classification
                )
                all_nodes.append(node)
                all_nodes_by_type[classification].append(node)

        for node_type in all_nodes_by_type.keys():
            print(str(node_type))
            for node in all_nodes_by_type[node_type]:
                print(node.text)
            print("")
