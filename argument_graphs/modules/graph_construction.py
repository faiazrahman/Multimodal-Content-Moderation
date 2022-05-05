import logging
from typing import List, Tuple

import transformers
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

ENTAILMENT_SCORE_MINIMUM_THRESHOLD = 0.2

class ArgumentGraphConstructor:

    def __init__(
        self,
        segmentation_method: str = "sentence",
        auc_trained_model_version: int = None,
        rtc_trained_model_version: int = None,
        auc_tokenizer_model_name: str = "roberta-base",
        rtc_tokenizer_model_name: str = "bert-base-uncased",
        auc_model_batch_size: int = 1, # 16, # TODO revert
        rtc_model_batch_size: int = 1, # 32, # TODO revert
        entailment_score_minimum_threshold: float = ENTAILMENT_SCORE_MINIMUM_THRESHOLD,
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
        self.entailment_score_minimum_threshold = entailment_score_minimum_threshold

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
            # print(preds)

            # Create a node for each argumentative unit in the batch
            for i, argumentative_unit in enumerate(argumentative_unit_batch):
                classification = ArgumentativeUnitType.from_label(preds[i])
                node = ArgumentativeUnitNode(
                    text=argumentative_unit,
                    classification=classification
                )
                all_nodes.append(node)
                all_nodes_by_type[classification].append(node)

        # # ::TESTING
        # for node_type in all_nodes_by_type.keys():
        #     print(str(node_type))
        #     for node in all_nodes_by_type[node_type]:
        #         print(node.text)
        #     print("")
        # # ::END

        # Instantiate an ArgumentGraph object
        graph = ArgumentGraph()

        # Premise-to-claim entailment
        for premise in all_nodes_by_type[ArgumentativeUnitType.PREMISE]:
            all_claims = all_nodes_by_type[ArgumentativeUnitType.CLAIM]
            max_claim, max_probability = ArgumentGraphConstructor.source_to_targets_entailment(
                source=premise,
                targets=all_claims,
                tokenizer=self.rtc_tokenizer,
                rtc_model=self.rtc_model,
                batch_size=self.rtc_model_batch_size
            )

            if max_probability and max_probability > self.entailment_score_minimum_threshold:
                edge = RelationshipTypeEdge(
                    source_node=premise,
                    target_node=max_claim,
                    relation=RelationshipType.SUPPORTS
                )
                graph.add_edge(edge)

        # Claim-to-claim entailment
        # Note that this process is slightly different than premise-to-claim
        # entailment, since we need to avoid adding edges which would create a
        # cycle; (Note that this was not possible in premise-to-claim
        # entailment, since we always drew edges from a premise to a claim, so
        # there was no possibility of a claim entailing a premise)

        # 1. First, store all potential claim-to-claim edges
        # This is a list of tuples of (potential_edge, entailment_score), where
        # the entailment score is the probability from the RTC model
        potential_claim_to_claim_edges = list() # type: List[Tuple[RelationshipTypeEdge, float]]
        for source_claim in all_nodes_by_type[ArgumentativeUnitType.CLAIM]:
            all_claims = all_nodes_by_type[ArgumentativeUnitType.CLAIM]
            max_target_claim, max_probability = ArgumentGraphConstructor.source_to_targets_entailment(
                source=source_claim,
                targets=all_claims,
                tokenizer=self.rtc_tokenizer,
                rtc_model=self.rtc_model,
                batch_size=self.rtc_model_batch_size
            )

            if max_probability:
                edge = RelationshipTypeEdge(
                    source_node=source_claim,
                    target_node=max_target_claim,
                    relation=RelationshipType.SUPPORTS
                )
                potential_claim_to_claim_edges.append(tuple([edge, max_probability]))

        # # ::TESTING
        # print(potential_claim_to_claim_edges)
        # # ::END

        # 2. Next, greedily add edges in order of decreasing entailment score
        # only if it does not create a cycle in the graph
        for edge, probability in sorted(
            potential_claim_to_claim_edges,
            key=lambda x: x[1], # Sort by probability (second value in tuple)
            reverse=True
        ):
            # Try adding the edge, but if it causes a cycle, remove it
            graph.add_edge(edge)
            if graph.has_cycle():
                # See docstring of `remove_edge_in_cycle()` in `ArgumentGraph`
                # class for explanation of how the edge is removed for this
                # specific case (i.e. an edge causing a cycle)
                graph.remove_edge_in_cycle(edge)

        # Root node linking
        root = ArgumentativeUnitNode(classification=ArgumentativeUnitType.ROOT_NODE)
        for claim in all_nodes_by_type[ArgumentativeUnitType.CLAIM]:
            if graph.node_entails_no_edges(claim):
                edge = RelationshipTypeEdge(
                    source_node=claim,
                    target_node=root,
                    relation=RelationshipType.TO_ROOT
                )
                graph.add_edge(edge)

        # Set the root node
        graph.root = root

        # Add leftover premises to the graph as children of the root
        for premise in all_nodes_by_type[ArgumentativeUnitType.PREMISE]:
            if premise not in set(graph.mapping.keys()):
                edge = RelationshipTypeEdge(
                    source_node=premise,
                    target_node=root,
                    relation=RelationshipType.TO_ROOT
                )
                graph.add_edge(edge)

        # # ::TESTING
        # graph.print_mapping()
        # print(f"has_cycle: {graph.has_cycle()}")
        # graph.print_graph()
        # # ::END

        return graph

    @staticmethod
    def source_to_targets_entailment(
        source: ArgumentativeUnitNode = None,
        targets: List[ArgumentativeUnitNode] = None,
        tokenizer: transformers.AutoTokenizer = None,
        rtc_model: RelationshipTypeClassificationModel = None,
        batch_size: int = 32,
    ) -> Tuple[ArgumentativeUnitNode, int]:
        """
        Computes entailment score for given `source` against all other
        `targets` (checking that the other `targets` do not include the given
        `source`, i.e. no self-loops in the graph) and returns the node in
        `targets` which the given `source` entails with the highest score,
        and that score

        Returns None, None if there are no entailment (SUPPORTS) relationships
        between the source and all the targets
        """

        # If the `source` is in `targets`, remove it
        targets = [target for target in targets if target != source]

        # Create batches of targets (so we can pass more through the RTC model
        # simultaneously, increasing efficiency)
        targets_batches = generate_batches(targets, batch_size=batch_size)

        max_probability = None
        max_target = None
        for targets_batch in targets_batches:
            # Extract the text only
            targets_batch_text = [target.text for target in targets_batch]

            # Create a batch of the source text repeated, matching the size of
            # the targets batch; this is because we are passing the same source
            # every time to the RTC model (with every possible target)
            # e.g. source_batch = ["a", "a", "a", "a", "a", "a"]
            #      target_batch = ["b", "c", "d", "e", "f", "g"]
            source_batch_text = [source.text for _ in range(len(targets_batch))]

            encoded_inputs = encode_paired_inputs(
                source_batch_text,
                targets_batch_text,
                tokenizer=tokenizer
            )
            preds, losses = rtc_model(encoded_inputs) # TODO: Return probabilities too
            probabilities = [1.0 for _ in range(len(targets_batch))] # TODO: Get probabilities from model
            # print(preds)

            for i, target in enumerate(targets_batch):
                if (preds[i] == int(RelationshipType.SUPPORTS)
                    and (not max_probability or probabilities[i] > max_probability)):
                    max_probability = probabilities[i]
                    max_target = target

        # # ::TESTING
        # print(max_probability)
        # print(max_target)
        # if max_target: print(source.text); print(max_target.text)
        # # ::END

        return max_target, max_probability
