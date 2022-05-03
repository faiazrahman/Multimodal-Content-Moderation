"""
python data_preprocessing.py --test --from_dialogue_dataframe data/Fakeddit/test__dialogue_dataframe.pkl --dialogue_method graphlin --modality text-image-dialogue
python data_preprocessing.py --test --from_dialogue_dataframe data/Fakeddit/test__dialogue_dataframe.pkl --dialogue_method argsum --modality text-image-dialogue
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import yaml

from dataloader import MultimodalDataset, Modality
# from model import JointTextImageModel, JointTextImageDialogueModel

from sentence_transformers import SentenceTransformer

DATA_PATH = "./data/Fakeddit"
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000
DEFAULT_TRAIN_DATA_PATH = os.path.join(DATA_PATH, "multimodal_train_" + str(TRAIN_DATA_SIZE) + ".tsv")
DEFAULT_TEST_DATA_PATH = os.path.join(DATA_PATH, "multimodal_test_" + str(TEST_DATA_SIZE) + ".tsv")

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Running on training data")
    parser.add_argument("--test", action="store_true", help="Running on test (evaluation) data")
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--from_dialogue_dataframe", type=str, default=None, help="If you're using dialogue (comment) data and have already filtered the dialogue dataframe, pass the path to its serialized .pkl file to continue preprocessing from that point on")
    parser.add_argument("--dir_to_save_dataframe", type=str, default="data/Fakeddit", help="Path to store saved dataframe .pkl file after data preprocessing")
    parser.add_argument("--prefix_for_all_generated_pkl_files", type=str, default="", help="Adds a prefix to all generated .pkl files for this run, e.g. sampled_train_...")
    parser.add_argument("--dialogue_method", type=str, default="ranksum", help="ranksum | graphlin | argsum; Notes: RankSum ranks all comments then summarizes with a Transformers pipeline; GraphLin constructs and linearizes an argument graph (but does not summarize); ArgSum constructs, linearizes, and summarizes an argument graph")
    parser.add_argument("--modality", type=str, default=None, help="text | image | text-image | text-image-dialogue")
    parser.add_argument("--dialogue_summarization_model", type=str, default=None, help="Transformers model for summarization pipeline; e.g. facebook/bart-large-cnn")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.load(yaml_file)

    # TODO: Add these as args too
    if not args.train_data_path: args.train_data_path = config.get("train_data_path", DEFAULT_TRAIN_DATA_PATH)
    if not args.test_data_path: args.test_data_path = config.get("test_data_path", DEFAULT_TEST_DATA_PATH)
    if not args.modality: args.modality = config.get("modality", "text-image")
    # args.num_classes = config.get("num_classes", 2) # TODO rm, not needed for preproessing since we keep all labels and return the right one in __getitem__
    if not args.dialogue_summarization_model: args.dialogue_summarization_model = config.get("dialogue_summarization_model", "facebook/bart-large-cnn")
    logging.info(args)

    # Note that text_embedder, image_transform, and image_encoder do not need
    # to be passed to the torch.utils.data.Dataset class for data preprocessing,
    # since they are only used when getting an item from the dataset
    # Data preprocessing will save the preprocessed data to a dataframe .pkl
    # which is then quickly loaded when instantiating the same dataset for
    # training or evaluation
    # The summarization_model, however, is needed, since data preprocessing
    # will generate the summaries using that model and save them into the
    # aforementioned dataframe .pkl

    logging.info("Running data_preprocessing.py...")
    logging.info("NOTE: Make sure that the images have already been downloaded for the train and test data via data/Fakeddit/image_downloader.py")

    if args.train:
        # Calling the MultimodalDataset constructor (i.e. __init__) will run
        # the necessary data preprocessing steps and dump the resulting dataframe
        # into a serialized .pkl file; the path to that .pkl file can then be
        # passed as the `processed_dataframe_path` arg in the config  (and in
        # turn, the `from_preprocessed_dataframe` arg in MultimodalDataset) for
        # training and evaluation
        train_dataset = MultimodalDataset(
            from_dialogue_dataframe=args.from_dialogue_dataframe,
            data_path=args.train_data_path,
            dir_to_save_dataframe=args.dir_to_save_dataframe,
            prefix_for_all_generated_pkl_files=args.prefix_for_all_generated_pkl_files,
            dataset_type="train",
            modality=args.modality,
            text_embedder=None,
            image_transform=None,
            image_encoder=None,
            dialogue_method=args.dialogue_method,
            summarization_model=args.dialogue_summarization_model,
            # num_classes=args.num_classes
        )
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        logging.info(train_dataset)

    if args.test:
        # See comment above
        test_dataset = MultimodalDataset(
            from_dialogue_dataframe=args.from_dialogue_dataframe,
            data_path=args.test_data_path,
            dir_to_save_dataframe=args.dir_to_save_dataframe,
            prefix_for_all_generated_pkl_files=args.prefix_for_all_generated_pkl_files,
            dataset_type="test",
            modality=args.modality,
            text_embedder=None,
            image_transform=None,
            image_encoder=None,
            dialogue_method=args.dialogue_method,
            summarization_model=args.dialogue_summarization_model,
            # num_classes=args.num_classes
        )
        logging.info("Test dataset size: {}".format(len(test_dataset)))
        logging.info(test_dataset)
