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
IMAGES_DIR = os.path.join(DATA_PATH, "images")
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Running on training data")
    parser.add_argument("--test", action="store_true", help="Running on test (evaluation) data")
    parser.add_argument("--from_dialogue_dataframe", type=str, default=None, help="If you're using dialogue (comment) data and have already filtered the dialogue dataframe, pass the path to its serialized .pkl file to continue preprocessing from that point on")
    parser.add_argument("--dir_to_save_dataframe", type=str, default="data/Fakeddit", help="Path to store saved dataframe .pkl file after data preprocessing")
    parser.add_argument("--modality", type=str, default=None, help="text | image | text-image | text-image-dialogue")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.load(yaml_file)

    args.train_data_path = config.get("train_data_path", os.path.join(DATA_PATH, "multimodal_train_" + str(TRAIN_DATA_SIZE) + ".tsv"))
    args.test_data_path = config.get("test_data_path", os.path.join(DATA_PATH, "multimodal_test_" + str(TEST_DATA_SIZE) + ".tsv"))
    if not args.modality: args.modality = config.get("modality", "text-image")
    args.num_classes = config.get("num_classes", 2)
    args.text_embedder = config.get("text_embedder", "all-mpnet-base-v2")
    args.image_encoder = config.get("image_encoder", "resnet")
    args.dialogue_summarization_model = config.get("dialogue_summarization_model", "facebook/bart-large-cnn")
    logging.info(args)

    # text_embedder = SentenceTransformer(args.text_embedder)
    # image_transform = None
    # if Modality(args.modality) == Modality.TEXT_IMAGE_DIALOGUE:
    #     image_transform = JointTextImageDialogueModel.build_image_transform()
    # else:
    #     image_transform = JointTextImageModel.build_image_transform()

    logging.info("Running data_preprocessing.py...")
    logging.info("NOTE: Make sure that the images have already been downloaded for the train and test data via data/Fakeddit/image_downloader.py")

    if args.train:
        # Calling the MultimodalDataset constructor (i.e. __init__) will run
        # the necessary data preprocessing steps and dump the resulting dataframe
        # into a serialized .pkl file; the path to that .pkl file can then be
        # passed as the `processed_dataframe_path` arg in the config for training
        # and evaluation
        train_dataset = MultimodalDataset(
            from_dialogue_dataframe=args.from_dialogue_dataframe,
            data_path=args.train_data_path,
            dir_to_save_dataframe=args.dir_to_save_dataframe,
            dataset_type="train",
            modality=args.modality,
            text_embedder=None, # TODO text_embedder,
            image_transform=None, # TODO image_transform,
            summarization_model=args.dialogue_summarization_model,
            images_dir=IMAGES_DIR,
            num_classes=args.num_classes
        )
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        logging.info(train_dataset)

    if args.test:
        # See comment above
        test_dataset = MultimodalDataset(
            from_dialogue_dataframe=args.from_dialogue_dataframe,
            data_path=args.test_data_path,
            dir_to_save_dataframe=args.dir_to_save_dataframe,
            dataset_type="test",
            modality=args.modality,
            text_embedder=None, # TODO text_embedder,
            image_transform=None, # TODO image_transform,
            summarization_model=args.dialogue_summarization_model,
            images_dir=IMAGES_DIR,
            num_classes=args.num_classes
        )
        logging.info("Test dataset size: {}".format(len(test_dataset)))
        logging.info(test_dataset)
