import os
import logging
import argparse

from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sentence_transformers import SentenceTransformer

from dataloader import MultimodalDataset, Modality
from models.text_image_resnet_model import TextImageResnetMMFNDModel

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 to turn off multiprocessing
NUM_CPUS = 40

DATA_PATH = "./data/Fakeddit"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768
DEFAULT_GPUS = [0, 1]

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO rm these first two
    # parser.add_argument("--train", action="store_true", help="Running on training data")
    # parser.add_argument("--test", action="store_true", help="Running on test (evaluation) data")
    parser.add_argument("--in_house_dialogue_summ", action="store_true", help="For training a model using in-house dialogue summarization data")
    parser.add_argument("--argument_graph", action="store_true", help="For training a model using dialogue argument graphs")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")

    # We default all hyperparameters to None so that their default values can
    # be taken from a config file; if the config file is not specified, then we
    # use the given default values in the `config.get()` calls (see below)
    # Thus the order of precedence for hyperparameter values is
    #   passed manually as an arg -> specified in given config file -> default
    # This allows experiments defined in config files to be easily replicated
    # while tuning specific parameters via command-line args
    parser.add_argument("--modality", type=str, default=None, help="text | image | text-image | text-image-dialogue")
    parser.add_argument("--num_classes", type=int, default=None, help="2 | 3 | 6")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--dropout_p", type=float, default=None)
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    parser.add_argument("--text_embedder", type=str, default=None, help="all-mpnet-base-v2 | all-distilroberta-v1")
    parser.add_argument("--image_encoder", type=str, default=None, help="resnet | dino")
    parser.add_argument("--dialogue_summarization_model", type=str, default=None, help="(Does NOT use in-house dialogue summarization) None=Transformers.Pipeline default i.e. sshleifer/distilbart-cnn-12-6 | bart-large-cnn | t5-small | t5-base | t5-large")
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--preprocessed_train_dataframe_path", type=str, default=None)
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.load(yaml_file)

    # Defaults specified here, if not specified by command-line arg or config
    if not args.modality: args.modality = config.get("modality", "text-image")
    if not args.num_classes: args.num_classes = config.get("num_classes", 2)
    if not args.batch_size: args.batch_size = config.get("batch_size", 32)
    if not args.learning_rate: args.learning_rate = config.get("learning_rate", 1e-4)
    if not args.num_epochs: args.num_epochs = config.get("num_epochs", 10)
    if not args.dropout_p: args.dropout_p = config.get("dropout_p", 0.1)
    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)
    if not args.text_embedder:
        args.text_embedder = config.get("text_embedder", "all-mpnet-base-v2")
    if not args.image_encoder:
        args.image_encoder = config.get("image_encoder", "resnet")
    if not args.dialogue_summarization_model:
        args.dialogue_summarization_model = config.get("dialogue_summarization_model", "bart-large-cnn")
    if not args.train_data_path:
        args.train_data_path = config.get("train_data_path", os.path.join(DATA_PATH, "multimodal_train_" + str(TRAIN_DATA_SIZE) + ".tsv"))
    if not args.preprocessed_train_dataframe_path:
        args.preprocessed_train_dataframe_path = config.get("preprocessed_train_dataframe_path", None)
