import sys
import os
from pathlib import Path
import logging
import argparse
import enum

import pandas as pd
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

DATA_PATH = "./data"
PL_ASSETS_PATH = "./lightning_logs"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
IMAGE_EXTENSION = ".jpg"

class Modality(enum.Enum):
    """
    Note on Comparisons: Either use `string_value == enum.value`
    or `Modality(string_value) == enum`
    """
    TEXT = "text"
    IMAGE = "image"
    TEXT_IMAGE = "text-image"
    TEXT_IMAGE_DIALOGUE = "text-image-dialogue"

class MultimodalDataset(Dataset):

    def __init__(
        self,
        from_preprocessed_dataframe=None, # Preprocessed dataframe to load from
        from_dialogue_dataframe=None, # Partially preprocessed df to load from
        data_path=None, # Path to data (i.e. not using preprocessed dataframe)
        dir_to_save_dataframe="data", # Save the preprocessed dataframe here
        dataset_type="train",
        modality=None,
        text_embedder=None,
        image_encoder=None,
        image_transform=None,
        summarization_model=None,
        num_classes=2,
        images_dir=IMAGES_DIR
    ):
        df = None
