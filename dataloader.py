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

        self.dataset_type = dataset_type
        self.modality = modality
        self.num_classes = num_classes
        self.dir_to_save_dataframe = dir_to_save_dataframe

        self.label = "2_way_label"
        if num_classes == 3:
            self.label = "3_way_label"
        elif num_classes == 6:
            self.label = "6_way_label"

        self.saved_dataframe_filename_prefix = ""
        if Modality(modality) == Modality.TEXT:
            self.saved_dataframe_filename_prefix = "text"
        elif Modality(modality) == Modality.IMAGE:
            self.saved_dataframe_filename_prefix = "image"
        elif Modality(modality) == Modality.TEXT_IMAGE:
            self.saved_dataframe_filename_prefix = "text_image"
        elif Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
            self.saved_dataframe_filename_prefix = "text_image_dialogue"

        df = None
        if not from_preprocessed_dataframe:
            # This is the first time this data is being setup, so we run full preprocessing
            df = pd.read_csv(data_path, sep='\t', header=0)
            df = self._preprocess_df(df)
            logging.debug(df.columns)
            logging.debug(df['clean_title'])

            # Run dialogue preprocessing, if needed
            if Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
                # Special Case: Since the dialogue data is huge, we can first preprocess the
                # dialogue (comments) dataframe to only keep the comments that pertain to
                # posts in our dataset's data, and save it into a serialized .pkl file;
                # If we do that, then we'll run our dialogue preprocessing using that
                # dataframe (which we load from the .pkl file)
                if from_dialogue_dataframe:
                    self._preprocess_dialogue(from_saved_df_path=from_dialogue_dataframe)
                else:
                    self._preprocess_dialogue()
        else: # from_preprocessed_dataframe:
            df = None
            if isinstance(from_preprocessed_dataframe, pd.DataFrame):
                df = from_preprocessed_dataframe
            elif isinstance(from_preprocessed_dataframe, str):
                df = pd.read_pickle(from_preprocessed_dataframe)
            else:
                raise Exception("MultimodalDataset given invalid from_preprocessed_dataframe arg; \
                                 Must be path (str) to dataframe or pd.DataFrame")

        self.data_frame = df
        self.text_ids = set(self.data_frame['id'])
        logging.debug(self.data_frame)

        self.text_embedder = text_embedder
        self.image_transform = image_transform
        self.summarization_model = summarization_model

        # TODO: Handle in-house summarization model
        # self.summarizer = None
        # if Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE and summarization_model:
        #     # Model options: "bart-large-cnn", "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
        #     # https://huggingface.co/docs/transformers/master/en/main_classes/pipelines#transformers.SummarizationPipeline
        #     self.summarizer = transformers.pipeline("summarization", model=summarization_model)
        # elif Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
        #     self.summarizer = transformers.pipeline("summarization")

    def __len__(self):
        return len(self.data_frame.index)

    def __getitem__(self, idx):
        pass

    def _preprocess_df(self, df):
        def image_exists(row):
            """ Ensures that image exists and can be opened """
            image_path = os.path.join(IMAGES_DIR, row['id'] + IMAGE_EXTENSION)
            if not os.path.exists(image_path):
                return False

            try:
                image = Image.open(image_path)
                image.verify()
                image.close()
                return True
            except Exception:
                return False

        df['image_exists'] = df.apply(lambda row: image_exists(row), axis=1)
        df = df[df['image_exists'] == True].drop('image_exists', axis=1)
        df = df.drop(['created_utc', 'domain', 'hasImage', 'image_url'], axis=1)
        df.reset_index(drop=True, inplace=True)

        # Save this dataframe into a pickle
        # Filename will look something like "train__text_image_dialogue__dataframe.pkl"
        filename = "__".join([self.dataset_type, self.saved_dataframe_filename_prefix, "dataframe.pkl"])
        save_path = os.path.join(self.dir_to_save_dataframe, filename)
        df.to_pickle(save_path)
        print("Preprocessed dataframe saved to {}".format(save_path))
        logging.info("Preprocessed dataframe saved to {}".format(save_path))

        return

    def _preprocess_dialogue(self, from_saved_df_path=""):
        pass
