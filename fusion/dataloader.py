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

from dataloader import Modality # enum from root's dataloader

DATA_PATH = "./data/MMHS150K"
PL_ASSETS_PATH = "./lightning_logs"
IMAGES_DIR = os.path.join(DATA_PATH, "img_resized")
IMAGE_EXTENSION = ".jpg"

class MMHSDataset(Dataset):
    """
    torch.utils.data.Dataset for multi-modal hate speech detection

    Supports data for text, image, image-ocr, and text-image-ocr
    - Note that `ocr` refers to text within an image, which is stored in a
      string after running Optical Character Recognition
    """
    def __init__(
        self,
        from_dataframe_pkl_path=None,
        dataset_type="train", # "train" | "test" | "val"
        modality=None,
        text_embedder=None,
        image_transform=None,
        # image_encoder=None,
        num_classes=6,
    ):
        self.dataset_type = dataset_type
        self.modality = modality
        self.num_classes = num_classes

        self.text_embedder = text_embedder
        self.image_transform = image_transform
        # self.image_encoder = image_encoder

        if not from_dataframe_pkl_path:
            raise Exception("You must run fusion/data_preprocessing.py then pass the path to the saved pd.DataFrame .pkl into MMHSDataset()")

        df = None
        if os.path.exists(from_dataframe_pkl_path):
            df = pd.read_pickle(from_dataframe_pkl_path)
        else:
            raise Exception("Dataframe path passed to MMHSDataset() does not exist")
        self.data_frame = df
        # print(self.data_frame)

        # # Precompute embeddings for both text and image OCR text during dataset
        # # initialization for faster training
        # self.encoded_texts = [
        #     self.text_embedder.encode(text, convert_to_tensor=True)
        #     for text in self.data_frame['text']
        # ]

        # self.encoded_ocrs = [
        #     self.text_embedder.encode(ocr, convert_to_tensor=True)
        #     for ocr in self.data_frame['image_ocr_text']
        # ]

    def __len__(self):
        """ Returns the size of the dataset; called as len(dataset) """
        return len(self.data_frame.index)

    def __getitem__(self, idx: int):
        """
        Returns a dict containing
            text            embedding Tensor
            image           RGB Tensor
            image_ocr_text  embedding Tensor
            label           int
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_id = self.data_frame.loc[idx, 'id']

        label = torch.Tensor(
            [self.data_frame.loc[idx, 'label']]
        ).long().squeeze()

        item = {
            "id": item_id,
            "label": label
        }

        if Modality(self.modality) in [Modality.TEXT, Modality.TEXT_IMAGE, Modality.TEXT_IMAGE_OCR]:
            # encoded_text = self.encoded_texts[idx]
            text = self.data_frame.loc[idx, 'text']
            encoded_text = self.text_embedder.encode(text, convert_to_tensor=True)
            item['text'] = encoded_text

        if Modality(self.modality) in [Modality.IMAGE, Modality.IMAGE_OCR, Modality.TEXT_IMAGE, Modality.TEXT_IMAGE_OCR]:
            image_path = os.path.join(IMAGES_DIR, item_id + IMAGE_EXTENSION)
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image)
            item['image'] = image

        if Modality(self.modality) in [Modality.IMAGE_OCR, Modality.TEXT_IMAGE_OCR]:
            # encoded_ocr = self.encoded_ocrs[idx]
            ocr = self.data_frame.loc[idx, 'image_ocr_text']
            encoded_ocr = self.text_embedder.encode(ocr, convert_to_tensor=True)
            item['image_ocr_text'] = encoded_ocr

        return item
