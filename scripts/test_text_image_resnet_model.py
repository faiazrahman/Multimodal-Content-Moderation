import os
import logging
import argparse

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from sentence_transformers import SentenceTransformer

from dataloader import MultimodalDataset, Modality
from models.text_image_resnet_model import TextImageResnetMMFNDModel

SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768

if __name__ == "__main__":
    hparams = {
        "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
        "num_classes": 6
    }

    model = TextImageResnetMMFNDModel(hparams)
    print(model)
