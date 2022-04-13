import os
import logging

import torch
import torch.nn as nn
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import transformers
from sentence_transformers import SentenceTransformer

# NOTE: These should be initialized in the calling script
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
DROPOUT_P = 0.1

RESNET_OUT_DIM = 2048

losses = []

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

print("CUDA available:", torch.cuda.is_available())


