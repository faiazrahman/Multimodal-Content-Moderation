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

DATA_PATH = "./data"
IMAGES_DIR = os.path.join(DATA_PATH, "Fakeddit/images")
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000

