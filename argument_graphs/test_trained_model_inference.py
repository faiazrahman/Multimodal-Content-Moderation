"""
Only for testing out functionality of running inference on a trained model
(specifically, a pl.LightningModule)

Run (from root)
```
python -m argument_graphs.test_trained_model_inference
```
"""

import os
import logging
import argparse

from transformers import AutoTokenizer

from argument_graphs.dataloader import ArgumentativeUnitClassificationDataset
from argument_graphs.models import ArgumentativeUnitClassificationModel
from utils import get_checkpoint_filename_from_dir

TRAINED_MODEL_VERSION = 114
PL_ASSETS_PATH = "./lightning_logs"

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    trained_model_version = TRAINED_MODEL_VERSION
    assets_version = "version_" + str(trained_model_version)
    checkpoint_path = os.path.join(PL_ASSETS_PATH, assets_version, "checkpoints")
    checkpoint_filename = get_checkpoint_filename_from_dir(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, checkpoint_filename)
    print(checkpoint_path)

    model = ArgumentativeUnitClassificationModel.load_from_checkpoint(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = [
        # Claims (label:1)
        "I think we should make healthcare free",
        "I hate this post",
        # Premises (label:2)
        "Free healthcare is accomplished through policymakers",
        "The images in this post are scary",
        # Non-argumentative units (label:0)
        "and then",
        "because of this",
    ]

    encoded_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    pred = model(encoded_inputs)[0]
    print(pred)
    print(pred.tolist())
