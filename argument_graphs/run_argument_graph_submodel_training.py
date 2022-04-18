"""
Run (from root)
```
python -m argument_graphs.run_argument_graph_submodel_training [--args]
```
- Note: You must run this from root because the local package imports (e.g.,
  from `argument_graphs/`) and the data paths are defined based on the root
  directory
"""

import os
import logging
import argparse

from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from argument_graphs.dataloader import ArgumentativeUnitClassificationDataset
from argument_graphs.models import ArgumentativeUnitClassificationModel

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 to turn off multiprocessing
# If not specified by --num_cpus command-line arg or in config file, defaults
# to the following
DEFAULT_NUM_CPUS = 0

DEFAULT_GPUS = [0, 1]
DATA_PATH = "./data/ArgumentativeUnitClassification"

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")
    parser.add_argument("--only_check_args", action="store_true", help="(Only for testing) Stops script after printing out args; doesn't actually run")

    # We default all hyperparameters to None so that their default values can
    # be taken from a config file; if the config file is not specified, then we
    # use the given default values in the `config.get()` calls (see below)
    # Thus the order of precedence for hyperparameter values is
    #   passed manually as an arg -> specified in given config file -> default
    # This allows experiments defined in config files to be easily replicated
    # while tuning specific parameters via command-line args
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    parser.add_argument("--num_cpus", type=int, default=None, help="0 for no multi-processing, 24 on Yale Tangra server, 40 on Yale Ziva server")

    parser.add_argument("--model", type=str, default=None, help="Base model for sequence classification; must be in Hugging Face Transformers pretrained models repository; default `bert-base-uncased`")
    parser.add_argument("--tokenizer", type=str, default=None, help="Base tokenizer for sequence classification; must be in Hugging Face Transformers pretrained models repository; default `bert-base-uncased`")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

    # Defaults specified here, if not specified by command-line arg or config
    if not args.model: args.model = config.get("model", "bert-base-uncased")
    if not args.tokenizer: args.tokenizer = config.get("tokenizer", "bert-base-uncased")
    if not args.batch_size: args.batch_size = config.get("batch_size", 32)
    if not args.learning_rate: args.learning_rate = config.get("learning_rate", 1e-4)
    if not args.num_epochs: args.num_epochs = config.get("num_epochs", 5)
    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)
    if not args.num_cpus: args.num_cpus = config.get("num_cpus", DEFAULT_NUM_CPUS)

    dataset = ArgumentativeUnitClassificationDataset()
    print(dataset)

    print("Running training with the following configuration...")
    print(f"model: {args.model}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"batch_size: {args.batch_size}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"gpus: {args.gpus}")
    print(f"num_cpus: {args.num_cpus}")

    if args.only_check_args:
        quit()

    print("\nStarting training...")

    hparams = {
        # Used by pl.LightningModule
        "model": args.model,
        "tokenizer": args.tokenizer,
        "learning_rate": args.learning_rate,

        # For logging (in `lighting_logs/version_*/hparams.yaml`)
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
    }

    model = ArgumentativeUnitClassificationModel(hparams)
    print(model)
