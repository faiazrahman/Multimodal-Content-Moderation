"""
Run (from root)
```
python -m argument_graphs.run_argument_graph_submodel_evaluation --config CONFIG [--args]
```
- Note: You must run this from root because the local package imports (e.g.,
  from `argument_graphs/`) and the data paths are defined based on the root
  directory

You must specify either --argumentative_unit_classification or
  --relationship_type_classification (i.e. exactly one of those args)
```
python -m argument_graphs.run_argument_graph_submodel_evaluation --argumentative_unit_classification --config CONFIG
python -m argument_graphs.run_argument_graph_submodel_evaluation --relationship_type_classification --config CONFIG
```

Yale-specific notes
- Running on single-GPU on Ziva (NVIDIA GeForce RTX 3090) with batch_size 16
  has 96-99% GPU utilization for AUC (for both BERT and RoBERTa); with batch_size 32 has 98-100% GPU
  utilization for RTC
```
# BERT
(mmcm) faiaz@ziva:~/CS490/Multimodal-Content-Moderation$
python -m argument_graphs.run_argument_graph_submodel_evaluation --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__bert-base-uncased.yaml --batch_size 16 --gpus 3
(mmcm) faiaz@ziva:~/CS490/Multimodal-Content-Moderation$
python -m argument_graphs.run_argument_graph_submodel_evaluation --relationship_type_classification --config configs/argumentative_unit_classification/rtc__bert-base-uncased.yaml --batch_size 32 --gpus 7

# RoBERTa
(mmcm) faiaz@ziva:~/CS490/Multimodal-Content-Moderation$
python -m argument_graphs.run_argument_graph_submodel_evaluation --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__roberta-base.yaml --batch_size 16 --gpus 3
```
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

from argument_graphs.dataloader import ArgumentativeUnitClassificationDataset,\
                                       RelationshipTypeClassificationDataset
from argument_graphs.models import ArgumentativeUnitClassificationModel, \
                                   RelationshipTypeClassificationModel
from models.callbacks import PrintCallback
from utils import get_checkpoint_filename_from_dir

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 to turn off multiprocessing
# If not specified by --num_cpus command-line arg or in config file, defaults
# to the following
DEFAULT_NUM_CPUS = 0

DEFAULT_GPUS = [0, 1]
DATA_PATH = "./data/ArgumentativeUnitClassification"

PL_ASSETS_PATH = "./lightning_logs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print(device)

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # Only (and only one) of these MUST be specified as a command-line argument
    parser.add_argument("--argumentative_unit_classification", action="store_true", help="Runs training for the argumentative unit classification (AUC) model")
    parser.add_argument("--relationship_type_classification", action="store_true", help="Runs training for the relationship type classification (entailment) model")

    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")
    parser.add_argument("--only_check_args", action="store_true", help="(Only for testing) Stops script after printing out args; doesn't actually run")

    parser.add_argument("--trained_model_version", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    parser.add_argument("--num_cpus", type=int, default=None, help="0 for no multi-processing, 24 on Yale Tangra server, 40 on Yale Ziva server")
    args = parser.parse_args()

    if args.argumentative_unit_classification and args.relationship_type_classification:
        raise Exception("You can only specify ONE of the following at a time: --argumentative_unit_classification OR --relationship_type_classification")
    elif (not args.argumentative_unit_classification) and (not args.relationship_type_classification):
        raise Exception("You must specify one of the following: --argumentative_unit_classification OR --relationship_type_classification")

    # NOTE: We allow passing just a trained model version number, since we can
    # get the hyperparameters from its `lightning_logs/version_*/hparams.yaml` file
    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
    else:
        if not args.trained_model_version:
            raise Exception("You must either pass a config filename to --config (which must match the experiment configuration used for training the model) OR pass a --trained_model_version to run evaluation")

    if not args.batch_size: args.batch_size = config.get("batch_size", 16)
    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)
    if not args.num_cpus: args.num_cpus = config.get("num_cpus", DEFAULT_NUM_CPUS)

    original_batch_size = None
    if args.trained_model_version:
        # User specified a trained model version as a command-line arg
        # Get the hyperparameters from its trained model assets folder's hparams.yaml
        trained_model_assets_hparams_filepath = os.path.join(
            "lightning_logs",
            "version_" + str(args.trained_model_version),
            "hparams.yaml")
        hparams_config = {}
        with open(trained_model_assets_hparams_filepath, "r") as yaml_file:
            hparams_config = yaml.safe_load(yaml_file)
        args.model = hparams_config.get("model", None)
        args.tokenizer = hparams_config.get("tokenizer", None)
        original_batch_size = hparams_config.get("batch_size", None)
        args.num_epochs = hparams_config.get("num_epochs", None)
        args.learning_rate = hparams_config.get("learning_rate", None)
        args.optimizer = hparams_config.get("optimizer", None)
        if args.optimizer == "sgd":
            args.sgd_momentum = hparams_config.get("sgd_momentum", None)
    else:
        # Otherwise, load the hparams from the specified config file
        args.trained_model_version = config.get("trained_model_version", None)
        args.trained_model_path = config.get("trained_model_path", None)

        args.model = config.get("model", "bert-base-uncased")
        args.tokenizer = config.get("tokenizer", "bert-base-uncased")
        args.batch_size = config.get("batch_size", 16)
        args.num_epochs = config.get("num_epochs", 5)
        args.learning_rate = config.get("learning_rate", 1e-4)
        args.optimizer = config.get("optimizer", "adam")
        if args.optimizer == "sgd":
            args.sgd_momentum = config.get("sgd_momentum", 0.9)

    print(f"Running evaluation with batch_size={args.batch_size} for a model trained with the following configuration...")
    print(f"model: {args.model}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"batch_size: {original_batch_size}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"optimizer: {args.optimizer}")
    if args.optimizer == "sgd": print(f"sgd_momentum: {args.sgd_momentum}")
    print(f"gpus: {args.gpus}")
    print(f"num_cpus: {args.num_cpus}")

    if args.only_check_args:
        quit()

    print("\nStarting evaluation...")

    full_dataset = None
    if args.argumentative_unit_classification:
        full_dataset = ArgumentativeUnitClassificationDataset(tokenizer=args.tokenizer)
    elif args.relationship_type_classification:
        full_dataset = RelationshipTypeClassificationDataset(tokenizer=args.tokenizer)
    logging.info("Total dataset size: {}".format(len(full_dataset)))
    logging.info(full_dataset)

    # Split into train dataset and test dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # NOTE: You must use the same exact seed for torch.Generate() for both the
    # training and evaluation of a model to ensure that the two datasets have
    # no overlapping examples; otherwise, evaluation will not be truly
    # representative of model performance
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(6)
    )
    logging.info(f"Test dataset size: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus,
        drop_last=True
    )
    logging.info(test_loader)

    # TODO(later): Move this code before the dataset?
    checkpoint_path = None
    if args.trained_model_version:
        assets_version = None
        if isinstance(args.trained_model_version, int):
            assets_version = "version_" + str(args.trained_model_version)
        elif isinstance(args.trained_model_version, str):
            assets_version = args.trained_model_version
        else:
            raise Exception("assets_version must be either an int (i.e. the version number, e.g. 16) or a str (e.g. \"version_16\"")
        checkpoint_path = os.path.join(PL_ASSETS_PATH, assets_version, "checkpoints")
    elif args.trained_model_path:
        checkpoint_path = args.trained_model_path
    else:
        raise Exception("A trained model must be specified for evaluation, either by version number (in default PyTorch Lightning assets path ./lightning_logs) or by custom path")

    checkpoint_filename = get_checkpoint_filename_from_dir(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, checkpoint_filename)
    logging.info(checkpoint_path)

    model = None
    if args.argumentative_unit_classification:
        model = ArgumentativeUnitClassificationModel.load_from_checkpoint(checkpoint_path)
    elif args.relationship_type_classification:
        model = RelationshipTypeClassificationModel.load_from_checkpoint(checkpoint_path)
    logging.info(model)

    callbacks = [
        PrintCallback(),
        TQDMProgressBar(refresh_rate=10)
    ]

    trainer = None
    if torch.cuda.is_available() and len(args.gpus) > 1:
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        trainer = pl.Trainer(
            gpus=args.gpus,
            strategy="dp",
            callbacks=callbacks,
        )
    elif torch.cuda.is_available():
        # Single GPU training (i.e. data parallel is not specified to Trainer)
        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer(
            callbacks=callbacks,
        )

    trainer.test(model, dataloaders=test_loader)
    # pl.LightningModule has some issues displaying the results automatically
    # As a workaround, we can store the result logs as an attribute of the
    # class instance and display them manually at the end of testing
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
    results = model.test_results

    print(checkpoint_path)
    print(results)
    logging.info(checkpoint_path)
    logging.info(results)
