"""
Run (from root)
```
python -m argument_graphs.run_argument_graph_submodel_training [--args]
```
- Note: You must run this from root because the local package imports (e.g.,
  from `argument_graphs/`) and the data paths are defined based on the root
  directory

You must specify either --argumentative_unit_classification or
  --relationship_type_classification (i.e. exactly one of those args)
```
python -m argument_graphs.run_argument_graph_submodel_training --argumentative_unit_classification
python -m argument_graphs.run_argument_graph_submodel_training --relationship_type_classification
```

Yale-specific notes
- Running on single-GPU on Ziva (NVIDIA GeForce RTX 3090) with batch_size 16
  has 96-99% GPU utilization for AUC (for both BERT and RoBERTa); with batch_size 32 has 98-100% GPU
  utilization for RTC
```
# BERT
(mmcm) faiaz@ziva:~/CS490/Multimodal-Content-Moderation$
python -m argument_graphs.run_argument_graph_submodel_training --argumentative_unit_classification --batch_size 16 --gpus 3
(mmcm) faiaz@ziva:~/CS490/Multimodal-Content-Moderation$
python -m argument_graphs.run_argument_graph_submodel_training --relationship_type_classification --batch_size 32 --gpus 5

# RoBERTa
(mmcm) faiaz@ziva:~/CS490/Multimodal-Content-Moderation$
python -m argument_graphs.run_argument_graph_submodel_training --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__roberta-base.yaml --batch_size 16 --gpus 5
(mmcm) faiaz@ziva:~/CS490/Multimodal-Content-Moderation$
python -m argument_graphs.run_argument_graph_submodel_training --relationship_type_classification --config configs/argumentative_unit_classification/rtc__roberta-base.yaml --batch_size 32 --gpus 5
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
from argument_graphs.submodels import ArgumentativeUnitClassificationModel,\
                                   RelationshipTypeClassificationModel
from models.callbacks import PrintCallback

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 to turn off multiprocessing
# If not specified by --num_cpus command-line arg or in config file, defaults
# to the following
DEFAULT_NUM_CPUS = 0

DEFAULT_GPUS = [0, 1]
DATA_PATH = "./data/ArgumentativeUnitClassification"

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
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, help="adam | sgd")
    parser.add_argument("--sgd_momentum", type=float, default=None)
    args = parser.parse_args()

    if args.argumentative_unit_classification and args.relationship_type_classification:
        raise Exception("You can only specify ONE of the following at a time: --argumentative_unit_classification OR --relationship_type_classification")
    elif (not args.argumentative_unit_classification) and (not args.relationship_type_classification):
        raise Exception("You must specify one of the following: --argumentative_unit_classification OR --relationship_type_classification")

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

    # Defaults specified here, if not specified by command-line arg or config
    if not args.model: args.model = config.get("model", "bert-base-uncased")
    if not args.tokenizer: args.tokenizer = config.get("tokenizer", "bert-base-uncased")
    if not args.batch_size: args.batch_size = config.get("batch_size", 32)
    if not args.num_epochs: args.num_epochs = config.get("num_epochs", 5)
    if not args.learning_rate: args.learning_rate = config.get("learning_rate", 1e-4)
    if not args.optimizer: args.optimizer = config.get("optimizer", "adam")
    if not args.sgd_momentum: args.sgd_momentum = config.get("sgd_momentum", 0.9)
    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)
    if not args.num_cpus: args.num_cpus = config.get("num_cpus", DEFAULT_NUM_CPUS)

    print("Running training with the following configuration...")
    print(f"model: {args.model}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"optimizer: {args.optimizer}")
    if args.optimizer == "sgd": print(f"sgd_momentum: {args.sgd_momentum}")
    print(f"gpus: {args.gpus}")
    print(f"num_cpus: {args.num_cpus}")

    if args.only_check_args:
        quit()

    print("\nStarting training...")

    experiment_name = ""
    if args.argumentative_unit_classification:
        experiment_name = "ArgumentativeUnitClassificationModel"
    elif args.relationship_type_classification:
        experiment_name = "RelationshipTypeClassificationModel"

    hparams = {
        # Used by pl.LightningModule
        "model": args.model,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "sgd_momentum": args.sgd_momentum,

        # For logging (in `lighting_logs/version_*/hparams.yaml`)
        "tokenizer": args.tokenizer, # Used by torch.utils.data.Dataset
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "experiment_name": experiment_name,
    }

    full_dataset = None
    if args.argumentative_unit_classification:
        full_dataset = ArgumentativeUnitClassificationDataset(tokenizer=args.tokenizer)
    elif args.relationship_type_classification:
        full_dataset = RelationshipTypeClassificationDataset(tokenizer=args.tokenizer)

    logging.info("Total dataset size: {}".format(len(full_dataset)))
    logging.info(full_dataset)

    # Split into initial train dataset and test dataset
    # Note: The initial train dataset will further be split into train and val
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

    # Split into train and validation datasets
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(6)
    )
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(train_dataset)
    logging.info(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus,
        shuffle=False, # Do not shuffle DataLoader for validation
        drop_last=True
    )
    logging.info(train_loader)
    logging.info(val_loader)

    model = None
    if args.argumentative_unit_classification:
        model = ArgumentativeUnitClassificationModel(hparams)
    elif args.relationship_type_classification:
        model = RelationshipTypeClassificationModel(hparams)
    logging.info(model)

    trainer = None

    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=50,
        save_top_k=2,
    )
    final_checkpoint = ModelCheckpoint(
        filename="final-{epoch}-{step}",
        monitor="epoch",
        mode="max",
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=True
    )

    callbacks = [
        PrintCallback(),
        TQDMProgressBar(refresh_rate=10),
        latest_checkpoint,
        final_checkpoint
    ]

    if torch.cuda.is_available() and len(args.gpus) > 1:
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        trainer = pl.Trainer(
            gpus=args.gpus,
            strategy="dp",
            callbacks=callbacks,
            enable_checkpointing=True,
            max_epochs=args.num_epochs
        )
    elif torch.cuda.is_available():
        # Single GPU training (i.e. data parallel is not specified to Trainer)
        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=callbacks,
            enable_checkpointing=True,
            max_epochs=args.num_epochs
        )
    else:
        trainer = pl.Trainer(
            callbacks=callbacks,
            enable_checkpointing=True,
            max_epochs=args.num_epochs
        )
    logging.info(trainer)

    print(f"Starting training for {experiment_name} for {args.num_epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)
