"""
Run (from root)
```
python -m fusion.run_training [--args]
```

```
python -m fusion.run_training --only_check_args
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

from sentence_transformers import SentenceTransformer

from models.callbacks import PrintCallback
from fusion.dataloader import MMHSDataset
from fusion.model import TextImageResnetOcrMMHSModel

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 to turn off multiprocessing
# If not specified by --num_cpus command-line arg or in config file, defaults
# to the following
DEFAULT_NUM_CPUS = 0
# torch.multiprocessing.set_start_method('spawn')

DATA_PATH = "./data"
MMHS_DATA_PATH = os.path.join(DATA_PATH, "MMHS150K")
MMHS_TRAIN_DATAFRAME_PATH = os.path.join(MMHS_DATA_PATH, "mmhs_train_dataframe.pkl")
MMHS_TEST_DATAFRAME_PATH = os.path.join(MMHS_DATA_PATH, "mmhs_test_dataframe.pkl")
MMHS_VAL_DATAFRAME_PATH = os.path.join(MMHS_DATA_PATH, "mmhs_val_dataframe.pkl")

SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768
DEFAULT_GPUS = [0, 1]

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--only_check_args", action="store_true", help="(Only for testing) Stops script after printing out args; doesn't actually run")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")

    # We default all hyperparameters to None so that their default values can
    # be taken from a config file; if the config file is not specified, then we
    # use the given default values in the `config.get()` calls (see below)
    # Thus the order of precedence for hyperparameter values is
    #   passed manually as an arg -> specified in given config file -> default
    # This allows experiments defined in config files to be easily replicated
    # while tuning specific parameters via command-line args
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    parser.add_argument("--num_cpus", type=int, default=None, help="0 for no multi-processing, 24 on Yale Tangra server, 40 on Yale Ziva server")

    parser.add_argument("--model", type=str, default=None, help="Model to train; this must exactly match the filename that the model is saved in, excluding the .py extension; e.g. text_image_resnet_model")
    parser.add_argument("--modality", type=str, default=None, help="text | image | image-ocr | text-image-ocr")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--dropout_p", type=float, default=None)
    parser.add_argument("--fusion_output_size", type=int, default=None, help="Dimension after multi-modal embeddings fusion")
    parser.add_argument("--text_embedder", type=str, default=None, help="all-mpnet-base-v2 | all-distilroberta-v1")
    parser.add_argument("--image_encoder", type=str, default=None, help="resnet | dino")
    parser.add_argument("--dino_model", type=str, default=None, help="facebook/dino-vits16 | facebook/dino-vits8 | facebook/dino-vitb16 | facebook/dino-vitb8")
    parser.add_argument("--fusion_method", type=str, default=None, help="early-fusion | low-rank")
    parser.add_argument("--preprocessed_train_dataframe_path", type=str, default=None)
    parser.add_argument("--preprocessed_test_dataframe_path", type=str, default=None)
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

    # Defaults specified here, if not specified by command-line arg or config
    if not args.model: args.model = config.get("model", "text_image_resnet_ocr_mmhs_model")
    if not args.modality: args.modality = config.get("modality", "text-image-ocr")
    if not args.batch_size: args.batch_size = config.get("batch_size", 32)
    if not args.learning_rate: args.learning_rate = config.get("learning_rate", 1e-4)
    if not args.num_epochs: args.num_epochs = config.get("num_epochs", 5)
    if not args.dropout_p: args.dropout_p = config.get("dropout_p", 0.1)
    if not args.fusion_output_size: args.fusion_output_size = config.get("fusion_output_size", 512)
    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)
    if not args.num_cpus: args.num_cpus = config.get("num_cpus", DEFAULT_NUM_CPUS)
    if not args.text_embedder:
        args.text_embedder = config.get("text_embedder", "all-mpnet-base-v2")
    if not args.image_encoder:
        args.image_encoder = config.get("image_encoder", "resnet")
    if not args.dino_model and args.image_encoder == "dino":
        args.dino_model = config.get("dino_model", "facebook/dino-vitb16")
    if not args.fusion_method:
        args.fusion_method = config.get("fusion_method", "early-fusion")
    if not args.preprocessed_train_dataframe_path:
        args.preprocessed_train_dataframe_path = config.get("preprocessed_train_dataframe_path", MMHS_TRAIN_DATAFRAME_PATH)
    if not args.preprocessed_test_dataframe_path:
        args.preprocessed_test_dataframe_path = config.get("preprocessed_test_dataframe_path", MMHS_TEST_DATAFRAME_PATH)

    print("Running training with the following configuration...")
    print(f"model: {args.model}")
    print(f"modality: {args.modality}")
    print(f"batch_size: {args.batch_size}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"dropout_p: {args.dropout_p}")
    print(f"fusion_output_size: {args.fusion_output_size}")
    print(f"gpus: {args.gpus}")
    print(f"num_cpus: {args.num_cpus}")
    print(f"text_embedder: {args.text_embedder}")
    print(f"image_encoder: {args.image_encoder}")
    if args.image_encoder == "dino": print(f"dino_model: {args.dino_model}")
    print(f"fusion_method: {args.fusion_method}")
    print(f"preprocessed_train_dataframe_path: {args.preprocessed_train_dataframe_path}")
    print(f"preprocessed_test_dataframe_path: {args.preprocessed_test_dataframe_path}")

    if args.only_check_args:
        quit()

    print("\nStarting training...")

    hparams = {
        # Used by pl.LightningModule
        "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
        "learning_rate": args.learning_rate,
        "dropout_p": args.dropout_p,
        "fusion_output_size": args.fusion_output_size,
        "dino_model": args.dino_model,
        "fusion_method": args.fusion_method,

        # For logging
        "model": args.model,
        "modality": args.modality,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "text_embedder": args.text_embedder,
        "image_encoder": args.image_encoder,
        "preprocessed_train_dataframe_path": args.preprocessed_train_dataframe_path,
        "preprocessed_test_dataframe_path": args.preprocessed_test_dataframe_path,
    }

    model = None
    text_embedder = SentenceTransformer(args.text_embedder)
    image_transform = None

    if args.model == "text_image_resnet_ocr_mmhs_model":
        model = TextImageResnetOcrMMHSModel(hparams)
        image_transform = TextImageResnetOcrMMHSModel.build_image_transform()
    else:
        raise ValueError("fusion/run_training.py: Must pass a valid --model name to train")

    print(text_embedder)
    print(image_transform)

    train_dataset = MMHSDataset(
        from_dataframe_pkl_path=args.preprocessed_train_dataframe_path,
        dataset_type="train",
        modality=args.modality,
        text_embedder=text_embedder,
        image_transform=image_transform
    )
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    logging.info(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus
    )
    logging.info(train_loader)

    print(model)

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

    if torch.cuda.is_available():
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        trainer = pl.Trainer(
            gpus=args.gpus,
            strategy="dp",
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

    print(f"Starting training for {args.model} for {args.num_epochs} epochs...")
    trainer.fit(model, train_loader)
