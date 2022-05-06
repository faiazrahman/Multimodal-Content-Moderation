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

from dataloader import MultimodalDataset, Modality
from models.callbacks import PrintCallback
from models.text_baseline_model import TextBaselineMMFNDModel
from models.text_image_resnet_model import TextImageResnetMMFNDModel
from models.text_image_resnet_dialogue_summarization_model import TextImageResnetDialogueSummarizationMMFNDModel

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 to turn off multiprocessing
# If not specified by --num_cpus command-line arg or in config file, defaults
# to the following
DEFAULT_NUM_CPUS = 0
# torch.multiprocessing.set_start_method('spawn')

DATA_PATH = "./data/Fakeddit"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768
DEFAULT_GPUS = [0, 1]

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    # TODO rm these first two
    # parser.add_argument("--train", action="store_true", help="Running on training data")
    # parser.add_argument("--test", action="store_true", help="Running on test (evaluation) data")
    parser.add_argument("--argument_graph", action="store_true", help="For training a model using dialogue argument graphs")
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
    parser.add_argument("--modality", type=str, default=None, help="text | image | text-image | text-image-dialogue")
    parser.add_argument("--num_classes", type=int, default=None, help="2 | 3 | 6")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--dropout_p", type=float, default=None)
    parser.add_argument("--fusion_output_size", type=int, default=None, help="Dimension after multi-modal embeddings fusion")
    parser.add_argument("--text_embedder", type=str, default=None, help="all-mpnet-base-v2 | all-distilroberta-v1")
    parser.add_argument("--image_encoder", type=str, default=None, help="resnet | dino")
    parser.add_argument("--dialogue_summarization_model", type=str, default=None, help="(Does NOT use in-house dialogue summarization) None=Transformers.Pipeline default i.e. sshleifer/distilbart-cnn-12-6 | bart-large-cnn | t5-small | t5-base | t5-large")
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--preprocessed_train_dataframe_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--preprocessed_test_dataframe_path", type=str, default=None)
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

    # Defaults specified here, if not specified by command-line arg or config
    if not args.model: args.model = config.get("model", "text_image_resnet_model")
    if not args.modality: args.modality = config.get("modality", "text-image")
    if not args.num_classes: args.num_classes = config.get("num_classes", 2)
    if not args.batch_size: args.batch_size = config.get("batch_size", 32)
    if not args.learning_rate: args.learning_rate = config.get("learning_rate", 1e-4)
    if not args.num_epochs: args.num_epochs = config.get("num_epochs", 10)
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
    if not args.dialogue_summarization_model:
        args.dialogue_summarization_model = config.get("dialogue_summarization_model", "bart-large-cnn")
    if not args.train_data_path:
        args.train_data_path = config.get("train_data_path", os.path.join(DATA_PATH, "multimodal_train_" + str(TRAIN_DATA_SIZE) + ".tsv"))
    if not args.preprocessed_train_dataframe_path:
        args.preprocessed_train_dataframe_path = config.get("preprocessed_train_dataframe_path", None)
    if not args.test_data_path:
        args.test_data_path = config.get("test_data_path", os.path.join(DATA_PATH, "multimodal_test_" + str(TEST_DATA_SIZE) + ".tsv"))
    if not args.preprocessed_test_dataframe_path:
        args.preprocessed_test_dataframe_path = config.get("preprocessed_test_dataframe_path", None)

    print("Running training with the following configuration...")
    print(f"model: {args.model}")
    print(f"modality: {args.modality}")
    print(f"num_classes: {args.num_classes}")
    print(f"batch_size: {args.batch_size}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"dropout_p: {args.dropout_p}")
    print(f"fusion_output_size: {args.fusion_output_size}")
    print(f"gpus: {args.gpus}")
    print(f"num_cpus: {args.num_cpus}")
    print(f"text_embedder: {args.text_embedder}")
    print(f"image_encoder: {args.image_encoder}")
    print(f"dialogue_summarization_model: {args.dialogue_summarization_model}")
    print(f"train_data_path: {args.train_data_path}")
    print(f"preprocessed_train_dataframe_path: {args.preprocessed_train_dataframe_path}")
    print(f"test_data_path: {args.test_data_path}")
    print(f"preprocessed_test_dataframe_path: {args.preprocessed_test_dataframe_path}")

    if args.only_check_args:
        quit()

    print("\nStarting training...")

    hparams = {
        # Used by pl.LightningModule
        "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
        "num_classes": args.num_classes,
        "learning_rate": args.learning_rate,
        "dropout_p": args.dropout_p,
        "fusion_output_size": args.fusion_output_size,

        # For logging
        "model": args.model,
        "modality": args.modality,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "text_embedder": args.text_embedder,
        "image_encoder": args.image_encoder,
        "dialogue_summarization_model": args.dialogue_summarization_model,
        "train_data_path": args.train_data_path,
        "preprocessed_train_dataframe_path": args.preprocessed_train_dataframe_path,
        "test_data_path": args.test_data_path,
        "preprocessed_test_dataframe_path": args.preprocessed_test_dataframe_path,
    }

    model = None
    text_embedder = SentenceTransformer(args.text_embedder)
    image_transform = None

    if args.model == "text_baseline_model":
        model = TextBaselineMMFNDModel(hparams)
    elif args.model == "text_image_resnet_model":
        model = TextImageResnetMMFNDModel(hparams)
        image_transform = TextImageResnetMMFNDModel.build_image_transform()
    elif args.model == "text_image_resnet_dialogue_summarization_model":
        model = TextImageResnetDialogueSummarizationMMFNDModel(hparams)
        image_transform = TextImageResnetDialogueSummarizationMMFNDModel.build_image_transform()
    else:
        raise Exception("run_training.py: Must pass a valid --model name to train")

    print(text_embedder)
    print(image_transform)

    train_dataset = MultimodalDataset(
        from_preprocessed_dataframe=args.preprocessed_train_dataframe_path,
        data_path=args.train_data_path,
        modality=args.modality,
        text_embedder=text_embedder,
        image_transform=image_transform,
        summarization_model=args.dialogue_summarization_model,
        num_classes=args.num_classes
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
