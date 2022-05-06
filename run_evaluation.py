import os
import logging
import argparse
from functools import cmp_to_key

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
from models.image_resnet_baseline_model import ImageResnetBaselineMMFNDModel
from models.text_image_resnet_model import TextImageResnetMMFNDModel
from models.text_image_dino_model import TextImageDinoMMFNDModel
from models.text_image_resnet_dialogue_summarization_model import TextImageResnetDialogueSummarizationMMFNDModel
from utils import get_checkpoint_filename_from_dir

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 to turn off multiprocessing
# If not specified by --num_cpus command-line arg or in config file, defaults
# to the following
DEFAULT_NUM_CPUS = 0
# torch.multiprocessing.set_start_method('spawn')

PL_ASSETS_PATH = "./lightning_logs"
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
    parser.add_argument("--argument_graph", action="store_true", help="For evaluating a model that used dialogue argument graphs")
    parser.add_argument("--only_check_args", action="store_true", help="(Only for testing) Stops script after printing out args; doesn't actually run")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")

    parser.add_argument("--trained_model_version", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    parser.add_argument("--num_cpus", type=int, default=None, help="0 for no multi-processing, 24 on Yale Tangra server, 40 on Yale Ziva server")
    args = parser.parse_args()

    # NOTE: We allow passing just a trained model version number, since we can
    # get the hyperparameters from its `lightning_logs/version_*/hparams.yaml` file
    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
    else:
        if not args.trained_model_version:
            raise Exception("You must either pass a config filename to --config (which must match the experiment configuration used for training the model) OR pass a --trained_model_version to run evaluation")

    if not args.batch_size: args.batch_size = config.get("batch_size", 32)
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
        args.modality = hparams_config.get("modality", None)
        args.num_classes = hparams_config.get("num_classes", None)
        original_batch_size = hparams_config.get("batch_size", None)
        args.learning_rate = hparams_config.get("learning_rate", None)
        args.num_epochs = hparams_config.get("num_epochs", None)
        args.dropout_p = hparams_config.get("dropout_p", None)
        args.fusion_output_size = hparams_config.get("fusion_output_size", None)
        args.text_embedder = hparams_config.get("text_embedder", None)
        args.image_encoder = hparams_config.get("image_encoder", None)
        args.dialogue_summarization_model = hparams_config.get("dialogue_summarization_model", None)
        args.test_data_path = hparams_config.get("test_data_path", None)
        args.preprocessed_test_dataframe_path = hparams_config.get("preprocessed_test_dataframe_path", None)
    else:
        # Otherwise, load the hparams from the specified config file
        args.model = config.get("model", "text_image_resnet_model")
        args.modality = config.get("modality", "text-image")
        args.num_classes = config.get("num_classes", 2)
        args.batch_size = config.get("batch_size", 32)
        original_batch_size = args.batch_size
        args.learning_rate = config.get("learning_rate", 1e-4)
        args.num_epochs = config.get("num_epochs", 10)
        args.dropout_p = config.get("dropout_p", 0.1)
        args.fusion_output_size = config.get("fusion_output_size", 512)
        args.text_embedder = config.get("text_embedder", "all-mpnet-base-v2")
        args.image_encoder = config.get("image_encoder", "resnet")
        args.dialogue_summarization_model = config.get("dialogue_summarization_model", "bart-large-cnn")

        args.trained_model_version = config.get("trained_model_version", None)
        args.trained_model_path = config.get("trained_model_path", None)
        args.test_data_path = config.get("test_data_path", os.path.join(DATA_PATH, "multimodal_test_" + str(TEST_DATA_SIZE) + ".tsv"))
        args.preprocessed_test_dataframe_path = config.get("preprocessed_test_dataframe_path", None)

    print(f"Running evaluation with batch_size={args.batch_size} for a model with the following configuration...")
    print(f"model: {args.model}")
    print(f"modality: {args.modality}")
    print(f"num_classes: {args.num_classes}")
    print(f"batch_size: {original_batch_size}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"dropout_p: {args.dropout_p}")
    print(f"fusion_output_size: {args.fusion_output_size}")
    print(f"gpus: {args.gpus}")
    print(f"num_cpus: {args.num_cpus}")
    print(f"text_embedder: {args.text_embedder}")
    print(f"image_encoder: {args.image_encoder}")
    print(f"dialogue_summarization_model: {args.dialogue_summarization_model}")
    print(f"test_data_path: {args.test_data_path}")
    print(f"preprocessed_test_dataframe_path: {args.preprocessed_test_dataframe_path}")

    if args.only_check_args:
        quit()

    print("\nStarting evaluation...")
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
    text_embedder = SentenceTransformer(args.text_embedder)
    image_transform = None

    if args.model == "text_baseline_model":
        model = TextBaselineMMFNDModel.load_from_checkpoint(checkpoint_path)
    elif args.model == "image_resnet_baseline_model":
        model = ImageResnetBaselineMMFNDModel.load_from_checkpoint(checkpoint_path)
        image_transform = ImageResnetBaselineMMFNDModel.build_image_transform()
    elif args.model == "text_image_resnet_model":
        model = TextImageResnetMMFNDModel.load_from_checkpoint(checkpoint_path)
        image_transform = TextImageResnetMMFNDModel.build_image_transform()
    elif args.model == "text_image_dino_model":
        model = TextImageDinoMMFNDModel.load_from_checkpoint(checkpoint_path)
        image_transform = TextImageDinoMMFNDModel.build_image_transform()
    elif args.model == "text_image_resnet_dialogue_summarization_model":
        model = TextImageResnetDialogueSummarizationMMFNDModel.load_from_checkpoint(checkpoint_path)
        image_transform = TextImageResnetDialogueSummarizationMMFNDModel.build_image_transform()
    else:
        raise Exception("run_evaluation.py: Must pass a valid --model name to evaluate")

    print(text_embedder)
    print(image_transform)

    test_dataset = MultimodalDataset(
        from_preprocessed_dataframe=args.preprocessed_test_dataframe_path,
        data_path=args.test_data_path,
        modality=args.modality,
        text_embedder=text_embedder,
        image_transform=image_transform,
        summarization_model=args.dialogue_summarization_model,
        num_classes=args.num_classes
    )
    logging.info("Test dataset size: {}".format(len(test_dataset)))
    logging.info(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus
    )
    logging.info(test_loader)

    print(model)

    callbacks = [
        PrintCallback(),
        TQDMProgressBar(refresh_rate=10)
    ]

    trainer = None
    if torch.cuda.is_available():
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        trainer = pl.Trainer(
            gpus=args.gpus,
            strategy="dp",
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer(
            callbacks=callbacks
        )
    logging.info(trainer)

    trainer.test(model, dataloaders=test_loader)
    # pl.LightningModule has some issues displaying the results automatically
    # As a workaround, we can store the result logs as an attribute of the
    # class instance and display them manually at the end of testing
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
    results = model.test_results

    print(args.test_data_path)
    print(checkpoint_path)
    print(results)
    logging.info(args.test_data_path)
    logging.info(checkpoint_path)
    logging.info(results)
