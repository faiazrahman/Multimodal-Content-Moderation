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
from models.text_image_resnet_model import TextImageResnetMMFNDModel
from models.text_image_resnet_dialogue_summarization_model import TextImageResnetDialogueSummarizationMMFNDModel

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

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

def get_checkpoint_filename_from_dir(path: str):
    """
    Gets the final checkpoint for the trained model, in the
    lightning_logs/version_{NUM}/checkpoints/ directory

    Final checkpoints are saved as `final-epoch={...}-step={...}.ckpt`
    In case that the final checkpoint was not saved (i.e. training was stopped
    early), we use the latest checkpoint with the largest epoch and step value
    i.e. `latest-epoch={...}-step={...}.ckpt`

    Note that this only works if checkpoints are saved in the formatting given;
    otherwise, it won't find the final or latest checkpoint
    """

    def compare_filenames(f1: str, f2: str):
        """ Custom comparator for latest checkpoint filenames, sorting ascending """
        epoch1 = int(f1.split("=")[1].split("-")[0])
        epoch2 = int(f2.split("=")[1].split("-")[0])
        if epoch1 < epoch2:
            return -1 # f1 comes before f2
        elif epoch1 > epoch2:
            return 1  # f1 comes after f2

        step1 = int(f1.split("=")[2].split(".")[0])
        step2 = int(f2.split("=")[2].split(".")[0])
        if step1 < step2:
            return -1 # f1 comes before f2
        elif step1 > step2:
            return 1  # f1 comes after f2
        else:
            return 0

    all_files = os.listdir(path)
    final_checkpoints = [filename for filename in all_files if filename.startswith("final")]
    if len(final_checkpoints) > 0:
        if len(final_checkpoints) == 1:
            return final_checkpoints[0]
        else:
            return sorted(
                final_checkpoints,
                key=cmp_to_key(compare_filenames),
                reverse=True
            )[0]

    # No final checkpoint, so use latest
    latest_checkpoints = [filename for filename in all_files if filename.startswith("latest")]
    if len(latest_checkpoints) > 0:
        return sorted(
            latest_checkpoints,
            key=cmp_to_key(compare_filenames),
            reverse=True
        )[0]

    return os.listdir(path)[0]

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_house_dialogue_summ", action="store_true", help="For evaluating a model that used in-house dialogue summarization data")
    parser.add_argument("--argument_graph", action="store_true", help="For evaluating a model that used dialogue argument graphs")
    parser.add_argument("--only_check_args", action="store_true", help="(Only for testing) Stops script after printing out args; doesn't actually run")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")

    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    parser.add_argument("--num_cpus", type=int, default=None, help="0 for no multi-processing, 24 on Yale Tangra server, 40 on Yale Ziva server")
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
    else:
        raise Exception("You must pass a config filename to --config to run evaluation; this must match the experiment configuration used for training the model")

    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)
    if not args.num_cpus: args.num_cpus = config.get("num_cpus", DEFAULT_NUM_CPUS)

    args.model = config.get("model", "text_image_resnet_model")
    args.modality = config.get("modality", "text-image")
    args.num_classes = config.get("num_classes", 2)
    args.batch_size = config.get("batch_size", 32)
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

    print("Running evaluation for a model with the following configuration...")
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

    if args.model == "text_image_resnet_model":
        model = TextImageResnetMMFNDModel.load_from_checkpoint(checkpoint_path)
        image_transform = TextImageResnetMMFNDModel.build_image_transform()
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
