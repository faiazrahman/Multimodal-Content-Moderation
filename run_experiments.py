import os
import sys
import subprocess
from datetime import datetime

GPU_ID = 5  # Can specify multiple as a comma-separated string, e.g. "0,1"
LOGS_DIR = "logs"

def run(command: str, log_prefix: str = ""):
    """
    Runs given command and redirects output to a file named after the config
    and the timestamp at which it was run
    """

    config_name = "unnamed"
    if "--config" in command:
        before_config, keyword, after_config = command.partition("--config") # Gets configs/config_name.yaml ...
        before_slash, keyword, after_slash = after_config.strip().partition("configs/") # Gets config_name.yaml ...
        # For configs inside a subdirectory, there are two slashes, so repeat
        if "/" in after_slash:
            before_slash, keyword, after_slash = after_slash.strip().partition("/")
        config_name = after_slash.split(".")[0] # Gets just config_name

    # When running evaluation with only --trained_model_version (and no config)
    # we can pass in the name we want for the log
    if log_prefix != "":
        if config_name == "unnamed":
            config_name = log_prefix
        else:
            config_name = log_prefix + "__" + config_name

    if "evaluation" in command:
        # Name evaluation logs with prefix
        config_name = "eval__" + config_name

    timestamp = str(datetime.now()).split(".")[0] # Removes decimal seconds
    log_filename = (config_name + "-" + timestamp).replace(" ", "-")
    log_filepath = os.path.join(LOGS_DIR, log_filename) + ".log"
    print(command + f" > {log_filepath}")

    subprocess.call(command + f" > {log_filepath}", shell=True)
    print(command + "\nCompleted!")

def train_text_baseline_roberta_mpnet():
    """ text: {roberta, mpnet} """
    run(f"python run_training.py --config configs/text__2_class__roberta.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text__3_class__roberta.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text__6_class__roberta.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text__2_class__mpnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text__3_class__mpnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text__6_class__mpnet.yaml --gpus {GPU_ID}")

def eval_text_baseline_roberta_mpnet():
    # NOTE: Make sure you specify the trained model version in the config (or pass it as --trained_model_version)
    trained_model_version_numbers = [i for i in range(318, 324)] # [318, 323] inclusive
    for version_number in trained_model_version_numbers:
        run(f"python run_evaluation.py --trained_model_version {version_number} --gpus {GPU_ID}",
            log_prefix="text_baseline")

def train_image_baseline_resnet():
    """ image: resnet """
    run(f"python run_training.py --config configs/image__2_class__resnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/image__3_class__resnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/image__6_class__resnet.yaml --gpus {GPU_ID}")

def eval_image_baseline_resnet():
    # NOTE: Make sure you specify the trained model version in the config (or pass it as --trained_model_version)
    trained_model_version_numbers = [325, 326, 328]
    for version_number in trained_model_version_numbers:
        run(f"python run_evaluation.py --trained_model_version {version_number} --gpus {GPU_ID}",
            log_prefix="image_baseline")

def train_roberta_mpnet_resnet_bart_ranksum():
    """
    text: {roberta, mpnet} + image: resnet
    text: {roberta, mpnet} + image: resnet + dialogue: RankSum-BART
    """
    run(f"python run_training.py --config configs/text_image__2_class__mpnet_resnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__3_class__mpnet_resnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__6_class__mpnet_resnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__2_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__3_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__6_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__2_class__roberta_resnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__3_class__roberta_resnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__6_class__roberta_resnet.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__2_class__roberta_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__3_class__roberta_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__6_class__roberta_resnet_bart.yaml --gpus {GPU_ID}")

def train_roberta_mpnet_dino():
    """
    text: {roberta, mpnet} + image: dino
    """
    run(f"python run_training.py --config configs/text_image__2_class__mpnet_dino.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__3_class__mpnet_dino.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__6_class__mpnet_dino.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__2_class__roberta_dino.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__3_class__roberta_dino.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image__6_class__roberta_dino.yaml --gpus {GPU_ID}")

def train_roberta_mpnet_dino_bart_ranksum():
    """
    text: {roberta, mpnet} + image: dino + dialogue: RankSum-BART
    """
    run(f"python run_training.py --config configs/text_image_dialogue__2_class__mpnet_dino_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__3_class__mpnet_dino_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__6_class__mpnet_dino_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__2_class__roberta_dino_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__3_class__roberta_dino_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/text_image_dialogue__6_class__roberta_dino_bart.yaml --gpus {GPU_ID}")

def train_graphlin_mpnet_resnet_bart():
    """ text: mpnet + image: resnet + dialogue: GraphLin-BART """
    run(f"python run_training.py --config configs/graphlin__text_image_dialogue__2_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/graphlin__text_image_dialogue__3_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/graphlin__text_image_dialogue__6_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")

def train_argsum_mpnet_resnet_bart():
    """ text: mpnet + image: resnet + dialogue: ArgSum-BART """
    run(f"python run_training.py --config configs/argsum__text_image_dialogue__2_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/argsum__text_image_dialogue__3_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")
    run(f"python run_training.py --config configs/argsum__text_image_dialogue__6_class__mpnet_resnet_bart.yaml --gpus {GPU_ID}")

if __name__ == "__main__":

    # subprocess.call("python run_training.py --config configs/sampled_text_image_dialogue__2_class__mpnet_resnet_bart.yaml", shell=True)
    # run("python run_training.py --only_check_args --config configs/sampled_text_image_dialogue__2_class__mpnet_resnet_bart.yaml")

    # train_graphlin_mpnet_resnet_bart()
    # train_argsum_mpnet_resnet_bart()

    # TESTING OUT TEXT BASELINE MODEL
    # run(f"python run_training.py --config configs/text__2_class__mpnet.yaml --gpus {GPU_ID}")

    # train_text_baseline_roberta_mpnet()
    # train_image_baseline_resnet()

    # eval_text_baseline_roberta_mpnet()
    # eval_image_baseline_resnet()

    # IN PROGRESS
    # train_roberta_mpnet_dino()

    # IN PROGRESS
    train_roberta_mpnet_dino_bart_ranksum()
