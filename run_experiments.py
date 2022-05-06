import os
import sys
import subprocess
from datetime import datetime

GPU_ID = 2  # Can specify multiple as a comma-separated string, e.g. "0,1"
LOGS_DIR = "logs"

def run(command: str):
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
    run("python run_training.py --config configs/text__2_class__roberta.yaml")
    run("python run_training.py --config configs/text__3_class__roberta.yaml")
    run("python run_training.py --config configs/text__6_class__roberta.yaml")
    run("python run_training.py --config configs/text__2_class__mpnet.yaml")
    run("python run_training.py --config configs/text__3_class__mpnet.yaml")
    run("python run_training.py --config configs/text__6_class__mpnet.yaml")

def train_image_baseline_resnet():
    """ image: resnet """
    run("python run_training.py --config configs/image__2_class__resnet.yaml")
    run("python run_training.py --config configs/image__3_class__resnet.yaml")
    run("python run_training.py --config configs/image__6_class__resnet.yaml")

def train_roberta_mpnet_resnet_bart_ranksum():
    """
    text: {roberta, mpnet} + image: resnet
    text: {roberta, mpnet} + image: resnet + dialogue: RankSum-BART
    """
    run("python run_training.py --config configs/text_image__2_class__mpnet_resnet.yaml")
    run("python run_training.py --config configs/text_image__3_class__mpnet_resnet.yaml")
    run("python run_training.py --config configs/text_image__6_class__mpnet_resnet.yaml")
    run("python run_training.py --config configs/text_image_dialogue__2_class__mpnet_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image_dialogue__3_class__mpnet_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image_dialogue__6_class__mpnet_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image__2_class__roberta_resnet.yaml")
    run("python run_training.py --config configs/text_image__3_class__roberta_resnet.yaml")
    run("python run_training.py --config configs/text_image__6_class__roberta_resnet.yaml")
    run("python run_training.py --config configs/text_image_dialogue__2_class__roberta_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image_dialogue__3_class__roberta_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image_dialogue__6_class__roberta_resnet_bart.yaml")

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

    train_graphlin_mpnet_resnet_bart()
    train_argsum_mpnet_resnet_bart()
