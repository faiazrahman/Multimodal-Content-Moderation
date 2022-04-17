import os
import sys
import subprocess
from datetime import datetime

LOGS_DIR = "logs"

def run(command: str):
    """
    Runs given command and redirects output to a file named after the config
    and the timestamp at which it was run
    """

    before_config, keyword, after_config = command.partition("--config") # Gets configs/config_name.yaml ...
    before_slash, keyword, after_slash = after_config.strip().partition("configs/") # Gets config_name.yaml ...
    config_name = after_slash.split(".")[0] # Gets just config_name

    timestamp = str(datetime.now()).split(".")[0] # Removes decimal seconds
    log_filename = (config_name + "-" + timestamp).replace(" ", "-")
    log_filepath = os.path.join(LOGS_DIR, log_filename) + ".log"
    print(command + f" > {log_filepath}")

    subprocess.call(command + f" > {log_filepath}", shell=True)
    print(command + "\nCompleted!")

if __name__ == "__main__":

    # subprocess.call("python run_training.py --config configs/sampled_text_image_dialogue__2_class__mpnet_resnet_bart.yaml", shell=True)
    # run("python run_training.py --only_check_args --config configs/sampled_text_image_dialogue__2_class__mpnet_resnet_bart.yaml")

    run("python run_training.py --config configs/text_image__3_class__mpnet_resnet.yaml")
    run("python run_training.py --config configs/text_image__6_class__mpnet_resnet.yaml")
    run("python run_training.py --config configs/text_image_dialogue__3_class__mpnet_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image_dialogue__6_class__mpnet_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image__2_class__roberta_resnet.yaml")
    run("python run_training.py --config configs/text_image__3_class__roberta_resnet.yaml")
    run("python run_training.py --config configs/text_image__6_class__roberta_resnet.yaml")
    run("python run_training.py --config configs/text_image_dialogue__2_class__roberta_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image_dialogue__3_class__roberta_resnet_bart.yaml")
    run("python run_training.py --config configs/text_image_dialogue__6_class__roberta_resnet_bart.yaml")
