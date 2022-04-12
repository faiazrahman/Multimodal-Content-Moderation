"""
python -m scripts.test_train_script_args_config
"""

import subprocess

if __name__ == "__main__":

    print("Defaults")
    output_bytes = subprocess.check_output("python run_training.py --only_check_args", shell=True)
    output = output_bytes.decode("utf-8")
    print(output)

    print("Using a config with a couple not specified (should go to default)")
    config_path = "configs/test_config.yaml"
    output_bytes = subprocess.check_output(f"python run_training.py --only_check_args --config {config_path}", shell=True)
    output = output_bytes.decode("utf-8")
    print(output)

    print("Using a config and overriding some with command-line args")
    config_path = "configs/test_config.yaml"
    output_bytes = subprocess.check_output(f"python run_training.py --only_check_args --config {config_path} --batch_size 444444 --gpus 4,4,4,4,4", shell=True)
    output = output_bytes.decode("utf-8")
    print(output)
