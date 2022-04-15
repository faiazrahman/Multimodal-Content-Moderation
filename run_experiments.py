import subprocess

if __name__ == "__main__":
    subprocess.call("python run_training.py --config configs/sampled_text_image_dialogue__2_class__mpnet_resnet_bart.yaml", shell=True)

