"""
Run from root directory
```
python -m fusion.run_experiments
```
"""

# Reuse run() from root's ./run_experiments.py
from run_experiments import run

GPU_ID = 5

def train_mmhs_early_fusion():
    """ text: {roberta, mpnet} + image: resnet """
    run(f"python -m fusion.run_training --config configs/mmhs_fusion/mmhs__early_fusion__text_image_ocr__mpnet_resnet.yaml --gpus {GPU_ID}")
    run(f"python -m fusion.run_training --config configs/mmhs_fusion/mmhs__early_fusion__text_image_ocr__roberta_resnet.yaml --gpus {GPU_ID}")

def train_mmhs_low_rank_fusion():
    """ text: {roberta, mpnet} + image: resnet """
    run(f"python -m fusion.run_training --config configs/mmhs_fusion/mmhs__low_rank_fusion__text_image_ocr__mpnet_resnet.yaml --gpus {GPU_ID}")
    run(f"python -m fusion.run_training --config configs/mmhs_fusion/mmhs__low_rank_fusion__text_image_ocr__roberta_resnet.yaml --gpus {GPU_ID}")

if __name__ == "__main__":
    train_mmhs_early_fusion()
    train_mmhs_low_rank_fusion()
