"""
Various utilities for running experiments and evaluation
"""

import os
from functools import cmp_to_key

PL_ASSETS_PATH = "./lightning_logs"

def get_checkpoint_filename_from_dir(path: str) -> str:
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

def get_checkpoint_path_from_trained_model_version(
    trained_model_version: int
) -> str:
    assets_version = "version_" + str(trained_model_version)
    checkpoint_path = os.path.join(PL_ASSETS_PATH, assets_version, "checkpoints")
    checkpoint_filename = get_checkpoint_filename_from_dir(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, checkpoint_filename)
    return checkpoint_path
