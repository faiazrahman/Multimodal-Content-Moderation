"""
Generates an evenly-distributed sample of the Fakeddit dataset, for both
train data and test data

Notes
    Selects only multi-modal data examples with text, image, and dialogue
      modalities
    Downloads images for data examples if they are not already downloaded
"""

import os
import argparse
import pandas as pd
from PIL import Image
import urllib.request

TRAIN_DATA_DEFAULT = "../data/Fakeddit/multimodal_train.tsv"
TEST_DATA_DEFAULT = "../data/Fakeddit/multimodal_test_public.tsv"
TRAIN_SAMPLE_SIZE = 10000
TEST_SAMPLE_SIZE = 1000
DATA_DELIMITER = "\t"

DATA_PATH = "../data/Fakeddit"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
IMAGE_EXTENSION = ".jpg"
DIALOGUE_DATA_FILE = os.path.join(DATA_PATH, "all_comments.tsv")

DATA_ID_COL_INDEX = 5 # In the data, the post's id is the 6th column (i.e. indexed 5)
IMAGE_URL_COL_INDEX = 6 # In the data, the post's image_url is the 7th column

def generate_and_save_sample(data_path: str, sample_size: int) -> str:
    """ Returns filepath to saved sample """

    print(f"Running for {data_path}...")
    print(f"  Sample size: {sample_size}")

    def image_exists(item_id, image_url) -> bool:
        """
        Ensures that image exists (tries to download if initially not found)
        and can be opened properly
        """

        image_path = os.path.join(IMAGES_DIR, str(item_id) + IMAGE_EXTENSION)

        if not os.path.exists(image_path):
            # If image does not exist, try to download it
            try:
                # Download, then continue on to the Image.verify() check
                urllib.request.urlretrieve(image_url, image_path) # Save to expected image_path
            except:
                return False
            else:
                # Successfully downloaded
                print(f"Downloaded image for post id:{item_id}")

        try:
            image = Image.open(image_path)
            image.verify()
            image.close()
            return True
        except Exception:
            return False

    def has_dialogue_data(item_id, dialogue_submission_ids) -> bool:
        """ Ensures that this item has at least one comment """
        if item_id in dialogue_submission_ids:
            return True
        else:
            return False

    target_num_positive = sample_size // 2
    target_num_negative = sample_size - target_num_positive
    target_amounts_for_each_class = {
        "0": target_num_positive,
        "1": target_num_negative // 5,
        "2": target_num_negative // 5,
        "3": target_num_negative // 5,
        "4": target_num_negative // 5,
        "5": target_num_negative // 5,
    }
    curr_amounts_for_each_class = {
        "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
    }

    output_filename = data_path.replace(".tsv", "") + "_" + str(sample_size) + "_sampled.tsv"
    print(f"Creating sample and saving to to {output_filename}")

    print("Getting list of posts which have dialogue data...")
    dialogue_df = pd.read_csv(DIALOGUE_DATA_FILE, sep='\t')
    dialogue_submission_ids = dialogue_df["submission_id"].unique()

    print("Sampling...")
    curr_num_samples = 0
    on_header_line = True
    with open(output_filename, "w") as output_file:
        with open(data_path, "r") as data_file:
            for line in data_file:
                # Always copy over the first line (the header)
                if on_header_line:
                    output_file.write(line)
                    on_header_line = False
                    continue

                try:
                    # The 6-way label is the last column in the raw data
                    label = line.split(DATA_DELIMITER)[-1].strip()
                    if (label in target_amounts_for_each_class.keys()
                        and curr_amounts_for_each_class[label] < target_amounts_for_each_class[label]):
                        item_id = line.split(DATA_DELIMITER)[DATA_ID_COL_INDEX]
                        image_url = line.split(DATA_DELIMITER)[IMAGE_URL_COL_INDEX]
                        # We only add an example if it has image and dialogue data too
                        # Note: We check dialogue data first so we don't unnecessarily download images
                        # Note: None of the evaluation data for label 4 has images, so we allow it
                        if (label == "4" or
                            (has_dialogue_data(item_id, dialogue_submission_ids) and image_exists(item_id, image_url))):
                            output_file.write(line)
                            curr_amounts_for_each_class[label] += 1
                            curr_num_samples += 1
                except:
                    continue

                if curr_num_samples >= sample_size:
                    break

    # Verify even distribution of 6-way labels
    sample_df = pd.read_csv(output_filename, sep="\t", header=0)
    df = sample_df.groupby("2_way_label")["id"].nunique()
    print(df)
    df = sample_df.groupby("3_way_label")["id"].nunique()
    print(df)
    df = sample_df.groupby("6_way_label")["id"].nunique()
    print(df)

    return output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_DEFAULT)
    parser.add_argument("--test_data_path", type=str, default=TEST_DATA_DEFAULT)
    parser.add_argument("--train_sample_size", type=int, default=TRAIN_SAMPLE_SIZE)
    parser.add_argument("--test_sample_size", type=int, default=TEST_SAMPLE_SIZE)
    args = parser.parse_args()

    if args.train:
        generate_and_save_sample(args.train_data_path, args.train_sample_size)

    if args.test:
        generate_and_save_sample(args.test_data_path, args.test_sample_size)
