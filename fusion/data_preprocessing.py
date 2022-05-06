"""
Run (from root)
```
python -m fusion.data_preprocessing [--args]
```
- This runs data preprocessing for the MMHS150K dataset
- It creates a dataframe with the following columns
    TODO
"""

import os
import json
import logging
from collections import Counter
from typing import List

import pandas as pd

from PIL import Image

DATA_PATH = "./data"
MMHS_DATA_PATH = os.path.join(DATA_PATH, "MMHS150K")

# The dict (json) maps IDs to tweet_url, labels, img_url, tweet_text, labels
MMHS_TWEETS_DICT_PATH = os.path.join(MMHS_DATA_PATH, "MMHS150K_GT.json")

# These directories store .jpg and .json files (see data/MMHS150K/README.txt)
MMHS_IMAGES_DIR = os.path.join(MMHS_DATA_PATH, "img_resized")
MMHS_IMAGES_OCR_TEXT_DIR = os.path.join(MMHS_DATA_PATH, "img_txt")

# Data splits are stored as text files, with one ID per line
MMHS_TRAIN_IDS_FILE = os.path.join(MMHS_DATA_PATH, "splits/train_ids.txt")
MMHS_TEST_IDS_FILE = os.path.join(MMHS_DATA_PATH, "splits/test_ids.txt")
MMHS_VAL_IDS_FILE = os.path.join(MMHS_DATA_PATH, "splits/val_ids.txt")

MMHS_TRAIN_DATAFRAME_PATH = os.path.join(MMHS_DATA_PATH, "mmhs_train_dataframe.pkl")
MMHS_TEST_DATAFRAME_PATH = os.path.join(MMHS_DATA_PATH, "mmhs_test_dataframe.pkl")
MMHS_VAL_DATAFRAME_PATH = os.path.join(MMHS_DATA_PATH, "mmhs_val_dataframe.pkl")

logging.basicConfig(level=logging.INFO)

def preprocess_mmhs_data(
    data_ids_file: str,
    split_type: str = "val" # "train" | "test" | "val"
):
    print(f"Cleaning MMHS150K data for {split_type}...")
    print(f"> Data IDs extracted from: {data_ids_file}")
    data_ids: List[str] = list()
    with open(data_ids_file) as file:
        data_ids = file.readlines()
        data_ids = [line.rstrip() for line in data_ids]

    tweets_dict = dict()
    with open(MMHS_TWEETS_DICT_PATH) as file:
        tweets_dict = json.load(file)

    # print(len(data_ids))
    # print(data_ids[0])
    # print(type(data_ids[0]))

    # print(len(tweets_dict))
    # print(type(tweets_dict))
    # print(tweets_dict[list(tweets_dict.keys())[0]])
    # print(type(tweets_dict[list(tweets_dict.keys())[0]]))

    def get_majority_label(row):
        """
        Returns the majority label (0-5)
        - The `labels` value field in `tweets_dict` contains 3 labels (from 3
          different annotators), so we select the label with majority

        Returns -1 if no majority
        """
        tweet_id = row['id']
        try:
            label_counts = Counter(tweets_dict[tweet_id]['labels'])
            majority_label, count = label_counts.most_common(1)[0]
            if count > 1:
                return majority_label
            else:
                return -1
        except:
            return -1

    def get_text(row):
        """ Returns the tweet_id's text (str) """
        tweet_id = row['id']
        try:
            text = str(tweets_dict[tweet_id]['tweet_text'])
            return text
        except:
            return ""

    def image_exists(row):
        """
        Returns true if the image exists in the images dir and can be opened;
        otherwise false
        """
        tweet_id = row['id']
        image_path = os.path.join(MMHS_IMAGES_DIR, tweet_id + ".jpg")
        if not os.path.exists(image_path):
            return False

        try:
            image = Image.open(image_path)
            image.verify()
            image.close()
            return True
        except:
            return False

    def get_image_ocr_text(row):
        """
        Returns the tweet_id's image OCR text (str)

        If the image had no OCR text, returns the string "[none]"
        """
        tweet_id = row['id']
        image_ocr_text_json = os.path.join(MMHS_IMAGES_OCR_TEXT_DIR, tweet_id + ".json")
        try:
            with open(image_ocr_text_json) as file:
                image_ocr_text_dict = json.load(file)
                image_ocr_text = image_ocr_text_dict["img_text"]
                return image_ocr_text
        except:
            return "[none]"

    df = pd.DataFrame(data_ids, columns=['id'])

    # Get majority label, text, and image OCR text
    df['label'] = df.apply(lambda row: get_majority_label(row), axis=1)
    df['text'] = df.apply(lambda row: get_text(row), axis=1)
    df['image_ocr_text'] = df.apply(lambda row: get_image_ocr_text(row), axis=1)

    # Check if image exists
    df['image_exists'] = df.apply(lambda row: image_exists(row), axis=1)

    # Drop all rows with any invalid values (i.e. label -1, no text, image does
    # not exist), and drop any rows with a NaN
    # Note that not all images have OCR text, but we still keep them
    df = df[df['label'] != -1]
    df = df[df['text'] != ""]
    df = df[df['image_exists'] == True].drop('image_exists', axis=1)
    df = df.dropna()

    print(f"Saving dataframe...")
    save_path = None
    if split_type == "train":
        save_path = MMHS_TRAIN_DATAFRAME_PATH
    elif split_type == "test":
        save_path = MMHS_TEST_DATAFRAME_PATH
    elif split_type == "val":
        save_path = MMHS_VAL_DATAFRAME_PATH
    else:
        raise ValueError("preprocess_mmhs_data() received invalid split_type; should be train, test, or val")

    df.to_pickle(save_path)
    print(f"> Saved to {save_path}")

if __name__ == "__main__":
    preprocess_mmhs_data(MMHS_TRAIN_IDS_FILE, "train")
    preprocess_mmhs_data(MMHS_TEST_IDS_FILE, "test")
    preprocess_mmhs_data(MMHS_VAL_IDS_FILE, "val")
