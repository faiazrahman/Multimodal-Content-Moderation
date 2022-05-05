import os
import logging
import argparse
from typing import List

import pandas as pd

DATA_PATH = "../data/Fakeddit"
# TRAIN_PREPROCESSED_DATAFRAME_PATH = os.path.join(DATA_PATH, "train__text_image_dialogue__dataframe.pkl")
# TEST_PREPROCESSED_DATAFRAME_PATH = os.path.join(DATA_PATH, "test__text_image_dialogue__dataframe.pkl")
# TRAIN_DIALOGUE_DATAFRAME_PATH = os.path.join(DATA_PATH, "train__dialogue_dataframe.pkl")
# TEST_DIALOGUE_DATAFRAME_PATH = os.path.join(DATA_PATH, "test__dialogue_dataframe.pkl")
TRAIN_PREPROCESSED_DATAFRAME_PATH = os.path.join(DATA_PATH, "sampled_train__text_image_dialogue__dataframe.pkl")
TEST_PREPROCESSED_DATAFRAME_PATH = os.path.join(DATA_PATH, "sampled_test__text_image_dialogue__dataframe.pkl")
TRAIN_DIALOGUE_DATAFRAME_PATH = os.path.join(DATA_PATH, "sampled_train__dialogue_dataframe.pkl")
TEST_DIALOGUE_DATAFRAME_PATH = os.path.join(DATA_PATH, "sampled_test__dialogue_dataframe.pkl")

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

def print_list(lst: List[str]):
    for element in lst:
        print(">" + element)

def display_items_with_dialogue(df_path: str):
    """ Takes in preprocessed dataframe path """

    def has_dialogue(row):
        return row['comment_summary'] != "none"

    df = pd.read_pickle(df_path)
    df['has_dialogue'] = df.apply(lambda row: has_dialogue(row), axis=1)
    df = df[df['has_dialogue'] == True].drop('has_dialogue', axis=1)
    print(df)
    print(df.columns)

def view_dialogues(df_path: str):
    """ Takes in dialogue dataframe path """
    df = pd.read_pickle(df_path)
    print(df)
    print(df.columns)
    text_ids = df['submission_id'].unique()

    for indx, text_id in enumerate(text_ids):
        # Group comments by post id
        all_comments = df[df['submission_id'] == text_id]
        all_comments.sort_values(by=['ups'], ascending=False)
        all_comments = list(all_comments['body'])

        print("=" * 80)
        print(f"post id:{text_id}, {len(all_comments)} comments\n")
        print_list(all_comments)
        print("=" * 80)

        if indx > 20: break

if __name__ == "__main__":
    display_items_with_dialogue(TRAIN_PREPROCESSED_DATAFRAME_PATH)
    display_items_with_dialogue(TEST_PREPROCESSED_DATAFRAME_PATH)

    view_dialogues(TRAIN_DIALOGUE_DATAFRAME_PATH)
    view_dialogues(TEST_DIALOGUE_DATAFRAME_PATH)
