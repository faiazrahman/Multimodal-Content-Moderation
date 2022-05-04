"""
Run (from root)
```
python -m dialogue_summarization.data_preprocessing [--args]
```
- This runs data preprocessing for the SAMSum dataset
- It creates a dataframe with the following columns
    id (int): Identifier for the dialogue
    dialogue (List[str]): List of utterance strings for the dialogue
    summary (str): Reference summary
- The dataframe is saved as `samsum_{train, test, val}_dataframe.pkl` in the
  SAMSum data directory

Here is an example of the first five rows of the validation dataframe.
           id                                            summary                                           dialogue
0    13817023  A will go to the animal shelter tomorrow to ge...  [A: Hi Tom, are you busy tomorrow's afternoon?...
1    13716628  Emma and Rob love the advent calendar. Lauren ...  [Emma: I've just fallen in love with this adve...
2    13829420  Madison is pregnant but she doesn't want to ta...  [Jackie: Madison is pregnant, Jackie: but she ...
3    13819648        Marla found a pair of boxers under her bed.  [Marla: <file_photo>, Marla: look what I found...
4    13728448  Robert wants Fred to send him the address of t...  [Robert: Hey give me the address of this music...

```
python -m dialogue_summarization.data_preprocessing
```
"""

import sys
import os
import glob
import logging
import argparse

import pandas as pd

DATA_PATH = "./data"
SAMSUM_DATA_PATH = os.path.join(DATA_PATH, "SAMSum")
SAMSUM_TRAIN_DATA_PATH = os.path.join(SAMSUM_DATA_PATH, "train.json")
SAMSUM_TEST_DATA_PATH = os.path.join(SAMSUM_DATA_PATH, "test.json")
SAMSUM_VAL_DATA_PATH = os.path.join(SAMSUM_DATA_PATH, "val.json")

SAMSUM_TRAIN_DATAFRAME_PATH = os.path.join(SAMSUM_DATA_PATH, "samsum_train_dataframe.pkl")
SAMSUM_TEST_DATAFRAME_PATH = os.path.join(SAMSUM_DATA_PATH, "samsum_test_dataframe.pkl")
SAMSUM_VAL_DATAFRAME_PATH = os.path.join(SAMSUM_DATA_PATH, "samsum_val_dataframe.pkl")

logging.basicConfig(level=logging.DEBUG)

def preprocess_samsum_data(data_path: str):

    def clean_text(text):
        """ Removes carriage returns and splits into list of utterances """
        return text.replace("\r", "").split("\n")

    print(f"Cleaning SAMSum data for {data_path}...")
    df = pd.read_json(data_path)
    df['dialogue'] = df['dialogue'].apply(lambda text: clean_text(text))
    logging.debug(df)

    # Note that the original data files are "train.json", "test.json", and
    # "val.json", so we can use that to save to the appropriate dataframe
    # pickle filename
    print(f"Saving dataframe...")
    dataset_type = os.path.basename(data_path).replace(".json", "")
    save_path = None
    if dataset_type == "train":
        save_path = SAMSUM_TRAIN_DATAFRAME_PATH
    elif dataset_type == "test":
        save_path = SAMSUM_TEST_DATAFRAME_PATH
    elif dataset_type == "val":
        save_path = SAMSUM_VAL_DATAFRAME_PATH
    else:
        raise ValueError("preprocess_samsum_data")

    df.to_pickle(save_path)
    print(f"> Saved to {save_path}")

if __name__ == "__main__":
    preprocess_samsum_data(SAMSUM_TRAIN_DATA_PATH)
    preprocess_samsum_data(SAMSUM_TEST_DATA_PATH)
    preprocess_samsum_data(SAMSUM_VAL_DATA_PATH)
