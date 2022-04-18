import sys
import os
import logging

import pandas as pd
from tqdm import tqdm
from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

DATA_PATH = "../data/ArgumentativeUnitClassification"
AUC_DATA_FILE = "all_auc_data.tsv"
AUC_DATA_PATH = os.path.join(DATA_PATH, AUC_DATA_FILE)
AUC_DATAFRAME_FILE = "auc_dataframe.pkl"
AUC_DATAFRAME_PATH = os.path.join(DATA_PATH, AUC_DATAFRAME_FILE)

class ArgumentativeUnitClassificationDataset(Dataset):
    """
    torch.utils.data.Dataset for AUC data of the format (text, label), where
    the labels are as follows:
        0   Non-argumentative unit
        1   Claim
        2   Premise

    `__getitem__()` returns a dict containing "text" and "label" keys
    """

    def __init__(
        self,
        from_dataframe_pkl_path: str = AUC_DATAFRAME_PATH,
    ):
        df = None
        if os.path.exists(from_dataframe_pkl_path):
            df = pd.read_pickle(from_dataframe_pkl_path)
        else:
            raise Exception("AUC dataframe does not exist\n\
                             Run argument_graphs/data_preprocessing.py")
        self.data_frame = df

    def __len__(self):
        """ Returns the size of the dataset; called as len(dataset) """
        return len(self.data_frame.index)

    def __getitem__(self, idx: int):
        """ Returns a dict containing "text" and "label" """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.data_frame.loc[idx, 'text']
        label = torch.Tensor(
            [self.data_frame.loc[idx, 'label']]
        ).long().squeeze()

        item = {
            "text": text,
            "label": label,
        }
        return item

if __name__ == "__main__":
    print("WARNING: This is only for testing argument_graphs/dataloader.py")
    print("\t This file should otherwise not be run directly")

    dataset = ArgumentativeUnitClassificationDataset()
    print(type(dataset))
    print(f"Dataset size: {len(dataset)}\n")
    for idx, item in enumerate(dataset):
        text, label = item['text'], item['label']
        print(text); print(label); print("")
        if idx > 10: break
