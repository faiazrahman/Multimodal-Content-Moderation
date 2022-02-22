"""
This script is a modified version of the original image_downloader.py script
provided by the Fakeddit dataset

https://github.com/entitize/Fakeddit

The changes are minor: We wrapped the urllib.request in a try-except to avoid
having errors stop the entire download process, and we avoided redownloading
images which have already been downloaded by checking if the image already
exists (e.g. if a previous run failed or only downloaded part of the data)

We also added the `data_filename` argument to argparse.

Note that the text data must be downloaded first; see the Fakeeddit GitHub repo
for instructions on how to download the text (and the comment dialogue) data
"""

import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np
import sys

parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')

parser.add_argument('type', type=str, help='train, validate, or test')
parser.add_argument("data_filename", type=str, default="", help=".tsv file storing text data; only the images corresponding to those examples will be downloaded")

args = parser.parse_args()

df = None
if args.data_filename == "":
    # Original from Fakeddit repo
    df = pd.read_csv(args.type, sep="\t")
else:
    # Modification to only download a subset of images for quick experiments
    df = pd.read_csv(args.data_filename, sep="\t")

df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

pbar = tqdm(total=len(df))

num_failed = 0

if not os.path.exists("images"):
    os.makedirs("images")
for index, row in df.iterrows():
    if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
        image_url = row["image_url"]
        try:
            image_path = os.path.join("images", row["id"] + ".jpg")
            if not os.path.exists(image_path):
                urllib.request.urlretrieve(image_url, "images/" + row["id"] + ".jpg")
        except:
            num_failed += 1
        pbar.update(1)
print("done")
print("num_failed: {}".format(num_failed))
