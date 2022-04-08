import sys
import os
from pathlib import Path
import logging
import argparse
import enum

import pandas as pd
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

DATA_PATH = "./data/Fakeddit"
PL_ASSETS_PATH = "./lightning_logs"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
DIALOGUE_DATA_FILE = os.path.join(DATA_PATH, "all_comments.tsv")
IMAGE_EXTENSION = ".jpg"

class Modality(enum.Enum):
    """
    Note on Comparisons: Either use `string_value == enum.value`
    or `Modality(string_value) == enum`
    """
    TEXT = "text"
    IMAGE = "image"
    TEXT_IMAGE = "text-image"
    TEXT_IMAGE_DIALOGUE = "text-image-dialogue"

class MultimodalDataset(Dataset):

    def __init__(
        self,
        from_preprocessed_dataframe=None, # Preprocessed dataframe to load from
        from_dialogue_dataframe=None, # Dataframe containing only filtered dialogue data
        data_path=None, # Path to data (i.e. not using preprocessed dataframe)
        dir_to_save_dataframe="data/Fakeddit", # Save the preprocessed dataframe here
        dataset_type="train",
        modality=None,
        text_embedder=None,
        image_encoder=None,
        image_transform=None,
        summarization_model=None,
        num_classes=2,
        images_dir=IMAGES_DIR
    ):

        self.dataset_type = dataset_type
        self.modality = modality
        self.num_classes = num_classes
        self.dir_to_save_dataframe = dir_to_save_dataframe

        self.label = "2_way_label"
        if num_classes == 3:
            self.label = "3_way_label"
        elif num_classes == 6:
            self.label = "6_way_label"

        self.saved_dataframe_filename_prefix = ""
        if Modality(modality) == Modality.TEXT:
            self.saved_dataframe_filename_prefix = "text"
        elif Modality(modality) == Modality.IMAGE:
            self.saved_dataframe_filename_prefix = "image"
        elif Modality(modality) == Modality.TEXT_IMAGE:
            self.saved_dataframe_filename_prefix = "text_image"
        elif Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
            self.saved_dataframe_filename_prefix = "text_image_dialogue"

        self.text_embedder = text_embedder
        self.image_transform = image_transform
        self.summarization_model = summarization_model

        # TODO: Handle in-house summarization model
        self.summarizer = None
        if Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE and summarization_model:
            # Model options: "facebook/bart-large-cnn", "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
            # https://huggingface.co/docs/transformers/master/en/main_classes/pipelines#transformers.SummarizationPipeline
            self.summarizer = transformers.pipeline("summarization", model=summarization_model)
        elif Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
            self.summarizer = transformers.pipeline("summarization")


        df = None
        if not from_preprocessed_dataframe:
            print("Running data preprocessing from scratch...")
            # This is the first time this data is being setup, so we run full preprocessing
            df = pd.read_csv(data_path, sep='\t', header=0)
            df = self._preprocess_df(df)
            logging.debug(df.columns)
            logging.debug(df['clean_title'])

            # Save main dataframe and ids (needed for dialogue preprocessing)
            self.data_frame = df
            self.text_ids = set(self.data_frame['id'])

            # Run dialogue preprocessing, if needed
            if Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
                # Special Case: Since the dialogue data is huge, we can first preprocess the
                # dialogue (comments) dataframe to only keep the comments that pertain to
                # posts in our dataset's data, and save it into a serialized .pkl file;
                # If we do that, then we'll run our dialogue preprocessing using that
                # dataframe (which we load from the .pkl file)
                if from_dialogue_dataframe:
                    self._preprocess_dialogue(from_saved_dialogue_df_path=from_dialogue_dataframe)
                else:
                    self._preprocess_dialogue()

            # TODO: Save final dataframe

        else: # from_preprocessed_dataframe:
            print("Running data preprocessing from preprocessed dataframe...")
            df = None
            if isinstance(from_preprocessed_dataframe, pd.DataFrame):
                df = from_preprocessed_dataframe
            elif isinstance(from_preprocessed_dataframe, str):
                df = pd.read_pickle(from_preprocessed_dataframe)
            else:
                raise Exception("MultimodalDataset given invalid from_preprocessed_dataframe arg; \
                                 Must be path (str) to dataframe or pd.DataFrame")

        self.data_frame = df
        self.text_ids = set(self.data_frame['id'])
        logging.debug(self.data_frame)

        return

    def __len__(self):
        return len(self.data_frame.index)

    def __getitem__(self, idx):
        pass

    def _preprocess_df(self, df):
        """
        Preprocesses dataframe by only keeping multi-modal examples, i.e. those
        whose images exist

        Note that some posts inherently do not have images, and those are the
        examples we are removing; however, if you have not downloaded the
        images for a post, it will also be removed; thus, first run
        `data/Fakeddit/image_downloader.py` for both the train and test data
        (specifying the data file path via the `--data_filename` arg)

        Saves the dataframe to `{dataset_type}__{modality}__dataframe.pkl`,
        e.g. `train__text_image__dataframe.pkl`
        """

        def image_exists(row):
            """ Ensures that image exists and can be opened """
            image_path = os.path.join(IMAGES_DIR, row['id'] + IMAGE_EXTENSION)
            if not os.path.exists(image_path):
                return False

            try:
                image = Image.open(image_path)
                image.verify()
                image.close()
                return True
            except Exception:
                return False

        df['image_exists'] = df.apply(lambda row: image_exists(row), axis=1)
        df = df[df['image_exists'] == True].drop('image_exists', axis=1)
        if df.empty:
            print("WARNING: None of the images for this dataset have been downloaded, so the dataframe is empty!")
            print("Download the images first using data/Fakeddit/image_downloader.py then run data preprocessing again")
        df = df.drop(['created_utc', 'domain', 'hasImage', 'image_url'], axis=1)
        df.reset_index(drop=True, inplace=True)

        # Save this dataframe into a pickle
        # Filename will look something like "train__text_image__dataframe.pkl"
        filename = "__".join([self.dataset_type, self.saved_dataframe_filename_prefix, "dataframe.pkl"])
        save_path = os.path.join(self.dir_to_save_dataframe, filename)
        df.to_pickle(save_path)
        print("Preprocessed dataframe saved to {}".format(save_path))
        logging.info("Preprocessed dataframe saved to {}".format(save_path))

        return df

    def _preprocess_dialogue(self, from_saved_dialogue_df_path=""):
        """
        Preprocesses dialogue data by generating dialogue summaries and storing
        them as a column in the main dataframe

        A comment's 'submission_id' is linked (i.e. equal) to a post's 'id';
        'body' contains the comment text and 'ups' contains upvotes

        Saves the dialogue dataframe (i.e. containing only dialogue data,
        filtered to keep the comments with posts in the current main dataset)
        to {dataset_type}__dialogue_dataframe.pkl`,
        e.g. `train__dialogue_dataframe.pkl`

        Saves the final main dataframe (i.e. containing text, image, and
        dialogue data with dialogue summaries) to
        `{dataset_type}__{modality}__dataframe.pkl`,
        e.g. `train__text_image_data__dataframe.pkl`
        """

        def generate_summaries_and_save_df(df, save_path="data/Fakeddit"):
            """ df: Dataframe made from dialogue data file """

            logging.info("Generating summaries for current dataset...")

            # Add new column in main dataframe to hold dialogue summaries
            self.data_frame['comment_summary'] = ""

            failed_ids = []
            for iteration, text_id in enumerate(self.text_ids):
                if (iteration % 250 == 0):
                    print("Generating summaries for item {}...".format(iteration))
                    # Save progress so far
                    self.data_frame.to_pickle(save_path)

                try:
                    # Group comments by post id
                    all_comments = df[df['submission_id'] == text_id]
                    all_comments.sort_values(by=['ups'], ascending=False)
                    all_comments = list(all_comments['body'])

                    # Generate summary of comments via Transformers pipeline
                    corpus = "\n".join(all_comments)
                    summary = "none" # Default if no comments for this post
                    if len(all_comments) > 0:
                        # We define the summary's max_length as max(min(75, num_words // 2), 5)
                        # Note that num_words is calculated very roughly, splitting on whitespace
                        num_words = sum([len(comment.split()) for comment in all_comments])
                        max_length = min(75, num_words // 2) # For short comment threads, it'll be <75
                        max_length = max(max_length, 5) # Avoid 1-length maxes, which leads to unexpected behavior
                        min_length = min(5, max_length - 1)
                        summary = self.summarizer(corpus, min_length=min_length, max_length=max_length, truncation=True)

                        # Pipeline returns a list containing a dict
                        # https://huggingface.co/docs/transformers/master/en/main_classes/pipelines
                        summary = summary[0]['summary_text']

                    # Add summary to self.data_frame 'comment_summary' column
                    self.data_frame.loc[self.data_frame['id'] == text_id, 'comment_summary'] = summary
                except:
                    failed_ids.append(text_id)

            # Save final main dataframe
            self.data_frame.to_pickle(save_path)
            print("Preprocessed final dataframe (with dialogue summaries) saved to {}".format(save_path))
            logging.info("Preprocessed final dataframe (with dialogue summaries) saved to {}".format(save_path))

            logging.debug(self.data_frame)
            logging.debug(self.data_frame['comment_summary'])
            logging.debug("num_failed: " + str(len(failed_ids)))
            logging.debug(failed_ids)
            print("Preprocessed final dataframe (with dialogue summaries) saved to {}".format(save_path))
            logging.info("Preprocessed final dataframe (with dialogue summaries) saved to {}".format(save_path))

            return

        # After preprocessing, we will save this dialogue dataframe into a pickle
        # Filename will look something like "train__text_image_dialogue__dataframe.pkl"
        final_df_filename = "__".join([self.dataset_type, self.saved_dataframe_filename_prefix, "dataframe.pkl"])
        final_df_save_path = os.path.join(self.dir_to_save_dataframe, final_df_filename)

        if from_saved_dialogue_df_path != "":
            # Special Case (see above comment in __init__)
            df = pd.read_pickle(from_saved_dialogue_df_path)
            generate_summaries_and_save_df(df, save_path=final_df_save_path)
        else:
            df = pd.read_csv(DIALOGUE_DATA_FILE, sep='\t')
            logging.debug(df)

            def text_exists(row):
                """ Ensures that a comment's corresponding text exists """
                if row['submission_id'] in self.text_ids:
                    return True
                else:
                    return False

            def comment_deleted(row):
                """ If a comment was deleted, its body just contains [deleted] """
                return row['body'] == "[deleted]"

            # Filter dialogue dataframe to only keep comments with posts in the
            # current main dataset; note that the dialogue data is huge, so
            # this will take a while (but we will save it)
            logging.info("Filtering dialogue dataframe to keep only comments with posts in current dataset...")
            df['text_exists'] = df.apply(lambda row: text_exists(row), axis=1)
            df = df[df['text_exists'] == True].drop('text_exists', axis=1)
            df['comment_deleted'] = df.apply(lambda row: comment_deleted(row), axis=1)
            df = df[df['comment_deleted'] == False].drop('comment_deleted', axis=1)
            df.reset_index(drop=True, inplace=True)
            logging.debug(df)

            # Save dialogue dataframe so far before summary generation
            # Again: This is a dataframe of ONLY dialogue data, filtered to
            # keep the comments with posts in the current main dataset
            # Filename will look something like "train__dialogue_dataframe.pkl"
            dialogue_df_filename = "__".join([self.dataset_type, "dialogue_dataframe.pkl"])
            dialogue_df_save_path = os.path.join(self.dir_to_save_dataframe, dialogue_df_filename)
            df.to_pickle(dialogue_df_save_path)
            logging.info("Filtered dialogue dataframe saved to {}".format(dialogue_df_save_path))

            # Save the final main dataframe (with text, image, and dialogue
            # data with dialogue summaries); note that its filename is different
            # Filename will look something like "train__text_image_dialogue__dataframe.pkl"
            generate_summaries_and_save_df(df, save_path=final_df_save_path)
