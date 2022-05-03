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

from argument_graphs.argsum import ArgSum

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

DATA_PATH = "./data/Fakeddit"
PL_ASSETS_PATH = "./lightning_logs"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
IMAGE_EXTENSION = ".jpg"
DIALOGUE_DATA_FILE = os.path.join(DATA_PATH, "all_comments.tsv")

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
    """
    torch.utils.data.Dataset which supports data for the following modalities:
    text, image, text-image, text-image-dialogue with BART summarization (or
    any other model which can be used in HuggingFace transformers.pipeline
    for summarization)

    Data preprocessing functions are called by __init__(), producing saved
    dataframe .pkl files which can then be passed in as the
    `from_preprocessed_dataframe` arg on subsequent uses of the same dataset

    __getitem__() returns a dict containing the appropriate fields present
    depending on the modality: "id", "text", "image", "dialogue", "label"
    """

    def __init__(
        self,
        from_preprocessed_dataframe=None, # Path to preprocessed dataframe to load from (or the actual pd.DataFrame)
        from_dialogue_dataframe=None, # Path to dataframe containing only filtered dialogue data
        data_path=None, # Path to data (i.e. not using preprocessed dataframe)
        dir_to_save_dataframe="data/Fakeddit", # Save the preprocessed dataframe here
        prefix_for_all_generated_pkl_files="", # Adds a prefix to all .pkl files that are saved
        dataset_type="train",
        modality=None,
        text_embedder=None,
        image_transform=None,
        image_encoder=None,
        dialogue_method="ranksum", # "graphlin" | "argsum"
        summarization_model=None, # Transformers pipeline model for ranksum and argsum
        num_classes=2,
    ):

        self.dataset_type = dataset_type
        self.modality = modality
        self.num_classes = num_classes
        self.dir_to_save_dataframe = dir_to_save_dataframe
        self.prefix_for_all_generated_pkl_files = prefix_for_all_generated_pkl_files
        self.dialogue_method = dialogue_method

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
        self.image_encoder = image_encoder
        self.summarization_model = summarization_model

        # NOTE: In-house summarization model will be handled in a separate
        # torch.utils.data.Dataset class
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

            # NOTE: The final dataframe is saved in _preprocess_dialogue()

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
        """ Returns the size of the dataset; called as len(dataset) """
        return len(self.data_frame.index)

    def __getitem__(self, idx):
        """
        Returns a dict containing
            A text embedding Tensor (i.e. the post's main text, embedded)
            An image RGB Tensor (which is not encoded, allowing for the model
                to decide how it wants to generate image embeddings, e.g. via
                ResNet, DINO, etc.)
            A dialogue summary embedding Tensor (i.e. data preprocessing ran
                the dialogue summarization model to generate summaries for each
                post's comments; the dataset embeds that summary)

        For data parallel training, the text embedding step MUST happen in the
        torch.utils.data.Dataset __getitem__() method; otherwise, any data that
        is not embedded will not be distributed across the multiple GPUs
            Note that the image, although not embedded, is not an issue since
            it is also returned as a Tensor (rather than a string of text, etc.)

        The item returns looks something like the following (with the
        appropriate fields present depending on the modality)
            item = {
                "id": item_id,
                "text": text,
                "image": image,
                "dialogue": dialogue,
                "label": label
            }
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text, image, dialogue = None, None, None
        item_id = self.data_frame.loc[idx, 'id']

        label = torch.Tensor(
            [self.data_frame.loc[idx, self.label]]
        ).long().squeeze()

        item = {
            "id": item_id,
            "label": label
        }

        if Modality(self.modality) in [Modality.TEXT, Modality.TEXT_IMAGE, \
                                       Modality.TEXT_IMAGE_DIALOGUE]:
            text = self.data_frame.loc[idx, 'clean_title']
            text = self.text_embedder.encode(text, convert_to_tensor=True)
            item["text"] = text

        if Modality(self.modality) in [Modality.IMAGE, Modality.TEXT_IMAGE, \
                                       Modality.TEXT_IMAGE_DIALOGUE]:
            image_path = os.path.join(IMAGES_DIR, item_id + IMAGE_EXTENSION)
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image)
            item["image"] = image

        if Modality(self.modality) == Modality.TEXT_IMAGE_DIALOGUE:
            # Get the correct dialogue representation (i.e. RankSum, GraphLin,
            # or ArgSum)
            dialogue = None
            if self.dialogue_method == "ranksum":
                dialogue = self.data_frame.loc[idx, 'comment_summary']
            elif self.dialogue_method == "graphlin":
                dialogue = self.data_frame.loc[idx, 'dialogue_linearized_graph']
            elif self.dialogue_method == "argsum":
                dialogue = self.data_frame.loc[idx, 'dialogue_argsum_summary']

            # Generate an embedding for the dialogue representation
            dialogue = self.text_embedder.encode(dialogue, convert_to_tensor=True)
            item["dialogue"] = dialogue

        return item

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
        # Dataframes for GraphLin and ArgSum will have graphlin or argsum prefixed
        # (Note that ranksum is default and has no prefix)
        if self.dialogue_method == "graphlin" or self.dialogue_method == "argsum":
            filename = str(self.dialogue_method) + "__" + filename
        if self.prefix_for_all_generated_pkl_files != "":
            # Add prefix if specified, e.g. "sampled_train__{...}.pkl"
            filename = self.prefix_for_all_generated_pkl_files + "_" + filename
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
            """
            Runs RankSum to generate dialogue summaries and saves final
            dataframe

            Params
                df: Dataframe made from dialogue data file
                save_path: Path to save final dataframe
            """

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
        # Dataframes for GraphLin and ArgSum will have graphlin or argsum prefixed
        # (Note that ranksum is default and has no prefix)
        if self.dialogue_method == "graphlin" or self.dialogue_method == "argsum":
            final_df_filename = str(self.dialogue_method) + "__" + final_df_filename
        # Add the final prefix, if any
        if self.prefix_for_all_generated_pkl_files != "":
            final_df_filename = self.prefix_for_all_generated_pkl_files + "_" + final_df_filename
        final_df_save_path = os.path.join(self.dir_to_save_dataframe, final_df_filename)

        # Setup the dialogue dataframe (referred to as `df`)
        df = None
        if from_saved_dialogue_df_path != "":
            # Special Case (see above comment in __init__)
            df = pd.read_pickle(from_saved_dialogue_df_path)
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
            # Note that the dialogue dataframe is the same for RankSum, GraphLin, and ArgSum
            if self.prefix_for_all_generated_pkl_files != "":
                dialogue_df_filename = self.prefix_for_all_generated_pkl_files + "_" + dialogue_df_filename
            dialogue_df_save_path = os.path.join(self.dir_to_save_dataframe, dialogue_df_filename)
            df.to_pickle(dialogue_df_save_path)
            logging.info("Filtered dialogue dataframe saved to {}".format(dialogue_df_save_path))

        # Save the final main dataframe (with text, image, and dialogue
        # data with dialogue summaries); note that its filename is different
        # Filename will look something like "train__text_image_dialogue__dataframe.pkl"
        if self.dialogue_method == "ranksum":
            generate_summaries_and_save_df(df, save_path=final_df_save_path)
        elif self.dialogue_method == "graphlin":
            self.run_graphlin_and_save_df(df, save_path=final_df_save_path)
        elif self.dialogue_method == "argsum":
            self.run_argsum_and_save_df(df, save_path=final_df_save_path)

    def run_graphlin_and_save_df(self, df, save_path="data/Fakeddit"):
        """
        Runs ArgSum until the GraphLin step (i.e. constructs an argument graph
        and linearizes it, but does not summarize it)

        Saves to the "dialogue_linearized_graph" column of the dataframe

        Params
            df: Dataframe made from dialogue data file
            save_path: Path to save final dataframe
        """
        logging.info("Generating linearized argument graphs via ArgSum's GraphLin for current dataset...")

        # Add new column in main dataframe to hold dialogue linearized argument graphs
        self.data_frame['dialogue_linearized_graph'] = ""

        # Setup ArgSum instance
        # TODO: MAKE THE MODEL VERSIONS AND TOKENIZER MODEL NAMES ARGS
        argsum = ArgSum(
            auc_trained_model_version=209,
            rtc_trained_model_version=264,
            auc_tokenizer_model_name="bert-base-uncased",
            rtc_tokenizer_model_name="bert-base-uncased"
        )

        failed_ids = []
        for iteration, text_id in enumerate(self.text_ids):
            if (iteration % 250 == 0):
                print("Generating linearized argument graphs for item {}...".format(iteration))
                # Save progress so far
                self.data_frame.to_pickle(save_path)

            try:
                # Group comments by post id
                all_comments = df[df['submission_id'] == text_id]
                all_comments.sort_values(by=['ups'], ascending=False)
                all_comments = list(all_comments['body'])

                # Generate linearized argument graph (i.e. run ArgSum, but stop
                # at GraphLin step)
                linearized_graph: str = argsum.construct_and_linearize(all_comments)

                # Add to self.data_frame 'dialogue_linearized_graph' column
                self.data_frame.loc[self.data_frame['id'] == text_id, 'dialogue_linearized_graph'] = linearized_graph
            except:
                failed_ids.append(text_id)

        # Save final main dataframe
        self.data_frame.to_pickle(save_path)
        print("Preprocessed final dataframe (with dialogue linearized argument graphs) saved to {}".format(save_path))
        logging.info("Preprocessed final dataframe (with dialogue linearized argument graphs) saved to {}".format(save_path))

        logging.debug(self.data_frame)
        logging.debug(self.data_frame['dialogue_linearized_graph'])
        logging.debug("num_failed: " + str(len(failed_ids)))
        logging.debug(failed_ids)
        print("Preprocessed final dataframe (with dialogue linearized argument graphs) saved to {}".format(save_path))
        logging.info("Preprocessed final dataframe (with dialogue linearized argument graphs) saved to {}".format(save_path))

        return

    def run_argsum_and_save_df(self, df, save_path="data/Fakeddit"):
        """
        Runs the full ArgSum algorithm (i.e. graph construction, graph
        linearization via GraphLin, text summarization via Transformers)

        Saves to the "dialogue_argsum_summary" column of the dataframe

        Params
            df: Dataframe made from dialogue data file
            save_path: Path to save final dataframe
        """
        logging.info("Generating summaries via ArgSum for current dataset...")

        # Add new column in main dataframe to hold dialogue linearized argument graphs
        self.data_frame['dialogue_argsum_summary'] = ""

        # Setup ArgSum instance
        # TODO: MAKE THE MODEL VERSIONS AND TOKENIZER MODEL NAMES ARGS
        argsum = ArgSum(
            auc_trained_model_version=209,
            rtc_trained_model_version=264,
            auc_tokenizer_model_name="bert-base-uncased",
            rtc_tokenizer_model_name="bert-base-uncased"
        )

        failed_ids = []
        for iteration, text_id in enumerate(self.text_ids):
            if (iteration % 250 == 0):
                print("Generating summaries via ArgSum for item {}...".format(iteration))
                # Save progress so far
                self.data_frame.to_pickle(save_path)

            try:
                # Group comments by post id
                all_comments = df[df['submission_id'] == text_id]
                all_comments.sort_values(by=['ups'], ascending=False)
                all_comments = list(all_comments['body'])

                # Generate summary via full ArgSum algorithm (i.e. graph
                # construction, graph linearization via GraphLin, text
                # summarization via Transformers)
                summary: str = argsum.summarize(all_comments)

                # Add to self.data_frame 'dialogue_linearized_graph' column
                self.data_frame.loc[self.data_frame['id'] == text_id, 'dialogue_argsum_summary'] = summary
            except:
                failed_ids.append(text_id)

        # Save final main dataframe
        self.data_frame.to_pickle(save_path)
        print("Preprocessed final dataframe (with dialogue summaries via ArgSum) saved to {}".format(save_path))
        logging.info("Preprocessed final dataframe (with dialogue summaries via ArgSum) saved to {}".format(save_path))

        logging.debug(self.data_frame)
        logging.debug(self.data_frame['dialogue_argsum_summary'])
        logging.debug("num_failed: " + str(len(failed_ids)))
        logging.debug(failed_ids)
        print("Preprocessed final dataframe (with dialogue summaries via ArgSum) saved to {}".format(save_path))
        logging.info("Preprocessed final dataframe (with dialogue summaries via ArgSum) saved to {}".format(save_path))

        return
