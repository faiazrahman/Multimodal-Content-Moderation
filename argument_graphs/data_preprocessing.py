import sys
import os
import logging
import argparse

DATA_PATH = "../data"
AMPERSAND_DATA_PATH = os.path.join(DATA_PATH, "AMPERSAND")
SGAM_DATA_PATH = os.path.join(DATA_PATH, "StabGurevychArgMining")
AUC_DATA_PATH = os.path.join(DATA_PATH, "ArgumentativeUnitClassification")

# Original data files for AMPERSAND
AMPERSAND_ORIGINAL_TRAIN_DATA_PATH = os.path.join(AMPERSAND_DATA_PATH, "claimtrain.tsv")
AMPERSAND_ORIGINAL_DEV_DATA_PATH = os.path.join(AMPERSAND_DATA_PATH, "claimdev.tsv")

# Directory containing all original "*.ann" files for Stab & Gurevych data
SGAM_ORIGINAL_DATA_DIR = os.path.join(SGAM_DATA_PATH, "annotated_data")

# AMPERSAND data after `clean_ampersand_data()`
AMPERSAND_AUC_DATA_FILE = "ampersand_auc_data.tsv" # ['text', 'label']
AMPERSAND_AUC_DATA_PATH = os.path.join(AMPERSAND_DATA_PATH, AMPERSAND_AUC_DATA_FILE)

# Stab & Gurevych data after `clean_and_combine_sgam_data()`
SGAM_AUC_DATA_FILE = "sgam_auc_data.tsv" # ['text', 'label']
SGAM_AUC_DATA_PATH = os.path.join(SGAM_DATA_PATH, SGAM_AUC_DATA_FILE)

# Combined AMPERSAND and Stab & Gurevych data -> ['text', 'label']
# Our 'text' is the argumentative unit (e.g. a sentence)
# Our 'label' is one of 0: Non-argumentative data, 1: Claim, 2: Premise
ALL_AUC_DATA_FILE = "all_auc_data.tsv"
ALL_AUC_DATA_PATH = os.path.join(AUC_DATA_PATH, ALL_AUC_DATA_FILE)
ALL_AUC_DATAFRAME_FILE = "auc_dataframe.pkl"
ALL_AUC_DATAFRAME_PATH = os.path.join(AUC_DATA_PATH, ALL_AUC_DATAFRAME_FILE)

def clean_ampersand_data():
    """
    Cleans the `claimtrain.tsv` and `claimdev.tsv` data in `data/AMPERSAND` to
    keep only the relevant examples for argumentative unit classification, and
    stores it in a single .tsv file for easy loading into a pandas DataFrame
    later (note that the original .tsv files are left unchanged)

    Note: We combine the original train and dev since we will split the data
    later after it is combined with the Stab & Gurevych data

    Creates `ampersand_auc_data.tsv` in `data/AMPERSAND/`, which has two
    columns ['text', 'label']

    Notes about original data format
    - .tsv with two columns: [text, label]
    - Labels are as follows
        0   Non-argumentative unit
        1   Claim
        2   Premise
    - The created `ampersand_auc_data.tsv` has these same labels
    """
    pass

def clean_and_combine_sgam_data():
    """
    Combines the individual `*.ann` files in `data/StabGurevychArgMining/annotated_data`,
    cleans the data to keep only the relevant examples for argumentative unit
    classification, and stores it in a single .tsv for easy loading into a
    pandas DataFrame later

    Creates `sgam_auc_data.tsv` in `data/StabGurevychArgMining/`, which has
    two columns ['text', 'label']

    Notes about original data format
    - The annotated data is in the `*.ann` files, with one file per original
      document; note that this is a tab-separated text file (like a .tsv), where the second column is the label and the fifth column is the argumentative unit (i.e. sentence).
    - Labels are strings, as follows
        Claim
        MajorClaim
        Premise
    - Note that the annotated data has other labels, like "Stance", "supports",
      and so on. However, we will only be using the above three labels since
      we are using this data (along with AMPERSAND) to train an argumentative
      unit classification model.
    - Additionally, we will assign both "Claim" and "MajorClaim" to the label
      1 (for claim), and "Premise" to the label 2 (for premise), matching the
      AMPERSAND data. (Note that the AMPERSAND data also has the label 0 for
      non-argumentative units.)
    """
    pass

def aggregate_all_auc_data():
    """
    Combines the AMPERSAND dataset with the Stab & Gurevych argument mining
    dataset, producing a single .tsv file with all the data

    Creates `all_auc_data.tsv` in `data/ArgumentativeUnitClassification/`
    """
    pass

def save_auc_data_to_dataframe_pkl():
    """
    Loads the combined AMPERSAND and Stab & Gurevych data into a dataframe and
    serializes it into a .pkl file with pickle

    Creates `auc_dataframe.pkl` in `data/ArgumentativeUnitClassification/`
    - This is the final file which will be used by the torch.utils.data.Dataset
      for the argumentative unit classification submodel
    """
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--argument_unit_classification", action="store_true", help="Runs data preprocessing for the AUC datasets (AMPERSAND and Stab & Gurevych)")
    parser.add_argument("--relationship_type_classification", action="store_true", help="Runs data preprocessing for the MNLI entailment dataset")
    args = parser.parse_args()

    # If no flags are specified, run all preprocessing
    if (not args.argument_unit_classification) and (not args.relationship_type_classification):
        args.argument_unit_classification = True
        args.relationship_type_classification = True

    # Run data preprocessing for the AMPERSAND and Stab & Gurevych datasets
    if args.argument_unit_classification:
        clean_ampersand_data()
        clean_and_combine_sgam_data()
        aggregate_all_auc_data()
        save_auc_data_to_dataframe_pkl()

    # Run data preprocessing for the MNLI dataset
    if args.relationship_type_classification:
        print("TODO: relationship_type_classification data preprocessing")
        pass
