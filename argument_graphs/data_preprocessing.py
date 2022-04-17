import sys
import os
import logging
import argparse

AMPERSAND_DATA_PATH = "../data/AMPERSAND"
SGAM_DATA_PATH = "../data/StabGurevychArgMining"

def clean_ampersand_data():
    """
    Cleans the `claimtrain.tsv` and `claimdev.tsv` data in `data/AMPERSAND` to
    keep only the relevant examples for argumentative unit classification, and
    stores it in a single .tsv file for easy loading into a pandas DataFrame
    later (note that the original .tsv files are left unchanged)

    Note: We combine the original train and dev since we will split the data
    later after it is combined with the Stab & Gurevych data

    Creates `ampersand_auc_data.tsv` in `data/AMPERSAND/`
    """
    pass

def clean_and_combine_sgam_data():
    """
    Combines the individual `*.ann` files in `data/StabGurevychArgMining/annotated_data`,
    cleans the data to keep only the relevant examples for argumentative unit
    classification, and stores it in a single .tsv for easy loading into a
    pandas DataFrame later

    Creates `sgam_auc_data.tsv` in `data/StabGurevychArgMining/`
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
