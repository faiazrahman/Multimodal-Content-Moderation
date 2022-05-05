"""
Run (from root)
```
python -m dialogue_summarization.run_baseline_bart_rouge_evaluation
```
"""
import os
import logging
from pprint import pprint
from collections import defaultdict

import pandas as pd

import transformers
from rouge_metric import PyRouge

DATA_PATH = "./data"
SAMSUM_DATA_PATH = os.path.join(DATA_PATH, "SAMSum")
SAMSUM_TRAIN_DATA_PATH = os.path.join(SAMSUM_DATA_PATH, "train.json")
SAMSUM_TEST_DATA_PATH = os.path.join(SAMSUM_DATA_PATH, "test.json")
SAMSUM_VAL_DATA_PATH = os.path.join(SAMSUM_DATA_PATH, "val.json")

SAMSUM_TRAIN_DATAFRAME_PATH = os.path.join(SAMSUM_DATA_PATH, "samsum_train_dataframe.pkl")
SAMSUM_TEST_DATAFRAME_PATH = os.path.join(SAMSUM_DATA_PATH, "samsum_test_dataframe.pkl")
SAMSUM_VAL_DATAFRAME_PATH = os.path.join(SAMSUM_DATA_PATH, "samsum_val_dataframe.pkl")

logging.basicConfig(level=logging.DEBUG)

def evaluate_baseline_bart(force_regenerate_summaries=False):
    """ Evaluates baseline BART's summarization performance on SAMSum data """
    rouge = PyRouge(
        # ROUGE-N: Overlap of n-grams
        rouge_n=(1, 2, 4),
        # ROUGE-L: Longest common subsequence based statistics
        rouge_l=True,
        # ROUGE-W: Weighted LCS-based statistics that favors consecutive LCSs
        rouge_w=True,
        rouge_w_weight=1.2,
        # ROUGE-S: Skip-bigram based co-occurrence statistics
        # Note: A skip-bigram is any pair of words in their sentence order
        rouge_s=True,
        # ROUGE-SU: Skip-bigram plus unigram-based co-occurrence statistics
        rouge_su=True,
        skip_gap=4
    )
    logging.debug(rouge)

    df = pd.read_pickle(SAMSUM_TEST_DATAFRAME_PATH)
    logging.debug(df)

    # Transformer pipeline to generate BART summaries
    # We initialize it here so it is in-scope for `generate_summary()`, but we
    # do not load the model yet (since it may not be needed, if summaries have
    # already been generated; this saves the time spent loading the model)
    summarizer = None

    # Map id: summary (used to precompute summaries and then add them quickly
    # to the actual dataframe via `df.apply`)
    summaries = defaultdict(str)

    def generate_summary(idx):
        """
        Generate summary of dialogue using BART Transformer pipeline and save
        it to the summaries dict

        This is used to precompute the summaries so they can then be fetched
        in `df.apply`
        """
        row = df.iloc[idx]
        dialogue_utterances = row['dialogue']
        corpus = "\n".join(dialogue_utterances)

        # We define the summary's max_length as max(min(75, num_words // 2), 5)
        # Note that num_words is calculated very roughly, splitting on whitespace
        num_words = sum([len(utterance.split()) for utterance in dialogue_utterances])
        max_length = min(75, num_words // 2) # For short comment threads, it'll be <75
        max_length = max(max_length, 5) # Avoid 1-length maxes, which leads to unexpected behavior
        min_length = min(5, max_length - 1)
        summary = summarizer(corpus, min_length=min_length, max_length=max_length, truncation=True)

        # Pipeline returns a list containing a dict
        # https://huggingface.co/docs/transformers/master/en/main_classes/pipelines
        summary = summary[0]['summary_text']
        summaries[row['id']] = summary

    def fetch_summary(row):
        """
        Fetch the summary from the summaries dict after it's been generated

        This is used as the lambda function in `df.apply`
        """
        return summaries.get(row['id'], "")

    # If baseline BART summaries have not been generated (i.e. not in the saved
    # dataframe .pkl) or we're forcing re-generating summaries, run pipeline
    if 'baseline_bart_summary' not in df.columns or force_regenerate_summaries:
        logging.info("Generating baseline BART summaries for SAMSum (for ROUGE evaluation)...")
        # Transformer pipeline to generate BART summaries
        summarizer = transformers.pipeline("summarization", model="facebook/bart-large-cnn")

        # Generate the summaries and save them to the same dataframe .pkl
        for idx in range(len(df.index)):
            if idx % 50 == 0: print(f"Generating summaries for item {idx}...")
            generate_summary(idx)
        df['baseline_bart_summary'] = df.apply(lambda row: fetch_summary(row), axis=1)
        df.to_pickle(SAMSUM_TEST_DATAFRAME_PATH)

    print(df)

    hypotheses = df['baseline_bart_summary'].tolist()
    references = df['summary'].tolist()
    if len(hypotheses) != len(references):
        raise Exception(f"Error: The number of `baseline_bart_summary` strings ({len(hypotheses)}) does not match the number of reference `summary` strings ({len(references)}); Try re-running evaluate_baseline_bart() with force_regenerate_summaries=True")
    print(f"\nNumber of summaries to evaluate: {len(hypotheses)}")

    print("Evaluating baseline BART summaries on SAMSum with ROUGE...")
    scores = rouge.evaluate(hypotheses, references)
    pprint(scores)

if __name__ == "__main__":
    evaluate_baseline_bart()
