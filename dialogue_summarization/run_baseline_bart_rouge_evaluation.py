"""
Run (from root)
```
python -m dialogue_summarization.run_baseline_bart_rouge_evaluation
```
"""
import os
import logging

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
    summarizer = transformers.pipeline("summarization", model="facebook/bart-large-cnn")

    def generate_summary(row):
        """ Generate summary of dialogue using BART Transformer pipeline """
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

        return summary

    # If baseline BART summaries have not been generated (i.e. not in the saved
    # dataframe .pkl) or we're forcing re-generating summaries, run pipeline
    if 'baseline_bart_summary' not in df.columns or force_regenerate_summaries:
        # Generate the summaries and save them to the same dataframe .pkl
        df['baseline_bart_summary'] = df.apply(lambda row: generate_summary(row), axis=1)
        df.to_pickle(SAMSUM_TEST_DATAFRAME_PATH)

    print(df)

    hypotheses = df['baseline_bart_summary'].tolist()
    references = df['summary'].tolist()
    print(len(hypotheses), len(references))
    # TODO: Run through ROUGE and get scores

if __name__ == "__main__":
    evaluate_baseline_bart()
