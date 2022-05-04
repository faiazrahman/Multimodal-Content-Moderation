"""
The ROUGE metric for evaluating text summarization
- We use the rouge-metric package (https://pypi.org/project/rouge-metric/)

To run the example (defined in main)
```
python -m dialogue_summarization.evaluation.rouge
```
"""

from pprint import pprint

from rouge_metric import PyRouge

if __name__ == "__main__":
    # This is an example from the rouge-metric PyPI docs
    # https://pypi.org/project/rouge-metric/

    # Load summary results
    hypotheses = [
        'how are you\ni am fine',                     # document 1: hypothesis
        'it is fine today\nwe won the football game', # document 2: hypothesis
    ]
    references = [[
        'how do you do\nfine thanks',                # document 1: reference 1
        'how old are you\ni am three',               # document 1: reference 2
    ], [
        'it is sunny today\nlet us go for a walk',   # document 2: reference 1
        'it is a terrible day\nwe lost the game',    # document 2: reference 2
    ]]

    # Evaluate document-wise ROUGE scores
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
    scores = rouge.evaluate(hypotheses, references)
    pprint(scores)
