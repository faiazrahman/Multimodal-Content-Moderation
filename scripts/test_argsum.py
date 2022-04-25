"""
Run (from root)
```
python -m scripts.test_argsum
```
"""

from argument_graphs.argsum import ArgSum

BERT = "bert-base-uncased"
ROBERTA = "roberta-base"

if __name__ == "__main__":

    dialogue_utterances = [
        "I think we should make healthcare free. Free healthcare is accomplished through policymakers",
        "I hate this post. The images in this post are scary.",
        "Free healthcare should be a human right. Free healthcare in Canada is well-received. Citizens like free healthcare!",
        "I think free healthcare is great. My healthcare plan is free.",
        "The president's healthcare policies are horrible. Did you see the awful healthcare policy?",
        "Citizens are upset."
    ]

    argsum = ArgSum(
        auc_trained_model_version=209,
        rtc_trained_model_version=264,
        auc_tokenizer_model_name=BERT,
        rtc_tokenizer_model_name=BERT
    )
    print(argsum)

    summary = argsum.summarize(dialogue_utterances)
    print(summary)
