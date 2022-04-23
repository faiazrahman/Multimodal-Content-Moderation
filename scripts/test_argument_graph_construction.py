"""
Run (from root)
```
python -m scripts.test_argument_graph_construction
```
"""

from argument_graphs.modules import ArgumentGraphConstructor

BERT = "bert-base-uncased"
ROBERTA = "roberta-base"

if __name__ == "__main__":
    graph_constructor = ArgumentGraphConstructor(
        auc_trained_model_version=209,
        rtc_trained_model_version=264,
        auc_tokenizer_model_name=BERT,
        rtc_tokenizer_model_name=BERT
    )

    dialogue_utterances = [
        "I think we should make healthcare free. Free healthcare is accomplished through policymakers",
        "I hate this post. The images in this post are scary.",
        "Free healthcare should be a human right. Free healthcare in Canada is well-received. Citizens like free healthcare!",
        "I think free healthcare is great. My healthcare plan is free.",
        "The president's healthcare policies are horrible. Did you see the awful healthcare policy?",
        "Citizens are upset."
    ]

    graph_constructor.construct_graph(dialogue_utterances=dialogue_utterances)
