"""
Run from root directory
```
python -m argument_graphs.run_hyperparameter_tuning_evaluation
```
"""

import itertools

from run_experiments import run

GPU_ID = 7

def evaluation_auc():
    # Hyperparameter tuning for AUC created models 145 to 172, inclusive
    # The first half (145-158) are BERT; the second half are RoBERTa (159-172)
    bert_model_version_numbers = [i for i in range(145, 159)]
    roberta_model_version_numbers = [i for i in range(159, 173)]

    # BERT
    for version_number in bert_model_version_numbers:
        print(f"Evaluating model version_{version_number}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_evaluation --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__bert-base-uncased.yaml --batch_size 16 --trained_model_version {version_number} --gpus {GPU_ID}")

    # RoBERTa
    for version_number in roberta_model_version_numbers:
        print(f"Evaluating model version_{version_number}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_evaluation --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__roberta-base.yaml --batch_size 16 --trained_model_version {version_number} --gpus {GPU_ID}")

if __name__ == "__main__":
    evaluation_auc()
