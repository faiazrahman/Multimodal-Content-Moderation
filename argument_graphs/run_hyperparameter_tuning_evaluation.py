"""
Run from root directory
```
python -m argument_graphs.run_hyperparameter_tuning_evaluation
```
"""

import itertools

from run_experiments import run

GPU_ID = 3

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

def evaluation_auc_preshuffled():
    # Hyperparameter tuning for preshuffled AUC created models 202 to 228,
    # inclusive, and 230
    # Models 202-215 are BERT; models 216-228 and 230 are RoBERTa
    # (Note that 229 is an RTC model)
    bert_model_version_numbers = [i for i in range(202, 216)]
    roberta_model_version_numbers = [i for i in range(216, 229)]
    roberta_model_version_numbers.append(230)

    # BERT
    for version_number in bert_model_version_numbers:
        print(f"Evaluating model version_{version_number}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_evaluation --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__bert-base-uncased.yaml --batch_size 16 --trained_model_version {version_number} --gpus {GPU_ID}")

    # RoBERTa
    for version_number in roberta_model_version_numbers:
        print(f"Evaluating model version_{version_number}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_evaluation --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__roberta-base.yaml --batch_size 16 --trained_model_version {version_number} --gpus {GPU_ID}")

def evaluation_rtc():
    bert_model_version_numbers = [
        # 173, # bert, adam, 1e-3
        # 259, # bert, adam, 5e-4
        # 229, # bert, adam, 3e-4
        243, # bert, adam, 5e-5
        263, # bert, adam, 3e-5
    ]
    for version_number in bert_model_version_numbers:
        print(f"Evaluating model version_{version_number}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_evaluation --relationship_type_classification --config configs/relationship_type_classification/rtc__bert-base-uncased.yaml --batch_size 32 --trained_model_version {version_number} --gpus {GPU_ID}")

if __name__ == "__main__":
    # evaluation_auc()
    # evaluation_auc_preshuffled()
    evaluation_rtc()
