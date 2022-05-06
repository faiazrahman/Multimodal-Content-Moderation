"""
Run from root directory
```
python -m argument_graphs.run_hyperparameter_tuning
```

Note: If you notice that a job is not learning at all, you don't have to wait
for it to finish; you can stop it immediately and move on to the next job.
- Python's `subprocess` creates a new process, so you can kill just that
  process (and not this entire script) via pid using the usual Unix kill
  command; i.e., run `nvidia-smi` to see the process ID, then run `kill <pid>`.
"""

import itertools

from run_experiments import run

GPU_ID = 5

def hyperparameter_tuning_auc():
    # Hyperparameter tuning for AUC with {BERT, RoBERTA}
    optimizers = ["adam", "sgd"]
    learning_rates = [1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5]
    combinations = list(itertools.product(optimizers, learning_rates))

    # BERT
    for opt, lr in combinations:
        print(f"optimizer: {opt}, learning_rate: {lr}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_training --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__bert-base-uncased.yaml --batch_size 16 --num_epochs 5 --optimizer {opt} --learning_rate {lr} --gpus {GPU_ID}")

    # RoBERTa
    for opt, lr in combinations:
        print(f"optimizer: {opt}, learning_rate: {lr}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_training --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__roberta-base.yaml --batch_size 16 --num_epochs 5 --optimizer {opt} --learning_rate {lr} --gpus {GPU_ID}")

def hyperparameter_tuning_rtc():
    # Hyperparameter tuning for RTC with {BERT, RoBERTA}
    optimizers = ["adam", "sgd"]
    learning_rates = [1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5]
    combinations = list(itertools.product(optimizers, learning_rates))

    # BERT
    for opt, lr in combinations:
        print(f"optimizer: {opt}, learning_rate: {lr}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_training --relationship_type_classification --config configs/relationship_type_classification/rtc__bert-base-uncased.yaml --batch_size 32 --num_epochs 2 --optimizer {opt} --learning_rate {lr} --gpus {GPU_ID}")

    # RoBERTa
    for opt, lr in combinations:
        print(f"optimizer: {opt}, learning_rate: {lr}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_training --relationship_type_classification --config configs/relationship_type_classification/rtc__roberta-base.yaml --batch_size 32 --num_epochs 2 --optimizer {opt} --learning_rate {lr} --gpus {GPU_ID}")

def run_best_rtc():
    opt = "adam"
    lr = 3e-5
    run(f"python -m argument_graphs.run_argument_graph_submodel_training --relationship_type_classification --config configs/relationship_type_classification/rtc__bert-base-uncased.yaml --batch_size 32 --num_epochs 2 --optimizer {opt} --learning_rate {lr} --gpus {GPU_ID}")

if __name__ == "__main__":

    hyperparameter_tuning_auc()
    # hyperparameter_tuning_rtc()

    # run_best_rtc()
