"""
Run from root directory
```
python -m argument_graphs.run_hyperparameter_tuning
```
"""

import itertools

from run_experiments import run

GPU_ID = 5

if __name__ == "__main__":

    # Hyperparameter tuning for AUC with RoBERTA
    optimizers = ["adam", "sgd"]
    learning_rates = [1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5]

    combinations = list(itertools.product(optimizers, learning_rates))
    for opt, lr in combinations:
        print(f"optimizer: {opt}, learning_rate: {lr}")
        run(f"python -m argument_graphs.run_argument_graph_submodel_training --argumentative_unit_classification --config configs/argumentative_unit_classification/auc__roberta-base.yaml --batch_size 16 --optimizer {opt} --learning_rate {lr} --gpus {GPU_ID}")
