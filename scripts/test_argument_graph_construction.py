"""
Run (from root)
```
python -m scripts.test_argument_graph_construction
```
"""

from argument_graphs.modules import ArgumentGraphConstructor

if __name__ == "__main__":
    graph_constructor = ArgumentGraphConstructor(
        auc_trained_model_version=206,
        rtc_trained_model_version=264
    )
