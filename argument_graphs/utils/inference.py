"""
Utilities when running inference on ArgSum's submodels (specifically when
argumentative units are passed through the AUC and RTC submodels)
"""

from typing import List

def generate_batches(
    all_items: List[str],
    batch_size: int = 16,
    drop_last: bool = False
) -> List[List[str]]:
    """
    This batches the argumentative units to allow them to be passed in parallel
    through the submodels, allowing for faster inference

    Note that the last batch may be smaller than the `batch_size` (i.e. if the
    length of `all_items` is not divisible by the `batch_size`)
    - If `drop_last=True`, we will drop that last batch; otherwise, we include
      it as well

    Usage (run from root)
        ```
        from transformers import AutoTokenizer
        from argument_graphs.utils import generate_batches
        from argument_graphs.submodels import ArgumentativeUnitClassificationModel

        model = ArgumentativeUnitClassificationModel.load_from_checkpoint(...)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        all_argumentative_units = ["abc", "def" ..., "xyz"]
        batches = generate_batches(all_argumentative_units)

        for batch in batches:
            encoded_inputs = tokenizer(batch, ...)
            preds = model(encoded_inputs)[0]
            for item_pred in preds:
                ...
        ```
    """
    all_batches = list()

    for i in range(0, len(all_items), batch_size):
        batch = all_items[i : (i + batch_size)]
        all_batches.append(batch)

    if len(all_batches[-1]) != batch_size and drop_last == True:
        # Drop the last batch, since it is not of size `batch_size`
        all_batches = all_batches[:-1]

    return all_batches
