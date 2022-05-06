"""
This script tries out computing 2D and 3D outer products efficiently in PyTorch
- The findings from here are implemented in the multi-modal models using low
  rank tensor fusion

Run (from root)
```
python -m scripts.experiment_low_rank_fusion
```
"""

import torch
import torch.nn as nn

def test_out_outer_product_computation():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    b = torch.tensor([5, 6, 7], dtype=torch.float)
    c = torch.tensor([8, 9, 10, 11, 12], dtype=torch.float)
    print(a)
    print(b)
    print(c)

    # Note: `torch.outer()` only works on 2 tensors
    # https://pytorch.org/docs/stable/generated/torch.outer.html
    _outer_product_2d = torch.outer(a, b)
    print(_outer_product_2d)

    # Note: `torch.einsum()` can evaluate any operation in Einstein summation
    # notation, and thus can compute the outer product for n tensors
    # https://pytorch.org/docs/stable/generated/torch.einsum.html
    outer_product_2d = torch.einsum('i,j->ij', a, b)
    print(outer_product_2d)

    outer_product_3d = torch.einsum('i,j,k->ijk', a, b, c)
    print(outer_product_3d)

def outer_product_to_embedding():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    b = torch.tensor([5, 6, 7], dtype=torch.float)
    c = torch.tensor([8, 9, 10, 11, 12], dtype=torch.float)
    print(a)
    print(b)
    print(c)

    outer_product_3d = torch.einsum('i,j,k->ijk', a, b, c)
    print(outer_product_3d)
    print(outer_product_3d.shape)

    input = torch.flatten(outer_product_3d)
    # batch = outer_product_3d[None, :, :, :]
    batch = input.unsqueeze(0)
    print(batch)
    print(batch.shape)
    batch_size, input_dim = batch.shape
    print(input_dim)

    # nn.Linear() allows you to pass any-dimensional tensors as input, as long
    # as the last dimension of the n-dimensional tensor matches the in_features
    # parameter
    embedding_dim = 30
    embedder = torch.nn.Linear(in_features=input_dim, out_features=embedding_dim)

    embedding = embedder(batch)
    print(embedding)
    print(embedding.shape)

if __name__ == "__main__":
    # test_out_outer_product_computation()
    outer_product_to_embedding()
