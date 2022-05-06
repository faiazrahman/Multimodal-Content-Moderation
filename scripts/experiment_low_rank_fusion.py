"""
Run (from root)
```
python -m scripts.experiment_low_rank_fusion
```
"""

import torch

def test_out_outer_product_computation():
    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([5, 6, 7])
    c = torch.tensor([8, 9, 10, 11, 12])
    print(a)
    print(b)
    print(c)

    # Note: `torch.outer()` only works on 2 tensors
    _outer_product_2d = torch.outer(a, b)
    print(_outer_product_2d)

    # Note: `torch.einsum()` can evaluate any operation in Einstein summation
    # notation, and thus can compute the outer product for n tensors
    outer_product_2d = torch.einsum('i,j->ij', a, b)
    print(outer_product_2d)

    outer_product_3d = torch.einsum('i,j,k->ijk', a, b, c)
    print(outer_product_3d)

if __name__ == "__main__":
    test_out_outer_product_computation()
