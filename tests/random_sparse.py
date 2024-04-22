from typing import Iterable

import torch


def randint_sparse(
    shape: Iterable[int], ratio: float = 0.1, min_v: int = 0, max_v: int = 16
):
    random = torch.rand(shape)
    mask = random < ratio
    indices = mask.int().nonzero().t()
    values = torch.randint(min_v, max_v, (indices.shape[1],)).float()

    return indices, values
