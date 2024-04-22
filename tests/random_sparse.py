import torch
from typing import Tuple, Iterable


def randint_sparse(
    shape: Iterable[int], ratio: float = 0.1, min: int = 0, max: int = 16
):
    random = torch.rand(shape)
    mask = random < ratio
    indices = mask.int().nonzero().t()
    values = torch.randint(min, max, (indices.shape[1],)).float()

    return indices, values
