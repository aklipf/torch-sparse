from typing import List, Callable, Tuple

import torch
import torch.nn.functional as F

from .typing import Self
from .base import BaseSparse


class SparseIndexingMixin(BaseSparse):  # TODO: add unit test with None in shape

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        print("__getitem__", idx)


a = SparseIndexingMixin(
    torch.tensor([[0, 3, 1, 1, 2, 2, 3], [0, 0, 1, 2, 1, 2, 3]], dtype=torch.long),
    torch.tensor([[1], [5], [1], [1], [1], [1], [1]], dtype=torch.int32),
    shape=(4, 4),
)

a[0]
a[:, 0]
a[..., 0]
a[2, 3]
a[2, None, 3, None]
