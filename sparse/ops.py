from typing import List, Callable

import torch

from .typing import Self
from .base import BaseSparse


class SparseOpsMixin(BaseSparse):
    @classmethod
    def _intersection_op(
        cls,
        tensors: List[Self],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        pass

    @classmethod
    def _union_op(
        cls,
        tensors: List[Self],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        pass

    @classmethod
    def _set_minus_op(
        cls,
        tensors: List[Self],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        pass
