import torch

from .typing import Self
from .base import BaseSparse


class SparseMaskMixin(BaseSparse):

    def mask_(self, mask: torch.BoolTensor) -> Self:
        assert mask.ndim == 1 and mask.shape[0] == self._indices.shape[1]

        self._indices = self._indices[:, mask]

        if self._values is not None:
            self._values = self._values[mask]

        return self

    def mask(self, mask: torch.BoolTensor) -> Self:
        sparse = self.clone()
        sparse.mask_(mask)

        return sparse
