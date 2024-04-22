import torch

from .typing import Self
from .base import BaseSparse


class SparseMaskMixin(BaseSparse):

    def mask_(self, mask: torch.BoolTensor) -> Self:
        assert mask.ndim == 1 and mask.shape[0] == self.indices.shape[1]

        self.indices = self.indices[:, mask]

        if self.values is not None:
            self.values = self.values[mask]

        return self

    def mask(self, mask: torch.BoolTensor) -> Self:
        sparse = self.clone()
        sparse.mask_(mask)

        return sparse
