from typing import Literal, List, Tuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from .typing import Self
from .base import BaseSparse


class SparseScatterMixin(BaseSparse):

    def sum(self, dim: int | tuple = None) -> Self:
        return self.scatter(dim, "sum")

    def mean(self, dim: int | tuple = None) -> Self:
        return self.scatter(dim, "mean")

    def scatter(
        self, dim: int | tuple = None, reduce: Literal["sum", "mean"] = "sum"
    ) -> Self:
        dim = self._dim_to_list(dim)
        dim = sorted(dim, reverse=True)

        if len(dim) == len(self.shape):
            return self._scatter_all(reduce)

        indices, batch = self._unique_index(dim)

        if self.values is None:
            values = scatter(
                torch.ones_like(self.indices[0], dtype=self.dtype),
                batch,
                dim=0,
                reduce=reduce,
            )
        else:
            values = scatter(self.values, batch, dim=0, reduce=reduce)

        shape = list(
            map(lambda x: self.shape[x], set(range(len(self.shape))) - set(dim))
        )

        return self.__class__(indices, values, shape)

    def _scatter_all(self, reduce: Literal["sum", "mean"] = "sum") -> Self:
        indices = torch.tensor([[0]], dtype=torch.long, device=self.device)

        if reduce == "sum":
            if self.values is None:
                value = self.indices.shape[1]
            else:
                value = self.values.sum().item()

        elif reduce == "mean":
            if self.values is None:
                value = self.indices.shape[1] / self.numel()
            else:
                value = self.values.mean().item()

        values = torch.tensor([value], dtype=self.dtype, device=self.device)

        return self.__class__(indices, values, shape=(1,))

    def _unique_index(
        self, without_dim: List[int]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        keep_dim = sorted(list(set(range(len(self.shape))) - set(without_dim)))

        unique_index = self._global_index(without_dim)

        unique, batch = torch.unique(unique_index, return_inverse=True)

        indices = torch.zeros(
            (len(keep_dim), unique.shape[0]), dtype=torch.long, device=self.device
        )
        indices[:, batch] = self.indices[keep_dim]

        return indices, batch
