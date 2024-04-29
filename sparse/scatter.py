from typing import Literal

import torch

from .typing import Self
from .shape import SparseShapeMixin


class SparseScatterMixin(SparseShapeMixin):

    def sum(self, dim: int | tuple = None) -> Self:
        return self.scatter(dim, "sum")

    def mean(self, dim: int | tuple = None) -> Self:
        return self.scatter(dim, "mean")

    def scatter(
        self, dims: int | tuple = None, reduce: Literal["sum", "mean"] = "sum"
    ) -> Self:
        assert (
            self.dtype
            not in (torch.bool, torch.int, torch.int32, torch.int64, torch.long)
            or reduce != "mean"
        ), "Mean reduction can be computed only on real or complex numbers"

        dims = self._dim_to_list(dims)
        dims = sorted(dims, reverse=True)

        if len(dims) == len(self.shape):
            return self._scatter_all(reduce)

        sorted_sparse = self
        keeped_dims = self._included_dims(dims)
        if min(dims) < max(keeped_dims):
            sorted_sparse = self.clone()
            # pylint: disable=protected-access
            sorted_sparse._sort_by_indices_(dims)

        batch = sorted_sparse.index_sorted(dims)
        indices = torch.empty(
            (len(keeped_dims), batch[-1] + 1),
            dtype=torch.long,
            device=sorted_sparse.device,
        )
        indices[:, batch] = sorted_sparse.indices[keeped_dims]

        if self.values is None:
            values = torch.zeros(
                (indices.shape[1], 1),
                dtype=torch.long,
                device=sorted_sparse.indices.device,
            ).scatter_add_(
                dim=0,
                index=batch[:, None],
                src=torch.ones_like(sorted_sparse.indices[0][:, None]),
            )
        else:
            values = torch.zeros(
                (indices.shape[1], sorted_sparse.values.shape[1]),
                dtype=sorted_sparse.values.dtype,
                device=sorted_sparse.values.device,
            ).scatter_add_(
                dim=0,
                index=batch[:, None].expand_as(sorted_sparse.values),
                src=sorted_sparse.values,
            )

        if reduce == "mean":
            total = self._prod(map(lambda i: self.shape[i], dims))
            values = values / total

        shape = tuple(
            map(lambda x: self.shape[x], set(range(len(self.shape))) - set(dims))
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
                value = self.values.sum().item() / self.numel()

        if self.values is None:
            values = torch.tensor([value], dtype=torch.long, device=self.device)
        else:
            values = torch.tensor([value], dtype=self.dtype, device=self.device)

        return self.__class__(indices, values, shape=(1,))
