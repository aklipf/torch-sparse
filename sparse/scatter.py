from typing import Literal, Iterable, List

import torch

from .typing import Self
from .shape import SparseShapeMixin
from .mapping import Mapping


class SparseScatterMixin(SparseShapeMixin):

    def sum(self, reduction: int | tuple | Mapping = None) -> Self:
        return self.scatter(reduction, "sum")

    def mean(self, reduction: int | tuple | Mapping = None) -> Self:
        return self.scatter(reduction, "mean")

    def scatter(
        self,
        reduction: int | tuple | Mapping = None,
        reduce: Literal["sum", "mean"] = "sum",
    ) -> Self:
        assert (
            self.dtype not in (torch.bool, torch.int32, torch.int64) or reduce != "mean"
        ), "Mean reduction can be computed only on real or complex numbers"

        if isinstance(reduction, Mapping):
            assert reduction.is_target(self)

            dims = range(len(reduction.source.shape), len(reduction.target.shape))
            values = self._scatter_value(self, reduction.mapping, dims, reduce)

            return reduction.create_source(values)

        dims = self._dim_to_list(reduction)
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

        values = self._scatter_value(sorted_sparse, batch, dims, reduce)

        shape = tuple(
            map(lambda x: self.shape[x], set(range(len(self.shape))) - set(dims))
        )
        return self.__class__(indices, values, shape)

    @classmethod
    def _scatter_value(
        cls,
        sorted_sparse: Self,
        batch: torch.LongTensor,
        dims: Iterable[int],
        reduce: Literal["sum", "mean"],
    ) -> torch.Tensor:
        if sorted_sparse.values is None:
            values = torch.zeros(
                (batch[-1] + 1, 1),
                dtype=torch.long,
                device=sorted_sparse.indices.device,
            ).scatter_add_(
                dim=0,
                index=batch[:, None],
                src=torch.ones_like(sorted_sparse.indices[0][:, None]),
            )
        else:
            values = torch.zeros(
                (batch[-1] + 1, sorted_sparse.values.shape[1]),
                dtype=sorted_sparse.values.dtype,
                device=sorted_sparse.values.device,
            ).scatter_add_(
                dim=0,
                index=batch[:, None].expand_as(sorted_sparse.values),
                src=sorted_sparse.values,
            )

        if reduce == "mean":
            total = cls._prod(map(lambda i: sorted_sparse.shape[i], dims))
            values = values / total

        return values

    def _scatter_all(self, reduce: Literal["sum", "mean"] = "sum") -> Self:
        indices = torch.tensor([[0]], dtype=torch.long, device=self.device)

        if reduce == "sum":
            if self._values is None:
                value = self._indices.shape[1]
            else:
                value = self._values.sum().item()

        elif reduce == "mean":  # no mean without values (bool type)
            value = self._values.sum().item() / self.numel()

        if self._values is None:
            values = torch.tensor([value], dtype=torch.long, device=self.device)
        else:
            values = torch.tensor([value], dtype=self.dtype, device=self.device)

        return self.__class__(indices, values, shape=(1,))
