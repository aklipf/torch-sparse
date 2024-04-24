from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from .typing import Self


class BaseSparse:
    def __init__(
        self,
        indices: torch.LongTensor,
        values: torch.Tensor = None,
        shape: tuple = None,
        sort: bool = True,
    ):
        assert indices.ndim == 2 and indices.dtype == torch.long
        assert values is None or (
            values.ndim in (1, 2) and indices.shape[1] == values.shape[0]
        )
        assert (shape is None and indices.shape[1] != 0) or isinstance(shape, tuple)
        assert values is None or indices.device == values.device

        if shape is None:
            shape = tuple((indices.amax(dim=1) + 1).tolist())
        else:
            indices = self._process_shape(indices, shape)

        if values is not None and values.ndim == 1:
            values.unsqueeze_(1)

        self.__shape = shape
        self.indices = indices
        self.values = values

        if sort and (not self._is_sorted()):
            self._sort_by_indices_()
            self._remove_sorted_duplicate_()

    @classmethod
    def _process_shape(
        cls, indices: torch.LongTensor, shape: tuple
    ) -> torch.LongTensor:
        dims = []
        current_dims = []
        count = 0
        for s in shape:
            if isinstance(s, int) and s > 0:
                current_dims.append(count)
                count += 1
            elif s is None:
                dims.append(current_dims)
                dims.append(None)
                current_dims = []
            else:
                raise ValueError("shape must be composed of int of None values")

        if len(current_dims) != 0:
            dims.append(current_dims)

        assert (
            count == indices.shape[0]
        ), "The number of dimension of the shape and the indices didn't match"

        if len(dims) == 1:
            return indices

        # didn't find a way to test this without gpu
        zeros = torch.zeros_like(indices[:1])
        cat_args = []
        for dims_subset in dims:
            if dims_subset is None:
                cat_args.append(zeros)
            else:
                cat_args.append(indices[dims_subset])

        return torch.cat(cat_args, dim=0)

    @property
    def shape(self) -> tuple:
        return self.__shape

    @property
    def real_shape(self) -> tuple:
        return tuple(map(lambda x: x if x is not None else 1, self.__shape))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> type:
        if self.values is None:
            return torch.bool

        return self.values.dtype

    @property
    def device(self) -> torch.device:
        return self.indices.device

    def to(self, device: torch.device) -> Self:
        return self.__class__(
            indices=self.indices.to(device),
            values=None if self.values is None else self.values.to(device),
            shape=self.real_shape,
        )._set_shape_(self.__shape)

    def clone(self) -> Self:
        return self.__class__(
            indices=self.indices.clone(),
            values=None if self.values is None else self.values.clone(),
            shape=self.real_shape,
        )._set_shape_(self.__shape)

    def detach(self) -> Self:
        return self.__class__(
            indices=self.indices.detach(),
            values=None if self.values is None else self.values.detach(),
            shape=self.real_shape,
        )._set_shape_(self.__shape)

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(shape={self.shape},
  indices={self.indices},
  values={self.values},
  device=\"{self.device}\")"""

    def to_dense(self) -> torch.Tensor:
        if self.values is None or self.values.shape[1] == 1:
            shape = self.real_shape
        else:
            shape = self.real_shape + self.values.shape[1:]

        x = torch.zeros(shape, dtype=self.dtype, device=self.device)
        indices = [self.indices[i] for i in range(self.indices.shape[0])]

        if self.values is None:
            x[indices] = 1
        elif self.values.shape[1] == 1:
            x[indices] = self.values.flatten()
        else:
            x[indices] = self.values

        return x

    @property
    def dims(self) -> tuple:
        return tuple(range(len(self.__shape)))

    @staticmethod
    def _get_ptr(idx: torch.LongTensor) -> torch.LongTensor:
        # pylint: disable=not-callable
        return F.pad(idx.cumsum(0), (1, 0), value=0)

    def index_sorted(self, except_dim: int | Iterable = None) -> torch.LongTensor:
        dims = self._included_dims(except_dim)

        diff = (self.indices[dims, 1:] != self.indices[dims, :-1]).any(dim=0)

        return self._get_ptr(diff)

    def _set_shape_(self, shape: tuple) -> Self:
        self.__shape = shape
        return self

    def _dim_to_list(self, dim: int | tuple = None) -> List[int]:
        if dim is None:
            return list(self.dims)

        if isinstance(dim, int):
            assert dim < self.ndim
            return [dim]

        lst_dim = sorted(list(dim))

        assert lst_dim[-1] < self.ndim
        assert len(lst_dim) == len(set(dim)), "multiple dimensions are the same"

        return lst_dim

    def _included_dims(self, except_dim: int | Iterable = None) -> List[int]:
        if isinstance(except_dim, int):
            except_dim = {except_dim}
        elif except_dim is None:
            except_dim = set()
        else:
            except_dim = set(except_dim)

        return sorted(list(set(self.dims) - except_dim))

    @classmethod
    def _argsort_indices(
        cls, indices: torch.LongTensor, dims: List[int]
    ) -> torch.LongTensor:
        perm = None

        for i in reversed(dims):

            if perm is None:
                current_indices = indices[i]
            else:
                current_indices = indices[i, perm]

            current_perm = torch.argsort(current_indices, stable=True)

            if perm is None:
                perm = current_perm
            else:
                perm = perm[current_perm]

        return perm

    def _sort_by_indices_(self, except_dim: int | Iterable = None):
        """Sort indices and values"""
        dims = self._included_dims(except_dim)
        perm = self._argsort_indices(self.indices, dims)

        # apply reindexing
        self.indices = self.indices[:, perm]

        if self.values is not None:
            self.values = self.values[perm]

    def _remove_sorted_duplicate_(self):
        # pylint: disable=not-callable
        mask = F.pad(
            (self.indices[:, 1:] != self.indices[:, :-1]).any(dim=0), (1, 0), value=True
        )

        self.indices = self.indices[:, mask]

        if self.values is not None:
            batch = self._get_ptr(mask)[:-1]
            self.values = scatter_add(self.values, batch, dim=0)

    def _is_sorted(self) -> bool:
        sorted_mask = self.indices.diff(dim=1) <= 0
        unsorted_mask = self.indices.diff(dim=1) >= 0

        dims = torch.tensor([self.dims], device=self.device).t()
        dims = dims.repeat(1, sorted_mask.shape[1])

        min_sorted = dims.clone()
        min_sorted[sorted_mask] = len(self.__shape)
        min_sorted = min_sorted.amin(dim=0)

        min_unsorted = dims.clone()
        min_unsorted[unsorted_mask] = len(self.__shape)
        min_unsorted = min_unsorted.amin(dim=0)

        return (min_sorted < min_unsorted).all()

    @classmethod
    def _prod(cls, values: Iterable[int]) -> int:
        result = 1
        for value in values:
            if value is None:
                continue

            result *= int(value)
        return result
