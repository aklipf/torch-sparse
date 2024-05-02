from typing import Iterable, List

import torch
import torch.nn.functional as F

from .typing import Self


class BaseSparse:
    def __init__(
        self,
        indices: torch.LongTensor,
        values: torch.Tensor = None,
        shape: tuple = None,
        sort: bool = True,
    ):
        self._indices = BaseSparse._process_indices(indices, values)
        self._values = BaseSparse._process_values(values)
        self.__shape = BaseSparse._process_shape(indices, shape)

        if sort and (not self._is_sorted()):
            self._sort_by_indices_()
            self._remove_sorted_duplicate_()

    def create_shared(self, values: torch.Tensor | None = None) -> Self:
        assert values is None or values.shape[0] == self._indices.shape[1]

        return self.__class__(self._indices, values, self.shape, sort=False)

    @staticmethod
    def _process_shape(indices: torch.LongTensor, shape: tuple) -> tuple:
        assert (shape is None and indices.shape[1] != 0) or isinstance(shape, tuple)

        if shape is None:
            shape = tuple((indices.amax(dim=1) + 1).tolist())
        else:
            assert len(shape) == indices.shape[0]
            assert all(map(lambda x: isinstance(x, int), shape))

        return shape

    @staticmethod
    def _process_indices(
        indices: torch.LongTensor, values: torch.Tensor
    ) -> torch.LongTensor:
        assert indices.ndim == 2 and indices.dtype == torch.long
        assert values is None or (
            values.ndim in (1, 2) and indices.shape[1] == values.shape[0]
        )
        assert values is None or indices.device == values.device

        return indices

    @staticmethod
    def _process_values(values: torch.Tensor) -> torch.Tensor:
        if values is not None and values.ndim == 1:
            values.unsqueeze_(1)

        return values

    @property
    def indices(self) -> torch.LongTensor:
        return self._indices

    @property
    def values(self) -> torch.Tensor:
        return self._values

    @property
    def shape(self) -> tuple:
        return self.__shape

    @property
    def ndim(self) -> int:
        return len(self.__shape)

    @property
    def dim(self) -> int:
        return len(self.__shape)

    @property
    def dtype(self) -> type:
        if self._values is None:
            return torch.bool

        return self._values.dtype

    @property
    def device(self) -> torch.device:
        return self._indices.device

    def to(self, device: torch.device) -> Self:
        return self.__class__(
            indices=self._indices.to(device),
            values=None if self._values is None else self._values.to(device),
            shape=self.shape,
        )

    def clone(self) -> Self:
        return self.__class__(
            indices=self._indices.clone(),
            values=None if self._values is None else self._values.clone(),
            shape=self.shape,
        )

    def detach(self) -> Self:
        return self.__class__(
            indices=self._indices.detach(),
            values=None if self._values is None else self._values.detach(),
            shape=self.shape,
        )

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(shape={self.shape},
  indices={self._indices},
  values={self._values},
  device=\"{self.device}\")"""

    def to_dense(self) -> torch.Tensor:
        if self._values is None or self._values.shape[1] == 1:
            shape = self.shape
        else:
            shape = self.shape + self._values.shape[1:]

        dense_matrix = torch.zeros(shape, dtype=self.dtype, device=self.device)
        indices = [self._indices[i] for i in range(self._indices.shape[0])]

        if self._values is None:
            dense_matrix[indices] = 1
        elif self._values.shape[1] == 1:
            dense_matrix[indices] = self._values.flatten()
        else:
            dense_matrix[indices] = self._values

        return dense_matrix

    @property
    def dims(self) -> tuple:
        return tuple(range(len(self.__shape)))

    @staticmethod
    def _get_ptr(idx: torch.LongTensor) -> torch.LongTensor:
        # pylint: disable=not-callable
        return F.pad(idx.cumsum(0), (1, 0), value=0)

    def index_sorted(self, except_dim: int | Iterable = None) -> torch.LongTensor:
        dims = self._included_dims(except_dim)

        diff = (self._indices[dims, 1:] != self._indices[dims, :-1]).any(dim=0)

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
        perm = self._argsort_indices(self._indices, dims)

        # apply reindexing
        self._indices = self._indices[:, perm]

        if self._values is not None:
            self._values = self._values[perm]

    def _remove_sorted_duplicate_(self):
        # pylint: disable=not-callable
        mask = F.pad(
            (self._indices[:, 1:] != self._indices[:, :-1]).any(dim=0),
            (1, 0),
            value=True,
        )

        self._indices = self._indices[:, mask]

        if self._values is not None:
            batch = self._get_ptr(mask)[:-1, None]
            self._values = torch.zeros(
                (batch[-1] + 1, self._values.shape[1]),
                dtype=self._values.dtype,
                device=self._values.device,
            ).scatter_add_(dim=0, index=batch.expand_as(self._values), src=self._values)

    def _is_sorted(self) -> bool:
        sorted_mask = self._indices.diff(dim=1) <= 0
        unsorted_mask = self._indices.diff(dim=1) >= 0

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
