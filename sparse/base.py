from __future__ import annotations
from typing import Iterable, List, Tuple, Literal, Any

import torch
import torch.nn.functional as F

from .typing import Self


class BaseSparse(object):
    def __init__(
        self,
        indices: torch.LongTensor,
        values: torch.Tensor = None,
        shape: tuple = None,
        sort: bool = True,
    ):
        assert indices.ndim == 2 and indices.dtype == torch.long
        assert values is None or values.ndim in (1, 2)
        assert values is None or indices.shape[1] == values.shape[0]
        assert shape is None or indices.shape[0] == len(shape)

        if shape is None:
            shape = tuple((indices.amax(dim=1) + 1).tolist())

        if values is not None and values.ndim == 1:
            values.unsqueeze_(1)

        self.shape = tuple(shape)
        self.indices = indices
        self.values = values

        if sort and (not self.is_sorted()):
            self.sort_indices_()

    def __included_dims(self, except_dim: int | Iterable = None) -> List[int]:
        if isinstance(except_dim, int):
            except_dim = {except_dim}
        elif except_dim is None:
            except_dim = set()
        else:
            except_dim = set(except_dim)

        return sorted(list(set(self.dims) - except_dim))

    def argsort_indices(self, except_dim: int | Iterable = None) -> torch.LongTensor:
        dims = self.__included_dims(except_dim)

        perm = None

        for i in reversed(dims):

            if perm is None:
                current_indices = self.indices[i]
            else:
                current_indices = self.indices[i, perm]

            current_perm = torch.argsort(current_indices, stable=True)

            if perm is None:
                perm = current_perm
            else:
                perm = perm[current_perm]

        return perm

    def sort_indices_(self):
        """Sort indices and values"""

        perm = self.argsort_indices()

        # apply reindexing
        self.indices = self.indices[:, perm]

        if self.values is not None:
            self.values = self.values[:, perm]

    def is_sorted(self) -> bool:
        return (self.indices[:, 1:] <= self.indices[:, :-1]).all()

    @property
    def dims(self) -> tuple:
        return tuple(range(len(self.shape)))

    @property
    def dtype(self) -> type:
        if self.values is None:
            return torch.float32

        return self.values.dtype

    @property
    def device(self) -> torch.device:
        return self.indices.device

    def to(self, device: torch.device) -> Self:
        if self.values is None:
            return self.__class__(
                indices=self.indices.to(device),
                values=None,
                shape=self.shape,
            )

        return self.__class__(
            indices=self.indices.to(device),
            values=self.values.to(device),
            shape=self.shape,
        )

    def clone(self) -> Self:
        if self.values is None:
            return self.__class__(
                indices=self.indices.clone(), values=None, shape=self.shape
            )

        return self.__class__(
            indices=self.indices.clone(), values=self.values.clone(), shape=self.shape
        )

    def detach(self) -> Self:
        if self.values is None:
            return self.__class__(
                indices=self.indices.detach(), values=None, shape=self.shape
            )

        return self.__class__(
            indices=self.indices.detach(), values=self.values.detach(), shape=self.shape
        )

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

    def __repr__(self) -> str:
        if self.values is None:
            return f"""{self.__class__.__name__}(shape={self.shape},
  indices={self.indices},
  device=\"{self.device}\")"""

        return f"""{self.__class__.__name__}(shape={self.shape},
  indices={self.indices},
  values={self.values},
  device=\"{self.device}\")"""

    @property
    def dense(self) -> torch.Tensor:
        if self.values is None:
            shape = self.shape
        else:
            shape = self.shape + self.values.shape[1:]

        x = torch.zeros(shape, dtype=self.dtype, device=self.device)
        indices = [self.indices[i] for i in range(self.indices.shape[0])]

        if self.values is None:
            x[indices] = 1
        else:
            x[indices] = self.values

        return x

    @classmethod
    def _dim_to_list(cls, dim: int | tuple = None) -> List[int]:
        if dim is None:
            return []

        elif isinstance(dim, int):
            return [dim]

        lst_dim = list(dim)

        assert len(lst_dim) == len(set(dim)), "multiple dimensions are the same"

        return lst_dim

    def _global_index(self, without_dim: List[int] = None) -> torch.LongTensor:
        shape = torch.tensor(self.shape, dtype=torch.long)

        if without_dim is not None:
            shape[without_dim] = 1

        prod_tensor = F.pad(shape, (1, 0), value=1).cumprod(0)

        if without_dim is not None:
            prod_tensor[without_dim] = 0

        return (self.indices * prod_tensor.to(self.device)[:-1, None]).sum(dim=0)

    @classmethod
    def _prod(cls, values: Iterable[int]) -> int:
        result = 1
        for value in values:
            result *= int(value)
        return result
