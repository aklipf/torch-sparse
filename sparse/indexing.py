from typing import Literal, List, Tuple, Iterable

import torch
import torch.nn.functional as F

from .typing import Self
from .base import BaseSparse


class SparseIndexingMixin(BaseSparse):

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

    def index_sorted(self, except_dim: int | Iterable = None) -> torch.LongTensor:
        dims = self.__included_dims(except_dim)

        diff = (self.indices[dims, 1:] != self.indices[dims, :-1]).any(dim=0)
        return F.pad(diff.cumsum(0), (1, 0), value=0)
