from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F

from .typing import Self
from .base import BaseSparse


class SparseShapeMixin(BaseSparse):

    def unsqueeze_(self, dim: int) -> Self:
        assert isinstance(dim, int)

        insert_zeros = torch.zeros(
            (1, self.indices.shape[1]), dtype=torch.long, device=self.indices.device
        )

        self.indices = torch.cat(
            (self.indices[:dim], insert_zeros, self.indices[dim:]), dim=0
        )
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        self.shape = tuple(new_shape)

        return self

    def squeeze_(self, dim: int = None) -> Self:
        assert dim is None or isinstance(dim, int)
        assert dim is None or dim <= self.indices.shape[0] and self.shape[dim] == 1

        if dim is None:
            keep_dim = [d for d, n in enumerate(self.shape) if n != 1]
        else:
            keep_dim = [d for d, _ in enumerate(self.shape) if d != dim]

        self.indices = self.indices[keep_dim]
        self.shape = tuple([self.shape[d] for d in keep_dim])

        return self

    def unsqueeze(self, dim: int) -> Self:
        sparse = self.clone()
        sparse.unsqueeze_(dim)

        return sparse

    def squeeze(self, dim: int) -> Self:
        sparse = self.clone()
        sparse.squeeze_(dim)

        return sparse

    def reshape_(self, shape: Iterable[int]) -> Self:
        shape = list(shape)

        indices, shape = self._indices_to_shape(shape)

        self.indices = indices
        self.shape = shape

        return self

    def reshape(self, shape: Iterable[int]) -> Self:
        shape = list(shape)

        indices, shape = self._indices_to_shape(shape)

        return self.__class__(indices, self.values, shape)

    def numel(self) -> int:
        total_size = 1
        for s in self.shape:
            total_size *= s

        return total_size

    def _indices_to_shape(self, shape: List[int]) -> Tuple[torch.LongTensor, List[int]]:
        numel = self.numel()

        num_anon = sum(map(lambda x: x == -1, shape))
        assert num_anon <= 1

        if num_anon == 1:
            total = self._prod(filter(lambda x: x != -1, shape))
            anon_dim = numel // total
            shape = list(map(lambda x: anon_dim if x == -1 else x, shape))

        assert self._prod(shape) == numel

        global_index = self._global_index()

        shape_tensor = torch.tensor(shape, dtype=torch.long, device=self.device)
        prod_tensor = F.pad(shape_tensor, (1, 0), value=1).cumprod(0)[:-1]

        indices = (global_index[None, :] // prod_tensor[:, None]) % shape_tensor[
            :, None
        ]

        return indices, shape
