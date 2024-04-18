from typing import Iterable, List, Tuple
import math

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
        self.__shape = tuple(new_shape)

        return self

    def squeeze_(self, dim: int = None) -> Self:
        assert dim is None or isinstance(dim, int)
        assert dim is None or dim <= self.indices.shape[0] and self.shape[dim] == 1

        if dim is None:
            keep_dim = [d for d, n in enumerate(self.shape) if n != 1]
        else:
            keep_dim = [d for d, _ in enumerate(self.shape) if d != dim]

        self.indices = self.indices[keep_dim]
        self.__shape = tuple([self.shape[d] for d in keep_dim])

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
        return self._prod(self.shape)

    def _inferre_shape(self, shape: int | Iterable[int]) -> List[int]:
        if isinstance(shape, int):
            shape = [shape]
        else:
            shape = list(shape)

        num_inferred = sum(map(lambda x: x == -1, shape))

        if num_inferred > 1:
            raise ValueError("Shape cannot be inferred from more than one dimension")

        elif num_inferred == 1:
            numel = self.numel()
            total_known = self._prod(filter(lambda x: x != -1, shape))

            inferred_shape = numel // total_known
            assert total_known * inferred_shape == numel

            shape[shape.index(-1)] = inferred_shape

        return shape

    def _indices_to_shape(self, shape: List[int]) -> Tuple[torch.LongTensor, List[int]]:
        if math.log2(self.numel()) > 63.0:
            raise IndexError(
                "Cannot calculate a global index of more than 63 bits (Sparse tensor with numel()>2^63)"
            )

        shape = self._inferre_shape(shape)
        in_shape = torch.tensor(self.shape, dtype=torch.long, device=self.device)
        out_shape = torch.tensor(shape, dtype=torch.long, device=self.device)

        in_bases = F.pad(in_shape.cumprod(0), (1, 0), value=1)[:-1].flip((0,))
        out_bases = F.pad(out_shape.cumprod(0), (1, 0), value=1)[:-1].flip((0,))

        global_index = (self.indices * in_bases[:, None]).sum(dim=0)
        indices = (global_index[None, :] // out_bases[:, None]) % out_shape[:, None]

        return indices, shape


if __name__ == "__main__":
    indices_initial = torch.randint(0, 1024, (6, 16))
    sparse = SparseShapeMixin(
        indices_initial, shape=(1024, 1024, 1024, 1024, 1024, 1024), sort=True
    )
    indices_initial = sparse.indices
    sparse = SparseShapeMixin(
        indices_initial, shape=(1024, 1024, 1024, 1024, 1024, 1024)
    )

    indices, shape = sparse._indices_to_shape((1 << 30, 1 << 30))
    sparse = SparseShapeMixin(indices, shape=shape)
    indices, shape = sparse._indices_to_shape((1024, 1024, 1024, 1024, 1024, 1024))
    print(indices_initial)
    print(indices)
    print(indices == indices_initial, shape)
