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
        self._set_shape_(tuple(new_shape))

        return self

    def squeeze_(self, dim: int = None) -> Self:
        assert dim is None or isinstance(dim, int)
        assert dim is None or dim <= self.indices.shape[0] and self.shape[dim] == 1

        if dim is None:
            keep_dim = [d for d, n in enumerate(self.shape) if n != 1]
        else:
            keep_dim = [d for d, _ in enumerate(self.shape) if d != dim]

        self.indices = self.indices[keep_dim]
        self._set_shape_(tuple(map(lambda d: self.shape[d], keep_dim)))

        return self

    def unsqueeze(self, dim: int) -> Self:
        sparse = self.clone()
        sparse.unsqueeze_(dim)

        return sparse

    def squeeze(self, dim: int) -> Self:
        sparse = self.clone()
        sparse.squeeze_(dim)

        return sparse

    def reshape_(self, shape: int | Iterable[int]) -> Self:
        indices, shape = self._indices_to_shape(shape)

        self.indices = indices
        self._set_shape_(shape)

        return self

    def reshape(self, shape: int | Iterable[int]) -> Self:
        cloned = self.clone()
        cloned.reshape_(shape)

        return cloned

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

        if num_inferred == 1:
            numel = self.numel()
            total_known = self._prod(filter(lambda x: x != -1, shape))

            inferred_shape = numel // total_known
            assert total_known * inferred_shape == numel

            shape[shape.index(-1)] = inferred_shape

        return shape

    def _indexing_from_shape(
        self, shape: List[int]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        tensor_shape = torch.tensor(shape, dtype=torch.long, device=self.device)
        # pylint: disable=not-callable
        tensor_offset = F.pad(
            tensor_shape.flip((0,)).cumprod(0)[:-1], (1, 0), value=1
        ).flip((0,))

        return tensor_offset, tensor_shape

    def _indices_to_shape(
        self, shape: int | Iterable[int]
    ) -> Tuple[torch.LongTensor, List[int]]:
        if math.log2(self.numel()) > 63.0:
            raise IndexError(
                "Cannot calculate a global index of more than 63 bits (numel()>2^63)"
            )

        shape = self._inferre_shape(shape)

        in_bases, _ = self._indexing_from_shape(self.shape)
        out_bases, out_shape = self._indexing_from_shape(shape)

        global_index = (self.indices * in_bases[:, None]).sum(dim=0)
        indices = (global_index[None, :] // out_bases[:, None]) % out_shape[:, None]

        return indices, shape
