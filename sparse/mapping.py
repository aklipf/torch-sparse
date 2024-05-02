from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F

from . import sparse


class Mapping:
    def __init__(
        self,
        source: sparse.SparseTensor,
        target: sparse.SparseTensor,
        mapping: torch.LongTensor,
    ):
        assert tuple(mapping.shape) == (target.indices.shape[1],)

        self._mapping = mapping
        self._source = source
        self._target = target

    @classmethod
    def repeat_last_dims(
        cls, source: sparse.SparseTensor, ndim: int = 1, repeat: int = 2
    ) -> Mapping:
        boadcasted_indices, mapping = cls._repeat_last_dims(
            source.indices, ndim, repeat
        )

        shape = source.shape[:-ndim] + tuple(
            sum([list(source.shape[-ndim:])] * repeat, [])
        )
        target = sparse.SparseTensor(boadcasted_indices, shape=shape, sort=False)
        return cls(source=source, target=target, mapping=mapping)

    @classmethod
    def _repeat_last_dims(
        cls, indices: torch.LongTensor, ndim: int, repeat: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        assert ndim > 0 and repeat > 1
        assert ndim <= indices.shape[0]

        # calculate base and reprtition
        unique_indices, count = Mapping._count_repeating_indices(indices[:-ndim])
        count_repeat = count.pow(repeat)

        repeated_base = unique_indices.repeat_interleave(count_repeat, dim=1)

        # calculating indexing of the repeated dims
        ptr_base = F.pad(count.cumsum(0), (1, 0))
        ptr_top = F.pad(count_repeat.cumsum(0), (1, 0))
        batch_top = torch.arange(
            count_repeat.shape[0], dtype=torch.long, device=indices.device
        ).repeat_interleave(count_repeat)
        idx_top = (
            torch.arange(batch_top.shape[0], dtype=torch.long, device=indices.device)
            - ptr_top[batch_top]
        )

        exp_top = F.pad(
            count[None, :].repeat(repeat - 1, 1).cumprod(0), (0, 0, 1, 0), value=1
        ).flip(0)

        idx_top = (
            idx_top[None, :] // exp_top[:, batch_top] % count[None, batch_top]
            + ptr_base[None, batch_top]
        )

        # repeat top dims and concatenate with base
        repeated_top = (
            indices[-ndim:, idx_top].swapdims(0, 1).reshape(ndim * repeat, -1)
        )
        result_indices = torch.cat((repeated_base, repeated_top), dim=0)

        return result_indices, idx_top[0]

    @staticmethod
    def _count_repeating_indices(
        indices: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        change = F.pad(
            (indices[:, :-1] != indices[:, 1:]).any(dim=0), (1, 0), value=True
        )
        _, count = torch.unique_consecutive(change.cumsum(0), return_counts=True)
        unique_indices = indices[:, change]

        return unique_indices, count

    def is_source(self, tensor: sparse.SparseTensor) -> bool:
        return id(self._source.indices) == id(tensor.indices)

    def is_target(self, tensor: sparse.SparseTensor) -> bool:
        return id(self._target.indices) == id(tensor.indices)

    def create_source(self, values: torch.Tensor | None = None) -> sparse.SparseTensor:
        return self._source.create_shared(values)

    def create_target(self, values: torch.Tensor | None = None) -> sparse.SparseTensor:
        return self._target.create_shared(values)

    @property
    def source(self) -> sparse.SparseTensor:
        return self._source

    @property
    def target(self) -> sparse.SparseTensor:
        return self._target

    @property
    def mapping(self) -> torch.LongTensor:
        return self._mapping
