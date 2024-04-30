from __future__ import annotations

import torch

from .sparse import SparseTensor


class Mapping:
    def __init__(
        self, source: SparseTensor, target: SparseTensor, indices: torch.LongTensor
    ):
        self._mapping = indices
        self._source = source
        self._target = target

    @classmethod
    def broadcast(cls, source: SparseTensor, target_dims: tuple) -> Mapping:
        pass

    def is_source(self, tensor: SparseTensor) -> bool:
        return id(self._source.indices) == id(tensor.indices)

    def is_target(self, tensor: SparseTensor) -> bool:
        return id(self._target.indices) == id(tensor.indices)

    @property
    def source_indices(self) -> torch.LongTensor:
        return self._source.indices

    @property
    def target_indices(self) -> torch.LongTensor:
        return self._target.indices

    @property
    def source_shape(self) -> tuple:
        return self._source.shape

    @property
    def target_shape(self) -> tuple:
        return self._target.shape

    @property
    def mapping(self) -> torch.LongTensor:
        return self._mapping
