import torch

from .typing import Self
from .base import BaseSparse


class SparseTypeMixin(BaseSparse):

    def type(self, dtype: type) -> Self:
        if self._values is None:
            raise ValueError(
                "Cannot convert the type of a sparse tensor without stored values."
            )

        return self.__class__(
            self._indices.clone(),
            self._values.type(dtype),
            self.shape,
        )

    def float(self) -> Self:
        return self.type(torch.float32)

    def double(self) -> Self:
        return self.type(torch.float64)

    def int(self) -> Self:
        return self.type(torch.int32)

    def long(self) -> Self:
        return self.type(torch.int64)

    def bool(self) -> Self:
        return self.type(torch.bool)
