import torch

from .typing import Self
from .base import BaseSparse


class SparseTypeMixin(BaseSparse):

    def type(self, dtype: type) -> Self:
        if self.values is None:
            return self.__class__(
                self.indices,
                torch.ones_like(self.indices[0], dtype=dtype),
                self.shape,
            )

        return self.__class__(
            self.indices,
            self.values.type(dtype),
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
