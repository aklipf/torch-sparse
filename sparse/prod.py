from typing import Iterable

from .typing import Self
from .base import BaseSparse


class SparseProdMixin(BaseSparse):

    @classmethod
    def prod(cls, sparse_tensors: Iterable[Self], dim: int) -> Self:
        """
        Cartesian product of sparse tensors over a dimension
        """
        pass
