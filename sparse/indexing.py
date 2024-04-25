from typing import Iterable

from .base import BaseSparse


class SparseIndexingMixin(BaseSparse):

    def __getitem__(self, indexing: Iterable[slice | None]):
        if not isinstance(indexing, tuple):
            indexing = (indexing,)

        result = self.clone()
        for i, idx in enumerate(indexing):
            assert idx == slice(None) or idx is None

            if idx is None:
                result.unsqueeze_(i)

        return result
