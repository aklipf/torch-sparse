from typing import List, Callable, Tuple

import torch
import torch.nn.functional as F

from .typing import Self
from .base import BaseSparse


def _intersection_mask(indices: torch.LongTensor, n_tensors: int) -> torch.BoolTensor:
    equal = (indices[:, 1:] != indices[:, :-1]).any(dim=0)
    equal_batch = F.pad(equal.cumsum(0), (1, 0), value=0)
    return equal_batch[: -(n_tensors - 1)] == equal_batch[n_tensors - 1 :]


class SparseOpsMixin(BaseSparse):

    def __and__(self, other: Self):
        assert self.dtype == other.dtype == torch.bool

        return self._generic_ops([self, other], _intersection_mask)

    def __mul__(self, other: Self):
        assert self.dtype == other.dtype

        return self._generic_ops(
            [self, other], _intersection_mask, lambda x: x.prod(dim=1)
        )

    @classmethod
    def _generic_ops(
        cls,
        tensors: List[Self],
        indices_mask: Callable[[torch.LongTensor, int], torch.BoolTensor],
        ops: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        assert len(tensors) > 1
        shape = cls._get_shape(tensors)
        device = tensors[0].device
        dims = tuple(range(len(shape)))

        # concatenating indices and values
        indices, values = cls._cat_values(tensors)

        # sorting
        perm = cls._argsort_indices(indices, dims=dims)
        indices = indices[:, perm]

        # keep indices when at least len(tensors) values are present (calculate intersection over the indices)
        mask = indices_mask(indices, len(tensors))

        # filter indices based on intersection
        indices = indices[:, : -(len(tensors) - 1)][:, mask]

        # filter values and performe the operation
        if values is not None:
            values = values[perm]

            # select the values
            indices_idx = torch.arange(
                values.shape[0] - len(tensors) + 1, dtype=torch.long, device=device
            )[mask]
            diff_idx = torch.arange(len(tensors), dtype=torch.long, device=device)
            idx = indices_idx[:, None] + diff_idx[None, :]

            if ops is None:  # concatenation if no ops
                values = values[idx].reshape(idx.shape[0], -1)
            else:
                values = ops(values[idx]).type(values.dtype)

            # filter nul values
            mask = (values != 0).all(dim=1)
            indices, values = indices[:, mask], values[mask]

        # create a new sparse tensor containing the result
        return cls(indices, values=values, shape=shape)

    @classmethod
    def _get_shape(cls, tensors: List[Self]) -> tuple:
        assert len(tensors) > 0

        shape = tensors[0].shape
        for t in tensors[1:]:
            assert shape == t.shape

        return shape

    @classmethod
    def _cat_values(cls, tensors: List[Self]) -> Tuple[torch.LongTensor, torch.Tensor]:
        indices = torch.cat([t.indices for t in tensors], dim=1)
        if tensors[0].values is None:
            values = None
        else:
            values = torch.cat([t.values for t in tensors], dim=0)

        return indices, values


a = SparseOpsMixin(
    torch.tensor([[0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 3]], dtype=torch.long),
    # torch.tensor([[1], [1], [1], [1], [1], [1]], dtype=torch.int32),
    shape=(4, 4),
)
b = SparseOpsMixin(
    torch.tensor([[0, 1, 1, 2, 3], [0, 1, 2, 2, 3]], dtype=torch.long),
    # torch.tensor([[1], [2], [1], [1], [1]], dtype=torch.int32),
    shape=(4, 4),
)
c = SparseOpsMixin(
    torch.tensor([[0, 1, 1, 2, 2], [0, 1, 2, 2, 3]], dtype=torch.long),
    # torch.tensor([[-2], [-3], [1], [1], [1]], dtype=torch.int32),
    shape=(4, 4),
)

print(a.to_dense())
print(b.to_dense())
print(c.to_dense())

result = a * b * c
print(result)
print(result.to_dense())
