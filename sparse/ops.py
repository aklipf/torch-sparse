from typing import List, Callable, Tuple

import torch
import torch.nn.functional as F

from .typing import Self
from .base import BaseSparse


class SparseOpsMixin(BaseSparse):
    @classmethod
    def _intersection_op(
        cls,
        tensors: List[Self],
        ops: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        assert len(tensors) > 1
        shape = cls._get_shape(tensors)
        device = tensors[0].device
        dims = tuple(range(len(shape)))

        indices, values = cls._cat_values(tensors)
        # print(indices)

        perm = cls._argsort_indices(indices, dims=dims)
        indices = indices[:, perm]

        equal = (indices[dims, 1:] != indices[dims, :-1]).any(dim=0)
        equal_batch = F.pad(equal.cumsum(0), (1, 0), value=0)

        # print(values)
        # print(indices)
        mask = equal_batch[: -(len(tensors) - 1)] == equal_batch[len(tensors) - 1 :]
        # print(indices[:, : -(len(tensors) - 1)][:, mask])
        indices = indices[:, : -(len(tensors) - 1)][:, mask]

        if values is not None:
            values = values[perm]

            indices_idx = torch.arange(
                values.shape[0] - len(tensors) + 1, dtype=torch.long, device=device
            )[mask]
            diff_idx = torch.arange(len(tensors), dtype=torch.long, device=device)
            idx = indices_idx[:, None] + diff_idx[None, :]

            if ops is None:
                values = values[idx].reshape(idx.shape[0], -1)
            else:
                values = ops(values[idx])

        return cls(indices, values=values, shape=shape)

    @classmethod
    def _union_op(
        cls,
        tensors: List[Self],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        shape = cls._get_shape(tensors)

    @classmethod
    def _set_minus_op(
        cls,
        tensors: List[Self],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        shape = cls._get_shape(tensors)

    @classmethod
    def _get_shape(cls, tensors: List[Self]) -> tuple:
        assert len(tensors) > 0

        shape = tensors[0].shape
        for t in tensors[1:]:
            assert shape == t.shape

        return shape

    @classmethod
    def _cat_values(cls, tensors: List[Self]) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        device = tensors[0].device

        size = torch.tensor(
            [t.indices.shape[1] for t in tensors], dtype=torch.long, device=device
        )
        batch = torch.arange(
            size.shape[0], dtype=torch.long, device=device
        ).repeat_interleave(size)
        """

        indices = torch.cat([t.indices for t in tensors], dim=1)
        if tensors[0].values is None:
            values = None
        else:
            values = torch.cat([t.values for t in tensors], dim=0)

        return indices, values


a = BaseSparse(
    torch.tensor([[0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 3]], dtype=torch.long),
    torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.int32),
)
b = BaseSparse(
    torch.tensor([[0, 1, 1, 2, 3], [0, 1, 2, 2, 3]], dtype=torch.long),
    torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.int32),
)
c = BaseSparse(
    torch.tensor([[0, 1, 1, 2, 2, 3], [0, 1, 2, 2, 3, 3]], dtype=torch.long),
    torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.int32),
)

print(a.to_dense())
print(b.to_dense())
print(c.to_dense())

result = SparseOpsMixin._intersection_op([a, b, c])
print(result.to_dense())
