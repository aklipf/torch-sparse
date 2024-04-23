from typing import List, Callable, Tuple

import torch
import torch.nn.functional as F

from .typing import Self
from .base import BaseSparse


def _intersection_mask(indices: torch.LongTensor, n_tensors: int) -> torch.BoolTensor:
    equal = (indices[:, 1:] != indices[:, :-1]).any(dim=0)
    equal_batch = F.pad(equal.cumsum(0), (1, 0), value=0)
    mask = equal_batch[: -(n_tensors - 1)] == equal_batch[n_tensors - 1 :]
    return F.pad(mask, (0, n_tensors - 1), value=False)


def _union_mask(indices: torch.LongTensor, n_tensors: int) -> torch.BoolTensor:
    equal = (indices[:, 1:] != indices[:, :-1]).any(dim=0)
    return F.pad(equal, (1, 0), value=True)


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
        cat_indices, cat_values = cls._cat_values(tensors)

        # sorting
        perm = cls._argsort_indices(cat_indices, dims=dims)
        cat_indices = cat_indices[:, perm]

        # keep indices when at least len(tensors) values are present (calculate intersection over the indices)
        mask = indices_mask(cat_indices, len(tensors))

        # filter indices based on intersection
        indices = cat_indices[:, mask]
        # indices = cat_indices[:, : -(len(tensors) - 1)][:, mask]

        # filter values and performe the operation
        if cat_values is not None:
            cat_values = cat_values[perm]

            # select the values
            indices_idx = mask.nonzero().flatten()
            diff_idx = torch.arange(len(tensors), dtype=torch.long, device=device)
            idx = indices_idx[:, None] + diff_idx[None, :]

            # print("idx", idx)
            # print(cat_values.shape)
            # print(F.pad(cat_values, (0, 0, 0, len(tensors) - 1), value=0))
            # print(F.pad(cat_values, (0, 0, 0, len(tensors) - 1), value=0).shape)
            cat_values = F.pad(cat_values, (0, 0, 0, len(tensors) - 1), value=0)[idx]
            # print("test")
            # print(cat_values)
            # print(cat_indices)
            # print(indices_idx)
            # print(idx)
            mask_idx = (idx < (cat_indices.shape[1] - len(tensors) + 1)).all(dim=1)
            # print(mask_idx)
            # print(cat_indices[:, indices_idx])
            # print(idx)
            pad_mask = (
                cat_indices[:, idx[mask_idx].t()]
                == cat_indices[:, None, indices_idx[mask_idx]]
            ).all(dim=0)
            # print(cat_values.shape)
            # print(pad_mask, pad_mask.shape)
            cat_values[mask_idx][~pad_mask.t()] = 0
            # print(cat_values)

            if ops is None:  # concatenation if no ops
                cat_values = cat_values.reshape(idx.shape[0], -1)
            else:
                cat_values = ops(cat_values).type(cat_values.dtype)

            # filter nul values
            mask = (cat_values != 0).all(dim=1)
            indices, cat_values = indices[:, mask], cat_values[mask]

        # create a new sparse tensor containing the result
        return cls(indices, values=cat_values, shape=shape)

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


if __name__ == "__main__":
    a = SparseOpsMixin(
        torch.tensor([[0, 3, 1, 1, 2, 2, 3], [0, 0, 1, 2, 1, 2, 3]], dtype=torch.long),
        torch.tensor([[1], [5], [1], [1], [1], [1], [1]], dtype=torch.int32),
        shape=(4, 4),
    )
    b = SparseOpsMixin(
        torch.tensor([[0, 1, 1, 2, 3], [0, 1, 2, 2, 3]], dtype=torch.long),
        torch.tensor([[1], [2], [1], [1], [1]], dtype=torch.int32),
        shape=(4, 4),
    )
    c = SparseOpsMixin(
        torch.tensor([[0, 1, 1, 2, 2], [0, 1, 2, 2, 3]], dtype=torch.long),
        torch.tensor([[-2], [-2], [1], [1], [1]], dtype=torch.int32),
        shape=(4, 4),
    )

    print(a.to_dense())
    print(b.to_dense())
    print(c.to_dense())
    result = SparseOpsMixin._generic_ops(
        [a, b, c], _intersection_mask, lambda x: x.prod(dim=1)
    )
    print(result)
    print(result.to_dense())
    print(a.to_dense() * b.to_dense() * c.to_dense())
    result = SparseOpsMixin._generic_ops([a, b, c], _union_mask, lambda x: x.sum(dim=1))
    print(result)
    print(result.to_dense())
    print(a.to_dense() + b.to_dense() + c.to_dense())
    exit(0)
    result = a * b * c
    print(result)
    print(result.to_dense())
