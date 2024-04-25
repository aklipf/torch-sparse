from typing import List, Callable, Tuple, Iterable

import torch
import torch.nn.functional as F

from .typing import Self
from .base import BaseSparse
from .scatter import SparseScatterMixin


def _intersection_mask(indices: torch.LongTensor, n_tensors: int) -> torch.BoolTensor:
    equal = (indices[:, 1:] != indices[:, :-1]).any(dim=0)
    equal_batch = BaseSparse._get_ptr(equal)
    mask = equal_batch[: -(n_tensors - 1)] == equal_batch[n_tensors - 1 :]
    return F.pad(mask, (0, n_tensors - 1), value=False)


def _union_mask(indices: torch.LongTensor, n_tensors: int) -> torch.BoolTensor:
    equal = (indices[:, 1:] != indices[:, :-1]).any(dim=0)
    return F.pad(equal, (1, 0), value=True)


class SparseOpsMixin(SparseScatterMixin):

    def __and__(self, other: Self):
        assert self.dtype == other.dtype == torch.bool

        return self._generic_ops([self, other], _intersection_mask)

    def __mul__(self, other: Self):
        assert self.dtype == other.dtype

        return self._generic_ops(
            [self, other], _intersection_mask, lambda x: x[:, 0] * x[:, 1]
        )

    def __or__(self, other: Self):
        assert self.dtype == other.dtype == torch.bool

        return self._generic_ops([self, other], _union_mask)

    def __add__(self, other: Self):
        assert self.dtype == other.dtype

        return self._generic_ops(
            [self, other], _union_mask, lambda x: x[:, 0] + x[:, 1]
        )

    def __sub__(self, other: Self):
        assert self.dtype == other.dtype

        return self._generic_ops(
            [self, other], _union_mask, lambda x: x[:, 0] - x[:, 1]
        )

    @classmethod
    def _generic_ops(
        cls,
        tensors: List[Self],
        indices_mask: Callable[[torch.LongTensor, int], torch.BoolTensor],
        ops: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        assert len(tensors) > 1

        tensors = cls._cast_sparse_tensors(tensors)

        shape = cls._get_shape(tensors)
        dims = tuple(range(len(shape)))

        # concatenating indices and values
        cat_indices, cat_values = cls._cat_values(tensors)

        # sorting
        perm = cls._argsort_indices(cat_indices, dims=dims)
        cat_indices = cat_indices[:, perm]

        # keep indices when at least len(tensors) values are present
        # (calculate intersection over the indices)
        mask = indices_mask(cat_indices, len(tensors))

        # filter indices based on intersection
        indices = cat_indices[:, mask]

        # filter values and performe the operation
        if cat_values is not None:
            cat_values = cat_values[perm]

            batch = cls._cat_batch(tensors)[perm]

            values = cls._select_values(
                cat_indices, cat_values, batch, mask, len(tensors)
            )

            if ops is None:  # concatenation if no ops
                values = values.reshape(values.shape[0], -1)
            else:
                values = ops(values).type(values.dtype)

            # filter nul values
            mask = (values != 0).all(dim=1)
            indices, values = indices[:, mask], values[mask]
        else:
            values = None

        # create a new sparse tensor containing the result
        return cls(indices, values=values, shape=shape)

    @classmethod
    def _select_values(
        cls,
        indices: torch.LongTensor,
        values: torch.Tensor,
        batch: torch.LongTensor,
        mask: torch.BoolTensor,
        n_tensors: int,
    ) -> torch.Tensor:
        # compute targeted index
        indices_idx = mask.nonzero().flatten()
        diff_idx = torch.arange(n_tensors, dtype=torch.long, device=mask.device)
        idx = indices_idx[:, None] + (diff_idx[None, :] + batch[mask, None]) % n_tensors

        # select in values (pad because of overflow)
        padded_values = F.pad(values, (0, 0, 0, n_tensors - 1), value=0)
        selected_values = padded_values[idx]

        # check if the indices match
        padded_indices = F.pad(indices, (0, n_tensors - 1), value=-1)
        index_mask = (
            padded_indices[:, idx.t()] == padded_indices[:, None, indices_idx]
        ).all(dim=0)

        # set to 0 when it doesn't match
        selected_values[~index_mask.t()] = 0

        return selected_values

    @classmethod
    def _get_shape(cls, tensors: List[Self]) -> tuple:
        assert len(tensors) > 0

        shape = tensors[0].shape
        for tensor in tensors[1:]:
            assert shape == tensor.shape

        return shape

    @classmethod
    def _cat_values(cls, tensors: List[Self]) -> Tuple[torch.LongTensor, torch.Tensor]:
        indices = torch.cat([t.indices for t in tensors], dim=1)

        if tensors[0].values is None:
            values = None
        else:
            values = torch.cat([t.values for t in tensors], dim=0)

        return indices, values

    @classmethod
    def _cat_batch(cls, tensors: List[Self]) -> torch.LongTensor:
        device = tensors[0].device

        batch = torch.arange(len(tensors), dtype=torch.long, device=device)
        size = torch.tensor(
            [t.indices.shape[1] for t in tensors], dtype=torch.long, device=device
        )

        return batch.repeat_interleave(size)

    @classmethod
    def _build_broadcast(
        cls,
        tensors: List[Self],
    ) -> List[Tuple[torch.LongTensor, int, int]]:
        shapes = [tensor.shape for tensor in tensors]

        broadcast = []
        for i, shape_i in enumerate(zip(*shapes)):
            count = len(set(shape_i))
            assert count == 1 or (count == 2 and min(shape_i) == 1), "Shape don't match"

            if count == 1:
                continue

            to_shape = max(shape_i)
            if to_shape == 1:
                continue

            cat_indices = torch.cat(
                [tensor.indices[i] for tensor in tensors if tensor.shape[i] > 1],
                dim=0,
            )
            unique_indices = torch.unique(cat_indices)

            broadcast.append((unique_indices, i, to_shape))

        return broadcast

    @classmethod
    def _cast_sparse_tensors(cls, tensors: List[Self]) -> List[Self]:
        broadcast = cls._build_broadcast(tensors)

        if len(broadcast) == 0:
            return tensors

        brodcasted_tensors = []
        for tensor in tensors:
            filtered_args = list(filter(lambda x: tensor.shape[x[1]] == 1, broadcast))

            if len(filtered_args) == 0:
                brodcasted_tensors.append(tensor)
            else:
                brodcasted_tensors.append(tensor._repeat_indices(*zip(*filtered_args)))

        return brodcasted_tensors

    def _repeat_indices(
        self,
        indices: Iterable[torch.LongTensor],
        dims: Iterable[int],
        sizes: Iterable[int],
    ) -> Self:
        cart_prod = self._sparse_cart_prod(*indices)

        repeated_dim = cart_prod.repeat_interleave(self.indices.shape[1], dim=1)
        repeated_indices = self.indices.repeat(1, cart_prod.shape[1])
        repeated_indices[list(dims)] = repeated_dim

        new_shape = list(self.shape)
        for dim, size in zip(dims, sizes):
            new_shape[dim] = size

        if self.values is None:
            repeated_values = None
        else:
            repeated_values = self.values.repeat(cart_prod.shape[1], 1)

        return self.__class__(
            repeated_indices, values=repeated_values, shape=tuple(new_shape)
        )

    @classmethod
    def _sparse_cart_prod(cls, *tensors: torch.LongTensor) -> torch.LongTensor:
        assert (
            len(tensors) > 0
        ), "No input tensor, can't calculate the cartesian product"

        if len(tensors) == 1:
            return tensors[0].unsqueeze(0)

        size = [tensor.shape[0] for tensor in tensors]

        repeat = [1]
        for t_size in size[:-1]:
            repeat.append(repeat[-1] * t_size)

        interleave = [1]
        for t_size in size[-1:0:-1]:
            interleave.insert(0, interleave[0] * t_size)

        prod = []
        for tensor, inter, rep in zip(tensors, interleave, repeat):
            prod.append(tensor.repeat_interleave(inter, dim=0).repeat(rep))

        return torch.stack(prod, dim=0)
