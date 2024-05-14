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


def _union_mask(indices: torch.LongTensor, _: int) -> torch.BoolTensor:
    equal = (indices[:, 1:] != indices[:, :-1]).any(dim=0)
    return F.pad(equal, (1, 0), value=True)


class SparseOpsMixin(SparseScatterMixin):
    def apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], *kargs, **kwargs
    ) -> Self:
        return self.create_shared(fn(self._values, *kargs, **kwargs))

    def __and__(self, other: Self):
        if isinstance(other, BaseSparse) and self.dtype == other.dtype == torch.bool:

            if self._values is None:
                return self._generic_ops([self, other], _intersection_mask)

            return self._generic_ops(
                [self, other], _intersection_mask, torch.logical_and
            )

        raise ValueError()

    def __or__(self, other: Self):
        if isinstance(other, BaseSparse) and self.dtype == other.dtype == torch.bool:

            if self._values is None:
                return self._generic_ops([self, other], _union_mask)

            return self._generic_ops([self, other], _union_mask, torch.logical_or)

        raise ValueError()

    def __mul__(self, other: Self | int | float | torch.Tensor):
        if isinstance(other, (int, float)):
            return self.create_shared(self._values * other)

        if isinstance(other, torch.Tensor):
            return self.create_shared(self._values * other[None])

        if isinstance(other, BaseSparse) and self.dtype == other.dtype:
            return self._generic_ops([self, other], _intersection_mask, torch.mul)

        raise ValueError()

    def __rmul__(self, other: Self | int | float | torch.Tensor):
        if isinstance(other, (int, float)):
            return self.create_shared(self._values * other)

        if isinstance(other, torch.Tensor):
            return self.create_shared(self._values * other[None])

        raise ValueError()

    def __floordiv__(self, other: int | float | torch.Tensor):
        if isinstance(other, (int, float)):
            return self.create_shared(self._values // other)

        if isinstance(other, torch.Tensor):
            return self.create_shared(self._values // other[None])

        raise ValueError()

    def __truediv__(self, other: int | float | torch.Tensor):
        if isinstance(other, (int, float)):
            return self.create_shared(self._values / other)

        if isinstance(other, torch.Tensor):
            return self.create_shared(self._values / other[None])

        raise ValueError()

    def __mod__(self, other: int | float | torch.Tensor):
        if isinstance(other, (int, float)):
            return self.create_shared(self._values % other)

        if isinstance(other, torch.Tensor):
            return self.create_shared(self._values % other[None])

        raise ValueError()

    def __add__(self, other: Self):
        if isinstance(other, BaseSparse) and self.dtype == other.dtype:
            return self._generic_ops([self, other], _union_mask, torch.add)

        raise ValueError()

    def __sub__(self, other: Self):
        if isinstance(other, BaseSparse) and self.dtype == other.dtype:
            return self._generic_ops([self, other], _union_mask, torch.sub)

        raise ValueError()

    def __neg__(self):
        if self.dtype != torch.bool:
            return self.__class__(self._indices, values=-self._values, shape=self.shape)

        raise ValueError()

    @classmethod
    def cross(cls, x: Self, y: Self) -> Self:
        assert x._values.ndim == y._values.ndim == 2

        return cls._generic_ops(
            [x, y], _intersection_mask, lambda x, y: torch.cross(x, y, 1)
        )

    @classmethod
    def dot(cls, x: Self, y: Self) -> Self:
        assert x._values.ndim == y._values.ndim == 2

        return cls._generic_ops(
            [x, y], _intersection_mask, lambda x, y: torch.sum(x * y, dim=1).to(x.dtype)
        )

    @classmethod
    def _generic_ops(
        cls,
        tensors: List[Self],
        indices_mask: Callable[[torch.LongTensor, int], torch.BoolTensor],
        ops: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        assert len(tensors) > 1

        if cls._is_shared_indices(tensors):
            return cls._generic_shared_idx_ops(tensors, ops)

        tensors = cls._cast_sparse_tensors(tensors)

        shape = cls._get_shape(tensors)
        dims = tuple(range(len(shape)))

        # concatenating indices and values
        cat_indices = cls._cat_indices(tensors)

        # sorting
        perm = cls._argsort_indices(cat_indices, dims=dims)
        sorted_cat_indices = cat_indices[:, perm]

        # keep indices when at least len(tensors) values are present
        # (calculate intersection over the indices)
        mask = indices_mask(sorted_cat_indices, len(tensors))

        # filter indices based on intersection
        indices = sorted_cat_indices[:, mask]

        # filter values and performe the operation
        if any(map(lambda x: x.values is not None, tensors)):
            values_idx, values_mask = cls._get_values_idx_mask(
                tensors, cat_indices, indices, mask, perm
            )

            list_values = []
            for i, tensor in enumerate(tensors):
                if tensor.values is None:
                    list_values.append(None)
                else:
                    values = tensor.values[values_idx[i]]
                    values[values_mask[i]] = 0
                    list_values.append(values)

            indices, values = cls._apply_ops_values(indices, list_values, ops)
        else:
            values = None

        # create a new sparse tensor containing the result
        return cls(indices, values=values, shape=shape)

    @classmethod
    def _get_values_idx_mask(
        cls,
        tensors: Iterable[Self],
        cat_indices: torch.LongTensor,
        indices: torch.LongTensor,
        mask: torch.BoolTensor,
        perm: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        idx_mask = mask.nonzero().flatten()
        filtered_perm = perm[mask]
        padded_perm = F.pad(perm, (0, len(tensors) - 1), value=-1)

        idx_values = torch.arange(len(tensors))
        n_values = torch.tensor([tensor.indices.shape[1] for tensor in tensors])
        ptr = F.pad(n_values.cumsum(0), (1, 0))
        mask_offset = (ptr[None, :-1] <= filtered_perm[:, None]) & (
            filtered_perm[:, None] < ptr[None, 1:]
        )
        offset = (idx_values[None, :] * mask_offset).sum(dim=1)

        origin_idx = (idx_mask - offset)[None, :] + idx_values[:, None]

        values_idx = (padded_perm[origin_idx] - ptr[:-1, None]) % n_values[:, None]
        values_mask = (cat_indices[:, padded_perm[origin_idx]] != indices[:, None]).any(
            dim=0
        )
        return values_idx, values_mask

    @classmethod
    def _is_shared_indices(cls, tensors: List[Self]) -> bool:
        return all(map(lambda x: id(x.indices) == id(tensors[0].indices), tensors[1:]))

    @classmethod
    def _generic_shared_idx_ops(
        cls,
        tensors: List[Self],
        ops: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> Self:
        shape = tensors[0].shape
        indices = tensors[0].indices
        values = [tensor.values for tensor in tensors]
        indices, values = cls._apply_ops_values(indices, values, ops)

        return cls(indices, values=values, shape=shape, sort=False)

    @classmethod
    def _apply_ops_values(
        cls,
        indices: torch.LongTensor,
        values: Iterable[torch.Tensor | None],
        ops: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        if ops is None:  # concatenation if no ops
            values = torch.stack(values, dim=1)
        else:
            values = ops(*values)

        # filter nul values
        mask = (values != 0).view(values.shape[0], -1).any(dim=1)
        if (~mask).any():
            return indices[:, mask], values[mask]
        return indices, values

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
        padded_values = torch.cat(
            (
                values,
                torch.zeros(
                    (n_tensors - 1, *values.shape[1:]),
                    dtype=values.dtype,
                    device=values.device,
                ),
            ),
            dim=0,
        )
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
    def _cat_indices(cls, tensors: List[Self]) -> torch.LongTensor:
        return torch.cat([t.indices for t in tensors], dim=1)

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
            filtered_args = [
                (idx, dim, shape)
                for idx, dim, shape in broadcast
                if tensor.shape[dim] == 1
            ]

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

        repeated_dim = cart_prod.repeat_interleave(self._indices.shape[1], dim=1)
        repeated_indices = self._indices.repeat(1, cart_prod.shape[1])
        repeated_indices[list(dims)] = repeated_dim

        new_shape = list(self.shape)
        for dim, size in zip(dims, sizes):
            new_shape[dim] = size

        if self._values is None:
            repeated_values = None
        else:
            repeat_args = [1] * (len(self._values.shape) - 1)
            repeated_values = self._values.repeat(cart_prod.shape[1], *repeat_args)

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
