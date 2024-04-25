from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F

from .typing import Self
from .base import BaseSparse


class SparseCatMixin(BaseSparse):

    @classmethod
    def cat(cls, sparse_tensors: Iterable[Self], dim: int | tuple = None) -> Self:
        """
        Concatenate sparse tensors
        """
        assert len(sparse_tensors) > 0

        dim = next(iter(sparse_tensors))._dim_to_list(dim)

        cls._assert_cat(sparse_tensors, dim)

        device, out_shape, ptr = cls._get_device_shape_ptr(sparse_tensors, dim)

        sparse_cat, cat_size = cls._cat_sparse(sparse_tensors, out_shape, device)

        if len(dim) > 0:
            sparse_cat._reindex_cat_dim_(dim, ptr, cat_size)

        if not sparse_cat._is_sorted():
            sparse_cat._sort_by_indices_()

        return sparse_cat

    @classmethod
    def _assert_cat(cls, sparse_tensors: Iterable[Self], dim: List[int]):
        for tensor in sparse_tensors:
            assert isinstance(
                tensor, cls
            ), "All inputs must be sparse tensors to be concatenated"

        first_elem = next(iter(sparse_tensors))
        device = first_elem.device
        out_ndim = len(first_elem.shape)
        for tensor in sparse_tensors[1:]:
            assert (
                tensor.device == device
            ), "All sparse tensors must be on the same device to be concatenated"
            assert (
                len(tensor.shape) == out_ndim
            ), "All sparse tensors must have the same number of dimensions (ndim)"

        for cat_dim in dim:
            assert (
                cat_dim < out_ndim
            ), "The concatenation dimension must be less than the number of dimention (ndim)"

    @classmethod
    def _get_device_shape_ptr(
        cls, sparse_tensors: Iterable[Self], dim: List[int]
    ) -> Tuple[torch.device, tuple, torch.LongTensor]:

        device = next(iter(sparse_tensors)).device

        shapes = torch.tensor([st.shape for st in sparse_tensors])
        out_shape = shapes.amax(dim=0)

        if len(dim) > 0:
            # pylint: disable=not-callable
            ptr = F.pad(shapes[:, dim].cumsum(0), (0, 0, 1, 0), value=0)
            out_shape[dim] = ptr[-1]
            ptr = ptr.to(device)
        else:
            ptr = None

        return device, tuple(out_shape.tolist()), ptr

    @classmethod
    def _cat_sparse(
        cls,
        sparse_tensors: Iterable[Self],
        out_shape: tuple,
        device: torch.device,
    ) -> Tuple[Self, torch.LongTensor]:
        has_values = next(iter(sparse_tensors)).values is not None
        cat_indices, cat_size = [], []

        if has_values:
            cat_values = []
        else:
            cat_values = None

        for tensor in sparse_tensors:
            cat_size.append(tensor.indices.shape[1])
            cat_indices.append(tensor.indices)
            if has_values:
                cat_values.append(tensor.values)

        cat_size = torch.tensor(cat_size, dtype=torch.long, device=device)
        cat_indices = torch.cat(cat_indices, dim=1)

        if has_values:
            cat_values = torch.cat(cat_values, dim=0)
        else:
            cat_values = None

        return (
            cls(indices=cat_indices, values=cat_values, shape=out_shape, sort=False),
            cat_size,
        )

    def _reindex_cat_dim_(
        self,
        dim: List[int],
        ptr: torch.LongTensor,
        cat_size: torch.LongTensor,
    ) -> Self:
        self.indices[dim] += (
            ptr[: cat_size.shape[0]].repeat_interleave(cat_size, dim=0).t()
        )

        return self
