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
        dim = cls._dim_to_list(dim)

        cls._assert_cat(sparse_tensors, dim)

        device, out_shape, ptr = cls._get_device_shape_ptr(sparse_tensors, dim)

        if sparse_tensors[0].values is None:
            sparse_cat, cat_size = cls._cat_index_sparse(
                sparse_tensors, out_shape, device
            )
        else:
            sparse_cat, cat_size = cls._cat_sparse(sparse_tensors, out_shape, device)

        if len(dim) > 0:
            cls._reindex_cat_dim_(sparse_cat, dim, ptr, cat_size, device)

        return sparse_cat

    @classmethod
    def _assert_cat(cls, sparse_tensors: Iterable[Self], dim: list):

        for tensor in sparse_tensors:
            assert isinstance(tensor, cls)

        device = sparse_tensors[0].device
        out_ndim = len(sparse_tensors[0].shape)
        for cat_dim in dim:
            assert cat_dim < out_ndim

        for tensor in sparse_tensors:
            assert isinstance(tensor, cls)
            assert tensor.device == device
            assert len(tensor.shape) == out_ndim

    @classmethod
    def _get_device_shape_ptr(
        cls, sparse_tensors: Iterable[Self], dim: List[int]
    ) -> Tuple[torch.device, torch.LongTensor, torch.LongTensor]:

        device = sparse_tensors[0].device

        shapes = torch.tensor(
            [st.shape for st in sparse_tensors], dtype=torch.long, device=device
        )
        out_shape = shapes.amax(dim=0)

        if len(dim) > 0:
            ptr = F.pad(shapes[:, dim].cumsum(0), (0, 0, 1, 0), value=0)
            out_shape[dim] = ptr[-1]
        else:
            ptr = None

        return device, out_shape, ptr

    @classmethod
    def _cat_index_sparse(
        cls,
        sparse_tensors: Iterable[Self],
        out_shape: torch.LongTensor,
        device: torch.device,
    ) -> Tuple[Self, torch.LongTensor]:
        cat_indices, cat_size = [], []

        for st in sparse_tensors:
            cat_size.append(st.indices.shape[1])
            cat_indices.append(st.indices)

        cat_size = torch.tensor(cat_size, dtype=torch.long, device=device)
        cat_indices = torch.cat(cat_indices, dim=1)

        return (
            cls(indices=cat_indices, values=None, shape=out_shape),
            cat_size,
        )

    @classmethod
    def _cat_sparse(
        cls,
        sparse_tensors: Iterable[Self],
        out_shape: torch.LongTensor,
        device: torch.device,
    ) -> Tuple[Self, torch.LongTensor]:
        cat_indices, cat_values, cat_size = [], [], []

        for st in sparse_tensors:
            cat_size.append(st.values.shape[0])
            cat_indices.append(st.indices)
            cat_values.append(st.values)

        cat_size = torch.tensor(cat_size, dtype=torch.long, device=device)
        cat_indices = torch.cat(cat_indices, dim=1)
        cat_values = torch.cat(cat_values, dim=0)

        return (
            cls.__class__(indices=cat_indices, values=cat_values, shape=out_shape),
            cat_size,
        )

    @classmethod
    def _reindex_cat_dim_(
        cls,
        sparse_cat: Self,
        dim: List[int],
        ptr: torch.LongTensor,
        cat_size: torch.LongTensor,
        device: torch.device,
    ) -> Self:

        idx = torch.arange(cat_size.shape[0], dtype=torch.long, device=device)
        batch = idx.repeat_interleave(cat_size)
        sparse_cat.indices[dim] += ptr[batch].t()

        return sparse_cat
