from __future__ import annotations
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F


class Sparse:
    def __init__(
        self, indices: torch.LongTensor, values: torch.Tensor, shape: tuple = None
    ):
        assert indices.ndim == 2 and indices.dtype == torch.long
        assert values.ndim == 1
        assert indices.shape[1] == values.shape[0]

        if shape is None:
            shape = tuple((indices.amax(dim=1) + 1).tolist())

        self.shape = shape
        self.indices = indices
        self.values = values

    @property
    def dtype(self) -> type:
        return self.values.dtype

    @property
    def device(self) -> torch.device:
        return self.values.device

    def to(self, device: torch.device) -> Sparse:
        return Sparse(
            indices=self.indices.to(device),
            values=self.values.to(device),
            shape=self.shape,
        )

    def unsqueeze_(self, dim: int) -> Sparse:
        assert isinstance(dim, int)

        insert_zeros = torch.zeros(
            (1, self.indices.shape[1]), dtype=torch.long, device=self.indices.device
        )

        self.indices = torch.cat(
            (self.indices[:dim], insert_zeros, self.indices[dim:]), dim=0
        )
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        self.shape = tuple(new_shape)

        return self

    def __repr__(self) -> str:
        return f"""Sparse(shape={self.shape},
  indices={self.indices},
  values={self.values},
  device=\"{self.device}\")"""

    @classmethod
    def __out_cat(
        cls, sparse_tensors: Iterable[Sparse], dim: tuple
    ) -> Tuple[int, torch.device]:

        for tensor in sparse_tensors:
            assert isinstance(tensor, Sparse)

        device = sparse_tensors[0].device
        out_ndim = len(sparse_tensors[0].shape)
        for cat_dim in dim:
            assert cat_dim < out_ndim

        for tensor in sparse_tensors:
            assert isinstance(tensor, Sparse)
            assert tensor.device == device
            assert len(tensor.shape) == out_ndim

        return out_ndim, device

    @classmethod
    def cat(cls, sparse_tensors: Iterable[Sparse], dim: int | tuple) -> Sparse:
        if isinstance(dim, int):
            dim = [dim]
        else:
            dim = list(dim)

        ndim, device = cls.__out_cat(sparse_tensors, dim)

        shapes = torch.tensor(
            [st.shape for st in sparse_tensors], dtype=torch.long, device=device
        )
        out_shape = shapes.amax(dim=0)
        ptr = F.pad(shapes[:, dim].cumsum(0), (0, 0, 1, 0), value=0)

        out_shape[dim] = ptr[-1]

        print(out_shape)

        cat_indices, cat_values, cat_size = [], [], []
        for st in sparse_tensors:
            cat_size.append(st.values.shape[0])
            cat_indices.append(st.indices)
            cat_values.append(st.values)

        cat_size = torch.tensor(cat_size, dtype=torch.long, device=device)
        cat_indices = torch.cat(cat_indices, dim=1)
        cat_values = torch.cat(cat_values, dim=0)

        idx = torch.arange(cat_size.shape[0], dtype=torch.long, device=device)
        batch = idx.repeat_interleave(cat_size)
        print(cat_size)
        print(cat_indices)
        print(cat_values)
        print(batch)


A = Sparse(
    torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]), torch.tensor([1.0, 1.0, 2.0, 2.0])
)
B = Sparse(torch.tensor([[0, 1, 2], [0, 1, 2]]), torch.tensor([3.0, 3.0, 4.0]))

print(A)
Sparse.cat((A.unsqueeze_(0), B.unsqueeze_(0)), dim=(1, 2))
