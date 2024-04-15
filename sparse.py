from __future__ import annotations
from typing import Iterable, List, Tuple, Literal

import torch
import torch.nn.functional as F
from torch_scatter import scatter


class Sparse:
    def __init__(
        self, indices: torch.LongTensor, values: torch.Tensor, shape: tuple = None
    ):
        assert indices.ndim == 2 and indices.dtype == torch.long
        assert values.ndim in (1, 2)
        assert indices.shape[1] == values.shape[0]
        assert shape is None or indices.shape[0] == len(shape)

        if shape is None:
            shape = tuple((indices.amax(dim=1) + 1).tolist())

        if values.ndim == 1:
            values.unsqueeze_(1)

        self.shape = tuple(shape)
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

    def unsqueeze(self, dim: int) -> Sparse:
        sparse = self.clone()
        sparse.unsqueeze_(dim)

        return sparse

    def clone(self) -> Sparse:
        return Sparse(
            indices=self.indices.clone(), values=self.values.clone(), shape=self.shape
        )

    def detach(self) -> Sparse:
        return Sparse(
            indices=self.indices.detach(), values=self.values.detach(), shape=self.shape
        )

    @classmethod
    def cat(cls, sparse_tensors: Iterable[Sparse], dim: int | tuple = None) -> Sparse:
        """
        Concatenate sparse tensors
        """
        dim = cls.__dim_to_list(dim)

        cls.__assert_cat(sparse_tensors, dim)

        device, out_shape, ptr = cls.__get_device_shape_ptr(sparse_tensors, dim)

        sparse_cat, cat_size = cls.__cat_sparse(sparse_tensors, out_shape, device)

        if len(dim) > 0:
            cls.__reindex_cat_dim_(sparse_cat, dim, ptr, cat_size, device)

        return sparse_cat

    def sum(self, dim: int | tuple = None) -> Sparse:
        return self.scatter(dim, "sum")

    def mean(self, dim: int | tuple = None) -> Sparse:
        return self.scatter(dim, "mean")

    def __repr__(self) -> str:
        return f"""Sparse(shape={self.shape},
  indices={self.indices},
  values={self.values},
  device=\"{self.device}\")"""

    @property
    def dense(self) -> torch.Tensor:
        x = torch.zeros(
            self.shape + self.values.shape[1:],
            dtype=self.values.dtype,
            device=self.device,
        )
        indices = [self.indices[i] for i in range(self.indices.shape[0])]
        x[indices] = self.values

        return x

    def scatter(
        self, dim: int | tuple = None, reduce: Literal["sum", "mean"] = "sum"
    ) -> Sparse:
        dim = self.__dim_to_list(dim)
        dim = sorted(dim, reverse=True)

        if len(dim) == len(self.shape):
            indices = torch.tensor([[0]], dtype=torch.long, device=self.device)

            if reduce == "sum":
                values = self.values.sum()
            elif reduce == "mean":
                values = self.values.mean()

            return Sparse(indices, values.view(1), shape=(1,))

        indices, batch = self.__unique_index(dim)

        values = scatter(self.values, batch, dim=0, reduce=reduce)

        shape = list(
            map(lambda x: self.shape[x], set(range(len(self.shape))) - set(dim))
        )

        return Sparse(indices, values, shape)

    @classmethod
    def __assert_cat(cls, sparse_tensors: Iterable[Sparse], dim: list):

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

    @classmethod
    def __dim_to_list(cls, dim: int | tuple = None) -> List[int]:
        if dim is None:
            return []

        elif isinstance(dim, int):
            return [dim]

        lst_dim = list(dim)

        assert len(lst_dim) == len(set(dim)), "multiple dimensions are the same"

        return lst_dim

    @classmethod
    def __get_device_shape_ptr(
        cls, sparse_tensors: Iterable[Sparse], dim: List[int]
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
    def __cat_sparse(
        cls,
        sparse_tensors: Iterable[Sparse],
        out_shape: torch.LongTensor,
        device: torch.device,
    ) -> Tuple[Sparse, torch.LongTensor]:
        cat_indices, cat_values, cat_size = [], [], []

        for st in sparse_tensors:
            cat_size.append(st.values.shape[0])
            cat_indices.append(st.indices)
            cat_values.append(st.values)

        cat_size = torch.tensor(cat_size, dtype=torch.long, device=device)
        cat_indices = torch.cat(cat_indices, dim=1)
        cat_values = torch.cat(cat_values, dim=0)

        return Sparse(indices=cat_indices, values=cat_values, shape=out_shape), cat_size

    @classmethod
    def __reindex_cat_dim_(
        cls,
        sparse_cat: Sparse,
        dim: List[int],
        ptr: torch.LongTensor,
        cat_size: torch.LongTensor,
        device: torch.device,
    ) -> Sparse:

        idx = torch.arange(cat_size.shape[0], dtype=torch.long, device=device)
        batch = idx.repeat_interleave(cat_size)
        sparse_cat.indices[dim] += ptr[batch].t()

        return sparse_cat

    def __unique_index(
        self, without_dim: List[int]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        keep_dim = sorted(list(set(range(len(self.shape))) - set(without_dim)))

        size = self.indices.amax(dim=1) + 1
        size[without_dim] = 1
        prod = F.pad(size, (1, 0), value=1).cumprod(0)
        prod[without_dim] = 0

        unique_index = (self.indices * prod[:-1, None]).sum(dim=0)

        unique, batch = torch.unique(unique_index, return_inverse=True)

        indices = torch.zeros(
            (len(keep_dim), unique.shape[0]), dtype=torch.long, device=self.device
        )
        indices[:, batch] = self.indices[keep_dim]

        return indices, batch


A = Sparse(
    torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
    torch.tensor([1.0, 1.0, 2.0, 2.0]),
).to("cuda")
B = Sparse(torch.tensor([[0, 1, 2], [0, 1, 2]]), torch.tensor([3.0, 3.0, 4.0])).to(
    "cuda"
)

# print(A)
# print(B)
# res = Sparse.cat((A.unsqueeze(0), B.unsqueeze(0)), dim=(0, 1))
# print(res)

x = Sparse(
    torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5],
        ]
    ),
    torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]),
).to("cuda")
print(x.dense.squeeze())
print(x.sum(0).dense.squeeze())
print(x.sum(1).dense.squeeze())
print(x.sum((0, 1)).dense.squeeze())
