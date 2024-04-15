from __future__ import annotations
from typing import Iterable, List, Tuple, Literal, Any

import torch
import torch.nn.functional as F
from torch_scatter import scatter


def prod(values: Iterable[Any]) -> Any:
    result = 1
    for value in values:
        result *= value
    return result


class Sparse:
    MAX_SIZE = 1 << 63

    def __init__(
        self,
        indices: torch.LongTensor,
        values: torch.Tensor = None,
        shape: tuple = None,
    ):
        assert indices.ndim == 2 and indices.dtype == torch.long
        assert values is None or values.ndim in (1, 2)
        assert values is None or indices.shape[1] == values.shape[0]
        assert shape is None or indices.shape[0] == len(shape)

        if shape is None:
            shape = tuple((indices.amax(dim=1) + 1).tolist())

        assert prod(shape) <= self.MAX_SIZE

        if values is not None and values.ndim == 1:
            values.unsqueeze_(1)

        self.shape = tuple(shape)
        self.indices = indices
        self.values = values

    @property
    def dtype(self) -> type:
        if self.values is None:
            return torch.float32

        return self.values.dtype

    @property
    def device(self) -> torch.device:
        return self.indices.device

    def to(self, device: torch.device) -> Sparse:
        if self.values is None:
            return Sparse(
                indices=self.indices.to(device),
                values=None,
                shape=self.shape,
            )

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

    def squeeze_(self, dim: int = None) -> Sparse:
        assert dim is None or isinstance(dim, int)
        assert dim is None or dim <= self.indices.shape[0] and self.shape[dim] == 1

        if dim is None:
            keep_dim = [d for d, n in enumerate(self.shape) if n != 1]
        else:
            keep_dim = [d for d, _ in enumerate(self.shape) if d != dim]

        self.indices = self.indices[keep_dim]
        self.shape = tuple([self.shape[d] for d in keep_dim])

        return self

    def unsqueeze(self, dim: int) -> Sparse:
        sparse = self.clone()
        sparse.unsqueeze_(dim)

        return sparse

    def squeeze(self, dim: int) -> Sparse:
        sparse = self.clone()
        sparse.squeeze_(dim)

        return sparse

    def clone(self) -> Sparse:
        if self.values is None:
            return Sparse(indices=self.indices.clone(), values=None, shape=self.shape)

        return Sparse(
            indices=self.indices.clone(), values=self.values.clone(), shape=self.shape
        )

    def detach(self) -> Sparse:
        if self.values is None:
            return Sparse(indices=self.indices.detach(), values=None, shape=self.shape)

        return Sparse(
            indices=self.indices.detach(), values=self.values.detach(), shape=self.shape
        )

    def mask_(self, mask: torch.BoolTensor) -> Sparse:
        assert mask.ndim == 1 and mask.shape[0] == self.indices.shape[1]

        self.indices = self.indices[:, mask]

        if self.values is not None:
            self.values = self.values[mask]

        return self

    def mask(self, mask: torch.BoolTensor) -> Sparse:
        sparse = self.clone()
        sparse.mask_(mask)

        return sparse

    @classmethod
    def cat(cls, sparse_tensors: Iterable[Sparse], dim: int | tuple = None) -> Sparse:
        """
        Concatenate sparse tensors
        """
        dim = cls.__dim_to_list(dim)

        cls.__assert_cat(sparse_tensors, dim)

        device, out_shape, ptr = cls.__get_device_shape_ptr(sparse_tensors, dim)

        if sparse_tensors[0].values is None:
            sparse_cat, cat_size = cls.__cat_index_sparse(
                sparse_tensors, out_shape, device
            )
        else:
            sparse_cat, cat_size = cls.__cat_sparse(sparse_tensors, out_shape, device)

        if len(dim) > 0:
            cls.__reindex_cat_dim_(sparse_cat, dim, ptr, cat_size, device)

        return sparse_cat

    @classmethod
    def prod(cls, sparse_tensors: Iterable[Sparse], dim: int) -> Sparse:
        """
        Cartesian product of sparse tensors over a dimension
        """
        pass

    def reshape_(self, shape: Iterable[int]) -> Sparse:
        shape = list(shape)

        indices, shape = self.__indices_to_shape(shape)

        self.indices = indices
        self.shape = shape

        return self

    def reshape(self, shape: Iterable[int]) -> Sparse:
        shape = list(shape)

        indices, shape = self.__indices_to_shape(shape)

        return Sparse(indices, self.values, shape)

    def sum(self, dim: int | tuple = None) -> Sparse:
        return self.scatter(dim, "sum")

    def mean(self, dim: int | tuple = None) -> Sparse:
        return self.scatter(dim, "mean")

    def __repr__(self) -> str:
        if self.values is None:
            return f"""Sparse(shape={self.shape},
  indices={self.indices},
  device=\"{self.device}\")"""

        return f"""Sparse(shape={self.shape},
  indices={self.indices},
  values={self.values},
  device=\"{self.device}\")"""

    @property
    def dense(self) -> torch.Tensor:
        if self.values is None:
            shape = self.shape
        else:
            shape = self.shape + self.values.shape[1:]

        x = torch.zeros(shape, dtype=self.dtype, device=self.device)
        indices = [self.indices[i] for i in range(self.indices.shape[0])]

        if self.values is None:
            x[indices] = 1
        else:
            x[indices] = self.values

        return x

    def numel(self) -> int:
        total_size = 1
        for s in self.shape:
            total_size *= s

        return total_size

    def __scatter_all(self, reduce: Literal["sum", "mean"] = "sum") -> Sparse:
        indices = torch.tensor([[0]], dtype=torch.long, device=self.device)

        if reduce == "sum":
            if self.values is None:
                value = self.indices.shape[1]
            else:
                value = self.values.sum().item()

        elif reduce == "mean":
            if self.values is None:
                value = self.indices.shape[1] / self.numel()
            else:
                value = self.values.mean().item()

        values = torch.tensor([value], dtype=self.dtype, device=self.device)

        return Sparse(indices, values, shape=(1,))

    def scatter(
        self, dim: int | tuple = None, reduce: Literal["sum", "mean"] = "sum"
    ) -> Sparse:
        dim = self.__dim_to_list(dim)
        dim = sorted(dim, reverse=True)

        if len(dim) == len(self.shape):
            return self.__scatter_all(reduce)

        indices, batch = self.__unique_index(dim)

        if self.values is None:
            values = scatter(
                torch.ones_like(self.indices[0], dtype=self.dtype),
                batch,
                dim=0,
                reduce=reduce,
            )
        else:
            values = scatter(self.values, batch, dim=0, reduce=reduce)

        shape = list(
            map(lambda x: self.shape[x], set(range(len(self.shape))) - set(dim))
        )

        return Sparse(indices, values, shape)

    def type(self, dtype: type) -> Sparse:
        if self.values is None:
            return Sparse(
                self.indices,
                torch.ones_like(self.indices[0], dtype=dtype),
                self.shape,
            )

        return Sparse(
            self.indices,
            self.values.type(dtype),
            self.shape,
        )

    def float(self) -> Sparse:
        return self.type(torch.float32)

    def double(self) -> Sparse:
        return self.type(torch.float64)

    def int(self) -> Sparse:
        return self.type(torch.int32)

    def long(self) -> Sparse:
        return self.type(torch.int64)

    def bool(self) -> Sparse:
        return self.type(torch.bool)

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
    def __cat_index_sparse(
        cls,
        sparse_tensors: Iterable[Sparse],
        out_shape: torch.LongTensor,
        device: torch.device,
    ) -> Tuple[Sparse, torch.LongTensor]:
        cat_indices, cat_size = [], []

        for st in sparse_tensors:
            cat_size.append(st.indices.shape[1])
            cat_indices.append(st.indices)

        cat_size = torch.tensor(cat_size, dtype=torch.long, device=device)
        cat_indices = torch.cat(cat_indices, dim=1)

        return Sparse(indices=cat_indices, values=None, shape=out_shape), cat_size

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

    def __global_index(self, without_dim: List[int] = None) -> torch.LongTensor:
        shape = torch.tensor(self.shape, dtype=torch.long)

        if without_dim is not None:
            shape[without_dim] = 1

        prod_tensor = F.pad(shape, (1, 0), value=1).cumprod(0)

        if without_dim is not None:
            prod_tensor[without_dim] = 0

        return (self.indices * prod_tensor.to(self.device)[:-1, None]).sum(dim=0)

    def __indices_to_shape(
        self, shape: List[int]
    ) -> Tuple[torch.LongTensor, List[int]]:
        numel = self.numel()

        num_anon = sum(map(lambda x: x == -1, shape))
        assert num_anon <= 1

        if num_anon == 1:
            total = prod(filter(lambda x: x != -1, shape))
            anon_dim = numel // total
            shape = list(map(lambda x: anon_dim if x == -1 else x, shape))

        assert prod(shape) == numel

        global_index = self.__global_index()

        shape_tensor = torch.tensor(shape, dtype=torch.long, device=self.device)
        prod_tensor = F.pad(shape_tensor, (1, 0), value=1).cumprod(0)[:-1]

        indices = (global_index[None, :] // prod_tensor[:, None]) % shape_tensor[
            :, None
        ]

        return indices, shape

    def __unique_index(
        self, without_dim: List[int]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        keep_dim = sorted(list(set(range(len(self.shape))) - set(without_dim)))

        unique_index = self.__global_index(without_dim)

        unique, batch = torch.unique(unique_index, return_inverse=True)

        indices = torch.zeros(
            (len(keep_dim), unique.shape[0]), dtype=torch.long, device=self.device
        )
        indices[:, batch] = self.indices[keep_dim]

        return indices, batch


"""
A = Sparse(
    torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
).to("cuda")
B = Sparse(torch.tensor([[0, 1, 2], [0, 1, 2]])).to("cuda")

print(A)
print(B)
res = Sparse.cat((A.unsqueeze(0), B.unsqueeze(0)), dim=(0, 1))
print(res)
"""

x = Sparse(
    torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5],
        ]
    ),
)

print(x)
print(x.dense.squeeze())
print(x.sum(0).dense.squeeze())
print(x.sum(1).dense.squeeze())
print(x.sum((0, 1)).dense.squeeze())

print(x.reshape_((14, 7)))
