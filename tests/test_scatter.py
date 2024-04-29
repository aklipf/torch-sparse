from unittest import mock

import pytest
import torch

from sparse.typing import Self
from sparse.scatter import SparseScatterMixin

from .mock_tensor import MockTensor
from .random_sparse import randint_sparse
from .assert_sys import assert_no_out_arr


def assert_scatter_sum(
    indices: torch.LongTensor, values: torch.Tensor, dims: int | tuple, shape: tuple
):
    tensor = SparseScatterMixin(indices.clone(), shape=shape)
    assert (tensor.indices == indices).all()
    assert (
        tensor.scatter(dims, reduce="sum").to_dense() == tensor.to_dense().sum(dims)
    ).all()
    assert (tensor.indices == indices).all()

    tensor = SparseScatterMixin(indices.clone(), values.clone(), shape=shape)
    assert (tensor.indices == indices).all()
    assert (
        tensor.scatter(dims, reduce="sum").to_dense() == tensor.to_dense().sum(dims)
    ).all()
    assert (tensor.indices == indices).all()


def assert_scatter_mean(
    indices: torch.LongTensor, values: torch.Tensor, dims: int | tuple
):
    tensor = SparseScatterMixin(indices.clone(), values.clone(), shape=(32, 32, 32, 32))
    assert (tensor.indices == indices).all()
    assert (
        tensor.scatter(dims, reduce="mean").to_dense() == tensor.to_dense().mean(dims)
    ).all()
    assert (tensor.indices == indices).all()


@assert_no_out_arr
def test_scatter_scatter():
    torch.manual_seed(0)

    indices, values = randint_sparse((4, 4))

    assert_scatter_sum(indices, values, (0,), (4, 4))

    indices, values = randint_sparse((32, 32, 32, 32))

    assert_scatter_sum(indices, values, (0,), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (0, 1), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (0, 1, 2), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (0, 1, 2, 3), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (0, 2, 3), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (3,), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (2,), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (2, 3), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (1, 2, 3), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, (0, 1, 3), (32, 32, 32, 32))
    assert_scatter_sum(indices, values, None, (32, 32, 32, 32))

    sparse = SparseScatterMixin(indices.clone(), shape=(32, 32, 32, 32))
    with pytest.raises(
        AssertionError,
        match="Mean reduction can be computed only on real or complex numbers",
    ):
        sparse.scatter(reduce="mean")

    assert_scatter_mean(indices, values, (0,))
    assert_scatter_mean(indices, values, (0, 1))
    assert_scatter_mean(indices, values, (0, 1, 2))
    assert_scatter_mean(indices, values, (0, 1, 2, 3))
    assert_scatter_mean(indices, values, (0, 2, 3))
    assert_scatter_mean(indices, values, (3,))
    assert_scatter_mean(indices, values, (2,))
    assert_scatter_mean(indices, values, (2, 3))
    assert_scatter_mean(indices, values, (1, 2, 3))
    assert_scatter_mean(indices, values, (0, 1, 3))
    assert_scatter_mean(indices, values, None)


@assert_no_out_arr
def test_scatter_sum():
    sparse = SparseScatterMixin(MockTensor((4, 16), dtype=torch.long))
    result = SparseScatterMixin(MockTensor((1, 3), dtype=torch.long))
    dim = (1, 2, 3)

    sparse.scatter = mock.MagicMock("scatter", return_value=result)

    result_sum = sparse.sum(dim)

    assert id(result) == id(result_sum)
    sparse.scatter.assert_called_once_with(dim, "sum")


@assert_no_out_arr
def test_scatter_mean():
    sparse = SparseScatterMixin(MockTensor((4, 16), dtype=torch.long))
    result = SparseScatterMixin(MockTensor((1, 3), dtype=torch.long))
    dim = (1, 2, 3)

    sparse.scatter = mock.MagicMock("scatter", return_value=result)

    result_mean = sparse.mean(dim)

    assert id(result) == id(result_mean)
    sparse.scatter.assert_called_once_with(dim, "mean")
