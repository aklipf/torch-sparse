import pytest
import torch
import unittest.mock as mock

from sparse.typing import Self
from sparse.scatter import SparseScatterMixin

from .mock_tensor import MockTensor


def assert_scatter_sum(
    indices: torch.LongTensor, values: torch.Tensor, dims: int | tuple
):
    sparse = SparseScatterMixin(indices.clone(), shape=(32, 32, 32, 32))
    assert (sparse.indices == indices).all()
    assert (
        sparse.scatter(dims, reduce="sum").to_dense() == sparse.to_dense().sum(dims)
    ).all()
    assert (sparse.indices == indices).all()

    sparse = SparseScatterMixin(indices.clone(), values.clone(), shape=(32, 32, 32, 32))
    assert (sparse.indices == indices).all()
    assert (
        sparse.scatter(dims, reduce="sum").to_dense() == sparse.to_dense().sum(dims)
    ).all()
    assert (sparse.indices == indices).all()


def assert_scatter_mean(
    indices: torch.LongTensor, values: torch.Tensor, dims: int | tuple
):
    sparse = SparseScatterMixin(indices.clone(), shape=(32, 32, 32, 32))
    assert (sparse.indices == indices).all()
    assert (
        sparse.scatter(dims, reduce="mean").to_dense() == sparse.to_dense().mean(dims)
    ).all()
    assert (sparse.indices == indices).all()

    sparse = SparseScatterMixin(indices.clone(), values.clone(), shape=(32, 32, 32, 32))
    assert (sparse.indices == indices).all()
    assert (
        sparse.scatter(dims, reduce="mean").to_dense() == sparse.to_dense().mean(dims)
    ).all()
    assert (sparse.indices == indices).all()


def test_scatter_scatter():
    torch.manual_seed(0)
    random = torch.rand((32, 32, 32, 32))
    mask = random < 0.1
    indices = mask.int().nonzero().t()
    values = torch.randint(0, 16, (indices.shape[1],)).float()

    assert_scatter_sum(indices, values, (0,))
    assert_scatter_sum(indices, values, (0, 1))
    assert_scatter_sum(indices, values, (0, 1, 2))
    assert_scatter_sum(indices, values, (0, 1, 2, 3))
    assert_scatter_sum(indices, values, (0, 2, 3))
    assert_scatter_sum(indices, values, (3,))
    assert_scatter_sum(indices, values, (2,))
    assert_scatter_sum(indices, values, (2, 3))
    assert_scatter_sum(indices, values, (1, 2, 3))
    assert_scatter_sum(indices, values, (0, 1, 3))
    assert_scatter_sum(indices, values, None)

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


def test_scatter_sum():
    sparse = SparseScatterMixin(MockTensor((4, 16), dtype=torch.long))
    result = SparseScatterMixin(MockTensor((1, 3), dtype=torch.long))
    dim = (1, 2, 3)

    sparse.scatter = mock.MagicMock("scatter", return_value=result)

    result_sum = sparse.sum(dim)

    assert id(result) == id(result_sum)
    sparse.scatter.assert_called_once_with(dim, "sum")


def test_scatter_mean():
    sparse = SparseScatterMixin(MockTensor((4, 16), dtype=torch.long))
    result = SparseScatterMixin(MockTensor((1, 3), dtype=torch.long))
    dim = (1, 2, 3)

    sparse.scatter = mock.MagicMock("scatter", return_value=result)

    result_mean = sparse.mean(dim)

    assert id(result) == id(result_mean)
    sparse.scatter.assert_called_once_with(dim, "mean")
