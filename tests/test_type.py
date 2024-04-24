from unittest import mock

import pytest
import torch

from sparse.typing import Self
from sparse.type import SparseTypeMixin

from .mock_tensor import MockTensor
from .assert_sys import assert_no_out_arr


@assert_no_out_arr
def test_type():
    sparse = SparseTypeMixin(MockTensor((2, 12), dtype=torch.long))

    with pytest.raises(ValueError):
        sparse.type(torch.long)

    sparse = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.float32)
    )
    result_indices = MockTensor((2, 12), dtype=torch.long)
    result_values = MockTensor((12, 1), dtype=torch.int)

    sparse.indices.clone = mock.MagicMock("clone", return_value=result_indices)
    sparse.values.type = mock.MagicMock("type", return_value=result_values)

    result = sparse.type(torch.int)

    assert isinstance(result, SparseTypeMixin)
    assert id(result) != id(sparse)
    assert result.indices == result_indices and result.values == result_values
    sparse.indices.clone.assert_called_once_with()
    sparse.values.type.assert_called_once_with(torch.int)


@assert_no_out_arr
def test_type_anon():

    sparse = SparseTypeMixin(
        torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.float),
        shape=(3, None, 3),
    )

    result = sparse.type(torch.int)

    assert isinstance(result, SparseTypeMixin)
    assert result.dtype == torch.int
    assert result.shape == (3, 1, 3)
    assert result.real_shape == (3, None, 3)
    assert (
        result.indices
        == torch.tensor([[0, 1, 2], [0, 0, 0], [0, 1, 2]], dtype=torch.long)
    ).all()


@assert_no_out_arr
def test_type_float():
    sparse = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.long)
    )
    result_float = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.float32)
    )
    sparse.type = mock.MagicMock("type", return_value=result_float)

    result = sparse.float()

    sparse.type.assert_called_once_with(torch.float32)
    assert result == result_float


@assert_no_out_arr
def test_type_double():
    sparse = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.long)
    )
    result_double = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.float64)
    )
    sparse.type = mock.MagicMock("type", return_value=result_double)

    result = sparse.double()

    sparse.type.assert_called_once_with(torch.float64)
    assert result == result_double


@assert_no_out_arr
def test_type_int():
    sparse = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.float32)
    )
    result_int = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.int32)
    )
    sparse.type = mock.MagicMock("type", return_value=result_int)

    result = sparse.int()

    sparse.type.assert_called_once_with(torch.int32)
    assert result == result_int


@assert_no_out_arr
def test_type_long():
    sparse = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.float32)
    )
    result_long = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.int64)
    )
    sparse.type = mock.MagicMock("type", return_value=result_long)

    result = sparse.long()

    sparse.type.assert_called_once_with(torch.int64)
    assert result == result_long


@assert_no_out_arr
def test_type_bool():
    sparse = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.float32)
    )
    result_bool = SparseTypeMixin(
        MockTensor((2, 12), dtype=torch.long), MockTensor((12, 1), dtype=torch.bool)
    )
    sparse.type = mock.MagicMock("type", return_value=result_bool)

    result = sparse.bool()

    sparse.type.assert_called_once_with(torch.bool)
    assert result == result_bool
