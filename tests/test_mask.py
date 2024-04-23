from unittest import mock

import torch

from sparse.mask import SparseMaskMixin

from .mock_tensor import MockTensor
from .assert_sys import assert_no_out_arr


@assert_no_out_arr
def test_mask_inplace_without_value():
    indices = torch.tensor(
        [[0, 1, 2, 2, 4, 4, 5, 5, 6, 6], [4, 2, 6, 6, 5, 5, 3, 3, 0, 0]]
    )
    sparse = SparseMaskMixin(indices, sort=False)

    mask = torch.tensor(
        [False, True, True, False, False, False, True, True, False, True]
    )
    result = sparse.mask_(mask)

    assert id(result) == id(sparse)
    assert (sparse.indices == torch.tensor([[1, 2, 5, 5, 6], [2, 6, 3, 3, 0]])).all()
    assert sparse.values is None


@assert_no_out_arr
def test_mask_inplace_with_value():
    indices = torch.tensor(
        [[0, 1, 2, 2, 4, 4, 5, 5, 6, 6], [4, 2, 6, 6, 5, 5, 3, 3, 0, 0]]
    )
    values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sparse = SparseMaskMixin(indices, values, sort=False)

    mask = torch.tensor(
        [False, True, True, False, False, False, True, True, False, True]
    )
    result = sparse.mask_(mask)

    assert id(result) == id(sparse)
    assert (sparse.indices == torch.tensor([[1, 2, 5, 5, 6], [2, 6, 3, 3, 0]])).all()
    assert (sparse.values == torch.tensor([[2], [3], [7], [8], [10]])).all()


@assert_no_out_arr
def test_mask_copy():
    sparse = SparseMaskMixin(MockTensor((2, 12), dtype=torch.long))

    mask = MockTensor((12,), dtype=torch.bool)

    cloned = SparseMaskMixin(MockTensor((2, 12), dtype=torch.long))
    sparse.clone = mock.MagicMock("clone", return_value=cloned)
    cloned.mask_ = mock.MagicMock("mask_")

    result = sparse.mask(mask)

    assert id(result) == id(cloned)
    sparse.clone.assert_called_once_with()
    cloned.mask_.assert_called_once_with(mask)
