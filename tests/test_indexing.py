from unittest import mock

import pytest
import torch

from sparse.indexing import SparseIndexingMixin

from .mock_tensor import MockTensor
from .assert_sys import assert_no_out_arr


@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def setup_for_indexing():
    tensor = SparseIndexingMixin(MockTensor((3, 5), dtype=torch.long), shape=(4, 4, 4))
    clone = SparseIndexingMixin(MockTensor((3, 5), dtype=torch.long), shape=(4, 4, 4))

    tensor.clone = mock.MagicMock("clone", return_value=clone)
    clone.unsqueeze_ = mock.MagicMock("unsqueeze_")

    return tensor, clone


@assert_no_out_arr
def test_indexing_get_item():
    tensor, clone = setup_for_indexing()

    result = tensor[None, :, :, None, None, :]

    assert id(result) == id(clone)
    tensor.clone.assert_called_once_with()
    clone.unsqueeze_.assert_has_calls([mock.call(0), mock.call(3), mock.call(4)])
    tensor, clone = setup_for_indexing()

    result = tensor[:]

    assert id(result) == id(clone)
    tensor.clone.assert_called_once_with()
    clone.unsqueeze_.assert_not_called()

    tensor, clone = setup_for_indexing()

    result = tensor[None]

    assert id(result) == id(clone)
    tensor.clone.assert_called_once_with()
    clone.unsqueeze_.assert_called_once_with(0)
