from unittest import mock

import pytest
import torch

from sparse import SparseTensor, Mapping
from sparse.indexing import SparseIndexingMixin

from tests.utils.mock_tensor import MockTensor
from tests.utils.random_sparse import randint_sparse
from tests.utils.assert_sys import assert_no_out_arr
from tests.utils.assert_equals_tensors import assert_equal_tensors


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


@assert_no_out_arr
def test_indexing_get_item_mapping():
    torch.manual_seed(2)

    indices, values = randint_sparse((4,), ratio=0.8)
    tensor = SparseTensor(indices, values, shape=(4,))
    mapping = Mapping.repeat_last_dims(tensor, 1, 2)

    assert_equal_tensors(
        -(tensor[mapping[0]] - tensor[mapping[1]]).to_dense(),
        (tensor[mapping[1]] - tensor[mapping[0]]).to_dense(),
    )

    indices, values = randint_sparse((16, 16))
    tensor = SparseTensor(indices, shape=(16, 16))
    result = tensor[:, :, None] & tensor[:, None, :]
    mapping = Mapping.repeat_last_dims(tensor, 1, 2)
    indexed = tensor[mapping]

    assert result.shape == indexed.shape
    assert_equal_tensors(result.indices, indexed.indices)
    assert indexed.values is None

    tensor = SparseTensor(indices, values, shape=(16, 16))

    assert_equal_tensors(
        -(tensor[mapping[0]] - tensor[mapping[1]]).to_dense(),
        (tensor[mapping[1]] - tensor[mapping[0]]).to_dense(),
    )

    indices, values = randint_sparse((16, 16, 16))
    tensor = SparseTensor(indices, shape=(16, 16, 16))
    result = tensor[:, :, :, None, None] & tensor[:, None, None, :, :]
    mapping = Mapping.repeat_last_dims(tensor, 2, 2)
    indexed = tensor[mapping]

    assert result.shape == indexed.shape
    assert_equal_tensors(result.indices, indexed.indices)
    assert indexed.values is None

    tensor = SparseTensor(indices, values, shape=(16, 16, 16))

    assert_equal_tensors(
        (tensor[mapping[0]] - tensor[mapping[1]])
        .to_dense()
        .swapdims(1, 3)
        .swapdims(2, 4),
        (tensor[mapping[1]] - tensor[mapping[0]]).to_dense(),
    )

    indices, values = randint_sparse((16, 16, 16), size=(6, 8, 2))

    tensor = SparseTensor(indices, values, shape=(16, 16, 16))
    mapping = Mapping.repeat_last_dims(tensor, 2, 2)

    assert_equal_tensors(
        (tensor[mapping[0]] - tensor[mapping[1]])
        .to_dense()
        .swapdims(1, 3)
        .swapdims(2, 4),
        (tensor[mapping[1]] - tensor[mapping[0]]).to_dense(),
    )

    indices, _ = randint_sparse((4, 4, 4))
    tensor = SparseTensor(indices, shape=(4, 4, 4))
    result = tensor[:, :, :, None] & tensor[:, :, None, :]
    result = result[:, :, :, :, None] & tensor[:, :, None, None, :]
    mapping = Mapping.repeat_last_dims(tensor, 1, 3)
    indexed = tensor[mapping]

    assert result.shape == indexed.shape
    assert_equal_tensors(result.indices, indexed.indices)
    assert indexed.values is None

    indices, values = randint_sparse((4, 4, 4))
    tensor = SparseTensor(indices, torch.ones_like(values), shape=(4, 4, 4))
    result = (tensor[:, :, :, None, None] * tensor[:, None, None, :, :])[
        :, :, :, :, :, None, None
    ] * tensor[:, None, None, None, None, :, :]
    mapping = Mapping.repeat_last_dims(tensor, 2, 3)
    indexed = tensor[mapping]

    assert result.shape == indexed.shape
    assert_equal_tensors(result.indices, indexed.indices)
    assert (result.values == indexed.values).all()

    indices, values = randint_sparse((4, 4, 4), size=(2, 3, 5))
    tensor = SparseTensor(indices, torch.ones_like(values), shape=(4, 4, 4))
    result = (tensor[:, :, :, None, None] * tensor[:, None, None, :, :])[
        :, :, :, :, :, None, None
    ] * tensor[:, None, None, None, None, :, :]
    mapping = Mapping.repeat_last_dims(tensor, 2, 3)
    indexed = tensor[mapping]

    assert result.shape == indexed.shape
    assert_equal_tensors(result.indices, indexed.indices)
    assert (result.values == indexed.values).all()
