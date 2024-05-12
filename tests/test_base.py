from unittest import mock

import pytest
import torch

import sparse
from sparse.base import BaseSparse

from .mock_tensor import MockTensor
from .assert_sys import assert_no_out_arr
from .assert_equals_tensors import assert_equal_tensors


@assert_no_out_arr
def test_base_sort():
    indices = torch.tensor(
        [
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6],
        ]
    )
    perm = torch.randperm(indices.shape[1])
    unique = torch.unique(indices).unsqueeze(0)

    tensor = BaseSparse(indices=indices[:, perm])

    assert_equal_tensors(tensor._indices, unique)

    indices = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6],
            [0, 2, 1, 3, 3, 2, 2, 2, 4, 3, 5, 4, 5, 5, 6],
        ]
    )
    unique = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6],
            [0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 6],
        ]
    )
    perm = torch.randperm(indices.shape[1])

    tensor = BaseSparse(indices=indices[:, perm])

    assert_equal_tensors(tensor._indices, unique)


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_no_sort():
    indices = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6],
            [0, 2, 1, 3, 3, 2, 2, 2, 4, 3, 5, 4, 5, 5, 6],
        ]
    )
    perm = torch.randperm(indices.shape[1])

    tensor = BaseSparse(indices=indices[:, perm])

    assert_equal_tensors(tensor._indices, indices[:, perm])


@assert_no_out_arr
def test_base_sort_with_values():
    indices = torch.unique(torch.randperm(128))[:12].unsqueeze(0)
    values = torch.arange(indices.shape[1], dtype=torch.long)
    perm = torch.randperm(indices.shape[1])

    tensor = BaseSparse(indices=indices[:, perm], values=values[perm])

    assert_equal_tensors(tensor._indices, indices)
    assert_equal_tensors(tensor._values, values)
    assert tensor._values.shape == (12,)

    tensor = BaseSparse(indices=indices[:, perm], values=values[perm, None])

    assert_equal_tensors(tensor._indices, indices)
    assert_equal_tensors(tensor._values, values[:, None])
    assert tensor._values.shape == (12, 1)

    indices = torch.unique(torch.randperm(128))[:12].unsqueeze(0)
    values = torch.arange(indices.shape[1], dtype=torch.long)[:, None].repeat(1, 4)
    perm = torch.randperm(indices.shape[1])

    tensor = BaseSparse(indices=indices[:, perm], values=values[perm])

    assert_equal_tensors(tensor._indices, indices)
    assert_equal_tensors(tensor._values, values)
    assert tensor._values.shape == (12, 4)


@assert_no_out_arr
def test_base_assert():  # TODO check
    # assert indices ndim and dtype
    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([]))

    BaseSparse(torch.tensor([[0]], dtype=torch.long))

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0]], dtype=torch.float32))

    # assert values ndim
    assert_equal_tensors(
        BaseSparse(
            torch.tensor([[0]], dtype=torch.long), values=torch.tensor([0])
        ).values,
        torch.tensor([0]),
    )
    assert_equal_tensors(
        BaseSparse(
            torch.tensor([[0]], dtype=torch.long), values=torch.tensor([[0]])
        ).values,
        torch.tensor([[0]]),
    )

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0]], dtype=torch.long), values=torch.tensor(0))

    assert_equal_tensors(
        BaseSparse(
            torch.tensor([[0]], dtype=torch.long), values=torch.tensor([[[0]]])
        ).values,
        torch.tensor([[[0]]]),
    )

    # assert compare indices shape and values shape
    assert_equal_tensors(
        BaseSparse(
            torch.tensor([[0]], dtype=torch.long), values=torch.tensor([[[0, 0]]])
        ).values,
        torch.tensor([[[0, 0]]]),
    )

    # assert compare indices shape and shape parameter
    BaseSparse(torch.tensor([[0]], dtype=torch.long), shape=(2,))
    BaseSparse(torch.tensor([[0], [0]], dtype=torch.long), shape=(2, 3))

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0]], dtype=torch.long), shape=(2, 3))

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0], [0]], dtype=torch.long), shape=(2,))
    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0], [0]], dtype=torch.long), shape=[2])

    # assert not empty when auto detect shape
    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[]], dtype=torch.long))
    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[], [], []], dtype=torch.long))


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_shape():
    tensor = BaseSparse(MockTensor(shape=(1, 3), dtype=torch.long), shape=(5,))
    assert tensor.shape == (5,)

    tensor = BaseSparse(MockTensor(shape=(2, 3), dtype=torch.long), shape=(12, 4))
    assert tensor.shape == (12, 4)

    indices = MockTensor(shape=(1, 3), dtype=torch.long)
    indices.amax = mock.Mock("amax", return_value=torch.tensor([3], dtype=torch.long))

    tensor = BaseSparse(indices)
    assert tensor.shape == (4,)

    indices.amax.assert_called_once_with(dim=1)

    indices = MockTensor(shape=(2, 3), dtype=torch.long)
    indices.amax = mock.Mock(
        "amax", return_value=torch.tensor([3, 5], dtype=torch.long)
    )

    tensor = BaseSparse(indices)
    assert tensor.shape == (4, 6)

    indices.amax.assert_called_once_with(dim=1)


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_ndim():
    tensor = BaseSparse(MockTensor(shape=(1, 3), dtype=torch.long), shape=(5,))
    assert tensor.ndim == 1
    assert tensor.dim == 1

    tensor = BaseSparse(MockTensor(shape=(3, 3), dtype=torch.long), shape=(5, 2, 9))
    assert tensor.ndim == 3
    assert tensor.dim == 3


@assert_no_out_arr
def test_base_dim_to_list():
    tensor = BaseSparse(
        MockTensor(shape=(4, 3), dtype=torch.long), shape=(16, 16, 16, 16)
    )
    assert tensor._dim_to_list() == [0, 1, 2, 3]
    assert tensor._dim_to_list(dim=None) == [0, 1, 2, 3]
    assert tensor._dim_to_list(dim=3) == [3]

    assert tensor._dim_to_list(dim=(1, 2)) == [1, 2]
    assert tensor._dim_to_list(dim=(2, 1)) == [1, 2]

    with pytest.raises(AssertionError):
        tensor._dim_to_list(dim=(1, 2, 2))

    with pytest.raises(AssertionError):
        tensor._dim_to_list(dim=4)

    with pytest.raises(AssertionError):
        tensor._dim_to_list(dim=(2, 4))

    with pytest.raises(AssertionError):
        tensor._dim_to_list(dim=(2, 4, 1))


@assert_no_out_arr
def test_base_included_dims():
    mocked_tensor = BaseSparse(
        indices=MockTensor(shape=(4, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    assert mocked_tensor._included_dims() == [0, 1, 2, 3]
    assert mocked_tensor._included_dims(except_dim=1) == [0, 2, 3]
    assert mocked_tensor._included_dims(except_dim=(1, 2)) == [0, 3]
    assert mocked_tensor._included_dims(except_dim=(1, 2, 1, 2)) == [0, 3]
    assert mocked_tensor._included_dims(except_dim=(2, 3, 1)) == [0]


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_argsort_indices():
    indices = torch.randint(0, 1024, (3, 1024))
    tensor = BaseSparse(indices)

    perm = BaseSparse._argsort_indices(tensor._indices, [0, 2])
    sorted_indices = (tensor._indices[0, perm] << 10) + tensor._indices[2, perm]
    assert (sorted_indices.diff() >= 0).all()

    perm = BaseSparse._argsort_indices(tensor._indices, [2])
    sorted_indices = tensor._indices[2, perm]
    assert (sorted_indices.diff() >= 0).all()

    perm = BaseSparse._argsort_indices(tensor._indices, [0])
    sorted_indices = tensor._indices[0, perm]
    assert (sorted_indices.diff() >= 0).all()


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_sort_indices():
    indices = torch.randint(0, 1024, (3, 1024))
    tensor = BaseSparse(indices.clone())

    tensor._sort_by_indices_()
    sorted_indices = (
        (tensor._indices[0] << 20) + (tensor._indices[1] << 10) + tensor._indices[2]
    )
    assert (sorted_indices.diff() >= 0).all()

    indices = torch.randperm(1024).unsqueeze(0)
    values = torch.randn(1024)
    tensor = BaseSparse(indices.clone(), values.clone())

    tensor._sort_by_indices_()
    assert (values == tensor._values[indices[0]].flatten()).all()


@assert_no_out_arr
def test_base_is_sorted():
    sorted_indices = torch.randint(0, 1 << 30, (64,)).sort().values
    indices = torch.stack(
        (sorted_indices >> 20, (sorted_indices >> 10) & 0x3FF, sorted_indices & 0x3FF)
    )

    with mock.patch(
        "sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True)
    ):
        tensor = BaseSparse(indices.clone())
    assert tensor._is_sorted()

    indices = torch.randint(0, 1024, (3, 1024))

    with mock.patch(
        "sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True)
    ):
        tensor = BaseSparse(indices.clone())
    assert not tensor._is_sorted()


@assert_no_out_arr
def test_base_prod():
    assert BaseSparse._prod([]) == 1
    assert BaseSparse._prod([1]) == 1
    assert BaseSparse._prod([2]) == 2
    assert BaseSparse._prod([2, 3]) == 6
    assert BaseSparse._prod([2, 3, 4]) == 24
    assert BaseSparse._prod([1, None]) == 1
    assert BaseSparse._prod([None, None]) == 1
    assert BaseSparse._prod([2, None, None, 3, 4]) == 24
    assert BaseSparse._prod([2, None, 3, 4]) == 24


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_dims():
    assert BaseSparse(MockTensor(shape=(1, 3), dtype=torch.long)).dims == (0,)
    assert BaseSparse(MockTensor(shape=(2, 3), dtype=torch.long)).dims == (0, 1)
    assert BaseSparse(MockTensor(shape=(3, 3), dtype=torch.long)).dims == (
        0,
        1,
        2,
    )


@assert_no_out_arr
def test_base_get_ptr():
    ptr = BaseSparse._get_ptr(torch.tensor([2, 3, 2, 0, 1, 2]))
    assert (ptr == torch.tensor([0, 2, 5, 7, 7, 8, 10], dtype=torch.long)).all()


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_set_shape():
    tensor = BaseSparse(MockTensor(shape=(2, 3), dtype=torch.long), shape=(4, 4))

    result = tensor._set_shape_((2, 8))

    assert result.shape == (2, 8)
    assert id(result) == id(tensor)


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_dtype():
    assert BaseSparse(MockTensor(shape=(2, 3), dtype=torch.long)).dtype == torch.bool
    assert (
        BaseSparse(
            MockTensor(shape=(2, 3), dtype=torch.long),
            values=MockTensor(shape=(3, 1), dtype=torch.long),
        ).dtype
        == torch.long
    )
    assert (
        BaseSparse(
            MockTensor(shape=(2, 3), dtype=torch.long),
            values=MockTensor(shape=(3, 1), dtype=torch.float32),
        ).dtype
        == torch.float32
    )


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_device():
    assert BaseSparse(
        MockTensor(shape=(2, 3), dtype=torch.long, device=torch.device("cuda"))
    ).device == torch.device("cuda")

    assert BaseSparse(
        MockTensor(shape=(2, 3), dtype=torch.long, device=torch.device("cuda")),
        values=MockTensor(
            shape=(3, 5), dtype=torch.float32, device=torch.device("cuda")
        ),
    ).device == torch.device("cuda")


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_to_without_values():
    mocked_tensor = BaseSparse(indices=MockTensor(shape=(2, 3), dtype=torch.long))

    to_tensor = MockTensor(shape=(2, 3), dtype=torch.long, device="cuda")
    mocked_tensor._indices.to = mock.MagicMock(name="to", return_value=to_tensor)

    result = mocked_tensor.to("cuda")

    assert isinstance(result, BaseSparse)
    assert id(result._indices) == id(to_tensor)
    assert result.device == "cuda"
    mocked_tensor._indices.to.assert_called_once_with("cuda")


@assert_no_out_arr
def test_base_to_with_values():
    mocked_tensor = BaseSparse(
        indices=MockTensor(shape=(2, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    result_indices = MockTensor(shape=(2, 3), dtype=torch.long)
    result_values = MockTensor(shape=(3, 1))
    mocked_tensor._indices.to = mock.MagicMock(name="to", return_value=result_indices)
    mocked_tensor._values.to = mock.MagicMock(name="to", return_value=result_values)

    result = mocked_tensor.to("cuda")

    assert isinstance(result, BaseSparse)
    assert id(result._indices) == id(result_indices)
    assert id(result._values) == id(result_values)
    mocked_tensor._indices.to.assert_called_once_with("cuda")
    mocked_tensor._values.to.assert_called_once_with("cuda")


@assert_no_out_arr
def test_base_clone_without_values():
    mocked_tensor = BaseSparse(indices=MockTensor(shape=(2, 3), dtype=torch.long))

    cloned_indices = MockTensor(shape=(2, 3), dtype=torch.long)
    mocked_tensor._indices.clone = mock.MagicMock(
        name="clone", return_value=cloned_indices
    )

    result = mocked_tensor.clone()
    assert isinstance(result, BaseSparse)
    assert id(result._indices) == id(cloned_indices)
    mocked_tensor._indices.clone.assert_called_once_with()


@assert_no_out_arr
def test_base_clone_with_values():
    mocked_tensor = BaseSparse(
        indices=MockTensor(shape=(2, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )
    result_indices = MockTensor(shape=(2, 3), dtype=torch.long)
    result_values = MockTensor(shape=(3, 1))
    mocked_tensor._indices.clone = mock.MagicMock(
        name="clone", return_value=result_indices
    )
    mocked_tensor._values.clone = mock.MagicMock(
        name="clone", return_value=result_values
    )

    result = mocked_tensor.clone()

    assert isinstance(result, BaseSparse)
    assert id(result._indices) == id(result_indices)
    assert id(result._values) == id(result_values)
    mocked_tensor._indices.clone.assert_called_once_with()
    mocked_tensor._values.clone.assert_called_once_with()


@assert_no_out_arr
def test_base_detach_without_values():
    mocked_tensor = BaseSparse(indices=MockTensor(shape=(2, 3), dtype=torch.long))

    result_indices = MockTensor(shape=(2, 3), dtype=torch.long)
    mocked_tensor._indices.detach = mock.MagicMock(
        name="detach", return_value=result_indices
    )

    result = mocked_tensor.detach()
    assert isinstance(result, BaseSparse)
    assert id(result._indices) == id(result_indices)
    mocked_tensor._indices.detach.assert_called_once_with()


@assert_no_out_arr
def test_base_detach_with_values():
    mocked_tensor = BaseSparse(
        indices=MockTensor(shape=(2, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    result_indices = MockTensor(shape=(2, 3), dtype=torch.long)
    result_values = MockTensor(shape=(3, 1))
    mocked_tensor._indices.detach = mock.MagicMock(
        name="detach", return_value=result_indices
    )
    mocked_tensor._values.detach = mock.MagicMock(
        name="detach", return_value=result_values
    )

    result = mocked_tensor.detach()

    assert isinstance(result, BaseSparse)
    assert id(result._indices) == id(result_indices)
    assert id(result._values) == id(result_values)
    mocked_tensor._indices.detach.assert_called_once_with()
    mocked_tensor._values.detach.assert_called_once_with()


@assert_no_out_arr
def test_base_repr():
    mocked_tensor = BaseSparse(
        indices=MockTensor(shape=(2, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    assert (
        repr(mocked_tensor)
        == """BaseSparse(shape=(2, 2),
  indices=MockTensor(shape=(2, 3), dtype=torch.int64, device=cpu),
  values=MockTensor(shape=(3, 1), dtype=torch.float32, device=cpu),
  device="cpu")"""
    )


@assert_no_out_arr
def test_base_to_dense():
    indices = torch.tensor([[0, 1, 2, 2], [0, 1, 0, 2]])

    assert (
        BaseSparse(indices).to_dense()
        == torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    ).all()

    assert (
        BaseSparse(indices, torch.tensor([1, 2, 3, 4])).to_dense()
        == torch.tensor([[1, 0, 0], [0, 2, 0], [3, 0, 4]])
    ).all()

    assert (
        BaseSparse(indices, torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8]])).to_dense()
        == torch.tensor(
            [
                [[1, 5], [0, 0], [0, 0]],
                [[0, 0], [2, 6], [0, 0]],
                [[3, 7], [0, 0], [4, 8]],
            ]
        )
    ).all()
    assert (
        BaseSparse(
            indices,
            torch.tensor(
                [[[1, 5], [4, 8]], [[2, 6], [3, 7]], [[3, 7], [2, 6]], [[4, 8], [1, 5]]]
            ),
        ).to_dense()
        == torch.tensor(
            [
                [[[1, 5], [4, 8]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                [[[0, 0], [0, 0]], [[2, 6], [3, 7]], [[0, 0], [0, 0]]],
                [[[3, 7], [2, 6]], [[0, 0], [0, 0]], [[4, 8], [1, 5]]],
            ]
        )
    ).all()


@assert_no_out_arr
@mock.patch("sparse.base.BaseSparse._is_sorted", mock.MagicMock(return_value=True))
def test_base_join():
    indices = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6],
            [0, 2, 1, 3, 3, 2, 2, 2, 4, 3, 5, 4, 5, 5, 6],
        ]
    )
    values = torch.randint(-32, 32, (indices.shape[1], 7))

    tensor1 = BaseSparse(indices, values[:, [0]], shape=(2, 8, 8))
    tensor2 = BaseSparse(indices, values[:, [1]], shape=(2, 8, 8))
    tensor3 = BaseSparse(indices, values[:, 2:5], shape=(2, 8, 8))
    tensor4 = BaseSparse(indices, values[:, 5:], shape=(2, 8, 8))
    tensor_alt = BaseSparse(indices.clone(), values[:, [0]], shape=(2, 8, 8))

    with pytest.raises(AssertionError):
        BaseSparse.join()

    with pytest.raises(AssertionError):
        BaseSparse.join(tensor1)

    with pytest.raises(AssertionError):
        BaseSparse.join(tensor1, tensor_alt)

    with pytest.raises(AssertionError):
        BaseSparse.join(tensor1, tensor2, tensor_alt)

    result = BaseSparse.join(tensor1, tensor2, tensor3, tensor4)
    assert id(result.indices) == id(indices)
    assert (result.values == values).all()

    result = BaseSparse.join(tensor1, tensor2)
    assert id(result.indices) == id(indices)
    assert (result.values == values[:, :2]).all()

    result = BaseSparse.join(tensor1, tensor3)
    assert id(result.indices) == id(indices)
    assert (result.values == values[:, [0, 2, 3, 4]]).all()

    result = BaseSparse.join(tensor3, tensor4)
    assert id(result.indices) == id(indices)
    assert (result.values == values[:, 2:]).all()


@assert_no_out_arr
def test_base_empty():
    zeros_tensor = BaseSparse(torch.zeros((2, 0), dtype=torch.long), shape=(4, 4))

    assert (zeros_tensor.to_dense() == torch.zeros(4, 4)).all()

    zeros_tensor = BaseSparse(
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0, 16), dtype=torch.long),
        shape=(3, 3),
    )

    assert (zeros_tensor.to_dense() == torch.zeros(3, 3, 16)).all()

    zeros_tensor = BaseSparse.zeros((3, 3), dtype=torch.long, size=16)

    assert zeros_tensor.dtype == torch.long
    assert (zeros_tensor.to_dense() == torch.zeros(3, 3, 16)).all()

    zeros_tensor = BaseSparse.zeros((3, 3))

    assert zeros_tensor.dtype == torch.bool
    assert (zeros_tensor.to_dense() == torch.zeros(3, 3)).all()

    zeros_tensor = BaseSparse.zeros((3, 3), dtype=torch.float32)

    assert zeros_tensor.dtype == torch.float32
    assert (zeros_tensor.to_dense() == torch.zeros(3, 3)).all()

    assert (zeros_tensor.index_sorted() == torch.tensor([0])).all()
    assert zeros_tensor._is_sorted()
    zeros_tensor._remove_sorted_duplicate_()
    zeros_tensor._sort_by_indices_()
