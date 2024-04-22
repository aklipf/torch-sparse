from unittest import mock

import pytest
import torch

from sparse.shape import SparseShapeMixin

from .mock_tensor import MockTensor


def __assert_unsqueeze_(
    in_indices: torch.LongTensor, out_indices: torch.LongTensor, dim: int
):
    sparse = SparseShapeMixin(in_indices)
    result = sparse.unsqueeze_(dim)

    assert id(result) == id(sparse)
    assert (sparse.indices == out_indices).all()


def test_shape_unsqueeze_inplace():
    sparse = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long))

    with pytest.raises(AssertionError):
        sparse.unsqueeze_(2.5)

    with pytest.raises(AssertionError):
        sparse.unsqueeze_([1, 2])

    __assert_unsqueeze_(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=torch.long),
        torch.tensor(
            [[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            dtype=torch.long,
        ),
        0,
    )
    __assert_unsqueeze_(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=torch.long),
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]],
            dtype=torch.long,
        ),
        1,
    )
    __assert_unsqueeze_(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=torch.long),
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        2,
    )


def __assert_squeeze_(
    in_indices: torch.LongTensor, out_indices: torch.LongTensor, dim: int
):
    sparse = SparseShapeMixin(in_indices)
    result = sparse.squeeze_(dim)

    assert id(result) == id(sparse)
    assert (sparse.indices == out_indices).all()


def test_shape_squeeze_inplace():
    sparse = SparseShapeMixin(MockTensor((3, 12), dtype=torch.long))

    with pytest.raises(AssertionError):
        sparse.squeeze_(2.5)

    with pytest.raises(AssertionError):
        sparse.squeeze_([1, 2])

    __assert_squeeze_(
        torch.tensor(
            [[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            dtype=torch.long,
        ),
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            dtype=torch.long,
        ),
        0,
    )

    __assert_squeeze_(
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]],
            dtype=torch.long,
        ),
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            dtype=torch.long,
        ),
        1,
    )

    __assert_squeeze_(
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            dtype=torch.long,
        ),
        2,
    )

    __assert_squeeze_(
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [5, 4, 3, 2, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]],
            dtype=torch.long,
        ),
        None,
    )


def test_shape_unsqueeze():
    sparse = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long))
    result = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long))

    sparse.clone = mock.MagicMock("clone", return_value=result)
    result.unsqueeze_ = mock.MagicMock("unsqueeze_")

    unsqueezed = sparse.unsqueeze(1)

    assert id(unsqueezed) == id(result)
    sparse.clone.assert_called_once_with()
    result.unsqueeze_.assert_called_once_with(1)


def test_shape_squeeze():
    sparse = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long))
    result = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long))

    sparse.clone = mock.MagicMock("clone", return_value=result)
    result.squeeze_ = mock.MagicMock("squeeze_")

    unsqueezed = sparse.squeeze(1)

    assert id(unsqueezed) == id(result)
    sparse.clone.assert_called_once_with()
    result.squeeze_.assert_called_once_with(1)


def test_shape_numel():
    assert (
        SparseShapeMixin(MockTensor((2, 12), dtype=torch.long), shape=(5, 6)).numel()
        == 30
    )
    assert (
        SparseShapeMixin(
            MockTensor((4, 12), dtype=torch.long), shape=(3, 4, 2, 1)
        ).numel()
        == 24
    )
    assert (
        SparseShapeMixin(MockTensor((1, 12), dtype=torch.long), shape=(3,)).numel() == 3
    )


def test_shape_inferre_shape():
    sparse = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long), shape=(5, 6))

    assert sparse._inferre_shape((30,)) == [30]
    assert sparse._inferre_shape((3, 10)) == [3, 10]
    assert sparse._inferre_shape((-1,)) == [30]
    assert sparse._inferre_shape((-1, 10)) == [3, 10]
    assert sparse._inferre_shape((3, -1)) == [3, 10]
    assert sparse._inferre_shape((3, 5, 2)) == [3, 5, 2]
    assert sparse._inferre_shape((3, 5, -1)) == [3, 5, 2]
    assert sparse._inferre_shape((3, -1, 2)) == [3, 5, 2]
    assert sparse._inferre_shape((-1, 5, 2)) == [3, 5, 2]
    assert sparse._inferre_shape((3, 5, 2, -1)) == [3, 5, 2, 1]
    assert sparse._inferre_shape((-1, 3, 5, 2)) == [1, 3, 5, 2]

    with pytest.raises(ValueError):
        sparse._inferre_shape((-1, -1, 3))

    with pytest.raises(AssertionError):
        sparse._inferre_shape((-1, 7, 3))


def test_shape_indices_to_shape():
    # sort indices
    indices_initial = torch.randint(0, 1024, (6, 16))
    sparse = SparseShapeMixin(
        torch.tensor([[105, 273, 364, 956], [789, 64, 334, 105], [56, 586, 641, 3]]),
        shape=(1024, 1024, 1024),
    )

    # backup sorted indices
    indices_initial = sparse.indices.clone()

    # convert to another bases
    indices_1d, shape_1d = sparse._indices_to_shape([1 << 30])
    assert shape_1d == [1 << 30]
    assert (
        indices_1d == torch.tensor([110908472, 286327370, 382024321, 1002546179])
    ).all()

    # diffrent shape
    other_shape = [1 << 5, 1 << 13, 1 << 8, 1 << 4]
    other_indices = torch.tensor(
        [[3, 8, 11, 29], [2501, 4368, 3155, 7194], [67, 36, 168, 64], [8, 10, 1, 3]]
    )
    sparse = SparseShapeMixin(indices_1d, shape=(1 << 30,))
    indices, shape_other = sparse._indices_to_shape(other_shape)
    assert (other_indices == indices).all()
    assert shape_other == other_shape

    # back to original
    sparse = SparseShapeMixin(other_indices, shape=other_shape)
    indices, shape_original = sparse._indices_to_shape((1024, 1024, 1024))
    assert (indices == indices_initial).all()
    assert shape_original == [1024, 1024, 1024]


def test_shape_reshape_inplace():
    sparse = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long), shape=(5, 6))
    reshaped_tensor = MockTensor((3, 12), dtype=torch.long)
    reshaped_shape = (3, 5, 2)

    sparse._indices_to_shape = mock.MagicMock(
        "_indices_to_shape", return_value=(reshaped_tensor, reshaped_shape)
    )
    result = sparse.reshape_(reshaped_shape)

    assert id(result) == id(sparse)
    sparse._indices_to_shape.assert_called_once_with(reshaped_shape)
    assert id(sparse.indices) == id(reshaped_tensor)
    assert result.shape == reshaped_shape


def test_shape_reshape_copy():
    sparse = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long), shape=(5, 6))
    cloned = SparseShapeMixin(MockTensor((2, 12), dtype=torch.long), shape=(5, 6))

    reshaped_shape = (3, 5, 2)

    sparse.clone = mock.MagicMock("clone", return_value=cloned)
    cloned.reshape_ = mock.MagicMock("reshape_")

    result = sparse.reshape(reshaped_shape)

    sparse.clone.assert_called_once_with()
    cloned.reshape_.assert_called_once_with(reshaped_shape)
    assert id(result) == id(cloned)
