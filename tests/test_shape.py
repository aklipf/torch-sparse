import pytest
import torch
import unittest.mock as mock

from sparse.typing import Self
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
        torch.tensor([[105, 364, 456, 273], [789, 334, 105, 64], [56, 641, 3, 586]]),
        shape=(1024, 1024, 1024),
        sort=True,
    )

    # backup sorted indices
    indices_initial = sparse.indices.clone()

    # convert to another bases
    bits = [torch.randint(0, 60, (20,))]

    indices, shape = sparse._indices_to_shape((1 << 30, 1 << 30))
    sparse = SparseShapeMixin(indices, shape=shape)
    indices, shape = sparse._indices_to_shape((1024, 1024, 1024, 1024, 1024, 1024))
    print(indices_initial)
    print(indices)
    print(indices == indices_initial, shape)
