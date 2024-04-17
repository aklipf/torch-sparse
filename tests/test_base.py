import pytest
import torch
import unittest.mock as mock

from sparse.typing import Self
from sparse.base import BaseSparse

from .mock_tensor import MockTensor

def test_base_sort():
    indices = torch.tensor(
        [
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6],
        ]
    )
    perm = torch.randperm(indices.shape[1])

    sparse = BaseSparse(indices=indices[:, perm])

    assert (sparse.indices == indices).all().item()

    indices = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6],
            [0, 2, 1, 3, 3, 2, 2, 2, 4, 3, 5, 4, 5, 5, 6],
        ]
    )
    perm = torch.randperm(indices.shape[1])

    sparse = BaseSparse(indices=indices[:, perm])

    assert (sparse.indices == indices).all().item()


def test_base_no_sort():
    indices = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6],
            [0, 2, 1, 3, 3, 2, 2, 2, 4, 3, 5, 4, 5, 5, 6],
        ]
    )
    perm = torch.randperm(indices.shape[1])

    sparse = BaseSparse(indices=indices[:, perm], sort=False)

    assert (sparse.indices == indices[:, perm]).all().item()


def test_base_sort_with_values():
    indices = torch.randperm(128)[:12].sort().values.unsqueeze(0)
    values = torch.arange(indices.shape[1], dtype=torch.long)
    perm = torch.randperm(indices.shape[1])

    sparse = BaseSparse(indices=indices[:, perm], values=values[perm])

    assert (sparse.indices == indices).all().item()
    assert (sparse.values.flatten() == values).all().item()
    assert sparse.values.shape == (12, 1)

    sparse = BaseSparse(indices=indices[:, perm], values=values[perm].unsqueeze(1))

    assert (sparse.indices == indices).all().item()
    assert (sparse.values.flatten() == values).all().item()
    assert sparse.values.shape == (12, 1)


def test_base_assert():
    # assert indices ndim and dtype
    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([]))

    BaseSparse(torch.tensor([[0]], dtype=torch.long))

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0]], dtype=torch.float32))

    # assert values ndim
    BaseSparse(torch.tensor([[0]], dtype=torch.long), values=torch.tensor([0]))
    BaseSparse(torch.tensor([[0]], dtype=torch.long), values=torch.tensor([[0]]))

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0]], dtype=torch.long), values=torch.tensor(0))

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0]], dtype=torch.long), values=torch.tensor([[[0]]]))

    # assert compare indices shape and values shape
    with pytest.raises(AssertionError):
        BaseSparse(
            torch.tensor([[0]], dtype=torch.long), values=torch.tensor([[[0, 0]]])
        )

    # assert compare indices shape and shape parameter
    BaseSparse(torch.tensor([[0]], dtype=torch.long), shape=(2,))
    BaseSparse(torch.tensor([[0], [0]], dtype=torch.long), shape=(2, 3))

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0]], dtype=torch.long), shape=(2, 3))

    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[0], [0]], dtype=torch.long), shape=(2,))

    # assert not empty when auto detect shape
    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[]], dtype=torch.long))
    with pytest.raises(AssertionError):
        BaseSparse(torch.tensor([[], [], []], dtype=torch.long))


def test_base_shape():
    sparse = BaseSparse(MockTensor(shape=(1, 3), dtype=torch.long), shape=(5,))
    assert sparse.shape == (5,)

    sparse = BaseSparse(MockTensor(shape=(2, 3), dtype=torch.long), shape=(12, 4))
    assert sparse.shape == (12, 4)

    indices = MockTensor(shape=(1, 3), dtype=torch.long)
    indices.amax = mock.Mock("amax", return_value=torch.tensor([3], dtype=torch.long))

    sparse = BaseSparse(indices)
    assert sparse.shape == (4,)

    indices.amax.assert_called_once_with(dim=1)

    indices = MockTensor(shape=(2, 3), dtype=torch.long)
    indices.amax = mock.Mock(
        "amax", return_value=torch.tensor([3, 5], dtype=torch.long)
    )

    sparse = BaseSparse(indices)
    assert sparse.shape == (4, 6)

    indices.amax.assert_called_once_with(dim=1)


def test_base_dims():
    assert BaseSparse(MockTensor(shape=(1, 3), dtype=torch.long)).dims == (0,)
    assert BaseSparse(MockTensor(shape=(2, 3), dtype=torch.long)).dims == (0, 1)
    assert BaseSparse(MockTensor(shape=(3, 3), dtype=torch.long)).dims == (0, 1, 2)


def test_base_dtype():
    assert BaseSparse(MockTensor(shape=(2, 3), dtype=torch.long)).dtype == torch.float32
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


def test_base_device():
    assert BaseSparse(
        MockTensor(shape=(2, 3), dtype=torch.long, device=torch.device("cuda")),
        sort=False,
    ).device == torch.device("cuda")


def test_base_to_without_values():
    mocked_sparce = BaseSparse(indices=MockTensor(shape=(2, 3), dtype=torch.long))

    mocked_sparce.indices.to = mock.MagicMock(
        name="to", return_value=MockTensor(shape=(2, 3), dtype=torch.long)
    )

    mocked_sparce.to("cuda")

    mocked_sparce.indices.to.assert_called_once_with("cuda")


def test_base_to_with_values():
    mocked_sparce = BaseSparse(
        indices=MockTensor(shape=(2, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    mocked_sparce.indices.to = mock.MagicMock(
        name="to", return_value=MockTensor(shape=(2, 3), dtype=torch.long)
    )
    mocked_sparce.values.to = mock.MagicMock(
        name="to", return_value=MockTensor(shape=(3, 1))
    )

    mocked_sparce.to("cuda")

    mocked_sparce.indices.to.assert_called_once_with("cuda")
    mocked_sparce.values.to.assert_called_once_with("cuda")


def test_base_clone_without_values():
    mocked_sparce = BaseSparse(indices=MockTensor(shape=(2, 3), dtype=torch.long))

    mocked_sparce.indices.clone = mock.MagicMock(
        name="clone", return_value=MockTensor(shape=(2, 3), dtype=torch.long)
    )

    mocked_sparce.clone()
    mocked_sparce.indices.clone.assert_called_once_with()


def test_base_clone_with_values():
    mocked_sparce = BaseSparse(
        indices=MockTensor(shape=(2, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    mocked_sparce.indices.clone = mock.MagicMock(
        name="clone", return_value=MockTensor(shape=(2, 3), dtype=torch.long)
    )
    mocked_sparce.values.clone = mock.MagicMock(
        name="clone", return_value=MockTensor(shape=(3, 1))
    )

    mocked_sparce.clone()

    mocked_sparce.indices.clone.assert_called_once_with()
    mocked_sparce.values.clone.assert_called_once_with()


def test_base_detach_without_values():
    mocked_sparce = BaseSparse(indices=MockTensor(shape=(2, 3), dtype=torch.long))

    mocked_sparce.indices.detach = mock.MagicMock(
        name="detach", return_value=MockTensor(shape=(2, 3), dtype=torch.long)
    )

    mocked_sparce.detach()
    mocked_sparce.indices.detach.assert_called_once_with()


def test_base_detach_with_values():
    mocked_sparce = BaseSparse(
        indices=MockTensor(shape=(2, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    mocked_sparce.indices.detach = mock.MagicMock(
        name="detach", return_value=MockTensor(shape=(2, 3), dtype=torch.long)
    )
    mocked_sparce.values.detach = mock.MagicMock(
        name="detach", return_value=MockTensor(shape=(3, 1))
    )

    mocked_sparce.detach()

    mocked_sparce.indices.detach.assert_called_once_with()
    mocked_sparce.values.detach.assert_called_once_with()


def test_base_repr():
    mocked_sparce = BaseSparse(
        indices=MockTensor(shape=(2, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    assert (
        repr(mocked_sparce)
        == """BaseSparse(shape=(2, 2),
  indices=MockTensor(shape=(2, 3), dtype=torch.int64, device=cpu),
  values=MockTensor(shape=(3, 1), dtype=torch.float32, device=cpu),
  device="cpu")"""
    )


def test_base_dense():
    indices = torch.tensor([[0, 1, 2, 2], [0, 1, 0, 2]])

    assert (
        BaseSparse(indices).dense == torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    ).all()

    assert (
        BaseSparse(indices, torch.tensor([1, 2, 3, 4])).dense
        == torch.tensor([[1, 0, 0], [0, 2, 0], [3, 0, 4]])
    ).all()

    assert (
        BaseSparse(indices, torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8]])).dense
        == torch.tensor(
            [
                [[1, 5], [0, 0], [0, 0]],
                [[0, 0], [2, 6], [0, 0]],
                [[3, 7], [0, 0], [4, 8]],
            ]
        )
    ).all()


def test_base_dim_to_list():
    assert BaseSparse._dim_to_list() == []
    assert BaseSparse._dim_to_list(dim=None) == []
    assert BaseSparse._dim_to_list(dim=3) == [3]

    assert BaseSparse._dim_to_list(dim=(1, 2)) == [1, 2]

    with pytest.raises(AssertionError):
        BaseSparse._dim_to_list(dim=(1, 2, 2))


def test_base_included_dims():
    mocked_sparce = BaseSparse(
        indices=MockTensor(shape=(4, 3), dtype=torch.long),
        values=MockTensor(shape=(3, 1)),
    )

    assert mocked_sparce._included_dims() == [0, 1, 2, 3]
    assert mocked_sparce._included_dims(except_dim=1) == [0, 2, 3]
    assert mocked_sparce._included_dims(except_dim=(1, 2)) == [0, 3]
    assert mocked_sparce._included_dims(except_dim=(1, 2, 1, 2)) == [0, 3]
    assert mocked_sparce._included_dims(except_dim=(2, 3, 1)) == [0]


def test_base_argsort_indices():
    indices = torch.randint(0, 1024, (3, 1024))
    sparse = BaseSparse(indices, sort=False)

    perm = sparse._argsort_indices()
    sorted = (
        (sparse.indices[0, perm] << 20)
        + (sparse.indices[1, perm] << 10)
        + sparse.indices[2, perm]
    )
    assert (sorted.diff() >= 0).all()

    perm = sparse._argsort_indices(1)
    sorted = (sparse.indices[0, perm] << 10) + sparse.indices[2, perm]
    assert (sorted.diff() >= 0).all()

    perm = sparse._argsort_indices((0, 1))
    sorted = sparse.indices[2, perm]
    assert (sorted.diff() >= 0).all()

    perm = sparse._argsort_indices((1, 2))
    sorted = sparse.indices[0, perm]
    assert (sorted.diff() >= 0).all()


def test_base_sort_indices():
    indices = torch.randint(0, 1024, (3, 1024))
    sparse = BaseSparse(indices.clone(), sort=False)

    sparse._sort_indices_()
    sorted = (sparse.indices[0] << 20) + (sparse.indices[1] << 10) + sparse.indices[2]
    assert (sorted.diff() >= 0).all()

    indices = torch.randperm(1024).unsqueeze(0)
    values = torch.randn(1024)
    sparse = BaseSparse(indices.clone(), values.clone(), sort=False)

    sparse._sort_indices_()
    assert (values == sparse.values[indices[0]].flatten()).all()


def test_base_is_sorted():
    sorted = torch.randint(0, 1 << 30, (64,)).sort().values
    indices = torch.stack((sorted >> 20, (sorted >> 10) & 0x3FF, sorted & 0x3FF))

    sparse = BaseSparse(indices.clone(), sort=False)
    assert sparse._is_sorted()

    indices = torch.randint(0, 1024, (3, 1024))

    sparse = BaseSparse(indices.clone(), sort=False)
    assert not sparse._is_sorted()


def test_base_prod():
    assert BaseSparse._prod([]) == 1
    assert BaseSparse._prod([1]) == 1
    assert BaseSparse._prod([2]) == 2
    assert BaseSparse._prod([2, 3]) == 6
    assert BaseSparse._prod([2, 3, 4]) == 24
