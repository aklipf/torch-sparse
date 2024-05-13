import torch
from sparse import SparseTensor, Mapping
import pytest

from unittest import mock

from tests.utils.assert_sys import assert_no_out_arr
from tests.utils.mock_tensor import MockTensor


@assert_no_out_arr
def test_mapping_repeat_last_dims():
    tensor_random = SparseTensor(torch.randint(0, 4, (4, 16)), shape=(4, 5, 6, 7))
    mapping = Mapping.repeat_last_dims(tensor_random, 3, 3)

    sorted_tensor = mapping.create_target()
    assert (mapping.target.indices == sorted_tensor.indices).all().item()
    assert mapping.target.shape == (4, 5, 6, 7, 5, 6, 7, 5, 6, 7)

    tensor_test = SparseTensor(
        torch.tensor(
            [
                [0, 0, 0, 1, 1, 2, 2, 2, 3],
                [0, 0, 1, 0, 3, 1, 3, 3, 2],
                [2, 3, 1, 2, 3, 1, 2, 3, 1],
            ]
        ),
        shape=(4, 4, 4),
    )
    mapping_test = Mapping.repeat_last_dims(tensor_test, 2, 2)
    assert mapping_test.target.shape == (4, 4, 4, 4, 4)
    assert (
        mapping_test.target.indices
        == torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2],
                [2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 3, 0, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 2],
                [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            ]
        )
    ).all()
    assert (
        mapping_test._batch
        == torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8],
                [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 5, 6, 7, 5, 6, 7, 8],
            ]
        )
    ).all()

    result = tensor_test[:, :, :, None, None] & tensor_test[:, None, None, :, :]
    assert result.shape == mapping_test.target.shape
    assert (result.indices == mapping_test.target.indices).all()


@assert_no_out_arr
def test_mapping_create_from():
    source = SparseTensor(
        MockTensor((3, 32), dtype=torch.long), shape=(1024, 1024, 1024)
    )
    target = SparseTensor(
        MockTensor((5, 128), dtype=torch.long), shape=(1024, 1024, 1024, 1024, 1024)
    )
    batch = MockTensor((1, 128), dtype=torch.long)
    mapping = Mapping(source, target, batch)

    values_source = MockTensor((32, 256))
    values_target = MockTensor((128, 256))

    new_source = mapping.create_source(values_source)
    new_target = mapping.create_target(values_target)

    assert isinstance(new_source, SparseTensor)
    assert new_source.shape == source.shape
    assert id(new_source.indices) == id(source.indices)
    assert id(new_source.values) == id(values_source)

    assert isinstance(new_target, SparseTensor)
    assert new_target.shape == target.shape
    assert id(new_target.indices) == id(target.indices)
    assert id(new_target.values) == id(values_target)

    new_source = mapping.create_source()
    new_target = mapping.create_target()

    assert isinstance(new_source, SparseTensor)
    assert new_source.shape == source.shape
    assert id(new_source.indices) == id(source.indices)
    assert new_source.values is None

    assert isinstance(new_target, SparseTensor)
    assert new_target.shape == target.shape
    assert id(new_target.indices) == id(target.indices)
    assert new_target.values is None


@assert_no_out_arr
def test_mapping_is():
    source = SparseTensor(MockTensor((3, 16), dtype=torch.long), shape=(128, 128, 128))
    target = SparseTensor(
        MockTensor((5, 32), dtype=torch.long), shape=(128, 128, 128, 128, 128)
    )
    batch = MockTensor((1, 32), dtype=torch.long)
    mapping = Mapping(source=source, target=target, batch=batch)

    assert mapping.is_source(source)
    assert not mapping.is_source(target)

    assert not mapping.is_target(source)
    assert mapping.is_target(target)

    assert id(mapping.source.indices) == id(source.indices)
    assert mapping.source.shape == (128, 128, 128)
    assert id(mapping.target.indices) == id(target.indices)
    assert mapping.target.shape == (128, 128, 128, 128, 128)
    assert id(mapping._batch) == id(batch)


@assert_no_out_arr
def test_mapping_selector():
    source = SparseTensor(MockTensor((3, 8), dtype=torch.long), shape=(4, 4, 4))
    target = SparseTensor(MockTensor((5, 32), dtype=torch.long), shape=(4, 4, 4, 4, 4))
    batch = MockTensor((3, 32), dtype=torch.long)
    mapping = Mapping(source, target, batch)

    assert len(mapping) == 3

    with mock.patch("sparse.Mapping.Selector", autospec=True) as mock_selector:
        mapping[0]
        mock_selector.assert_called_once_with(mapping, 0)

    with mock.patch("sparse.Mapping.Selector", autospec=True) as mock_selector:
        mapping[2]
        mock_selector.assert_called_once_with(mapping, 2)

    with mock.patch("tests.utils.MockTensor.__getitem__") as mock_getitem:
        Mapping.Selector(mapping, 0).batch
        mock_getitem.assert_called_once_with(0)

    with mock.patch("tests.utils.MockTensor.__getitem__") as mock_getitem:
        Mapping.Selector(mapping, 1).batch
        mock_getitem.assert_called_once_with(1)

    with mock.patch("tests.utils.MockTensor.__getitem__") as mock_getitem:
        Mapping.Selector(mapping, -1).batch
        mock_getitem.assert_called_once_with(-1)

    with pytest.raises(AssertionError):
        Mapping.Selector(mapping, -3).batch

    with pytest.raises(AssertionError):
        Mapping.Selector(mapping, 3).batch
