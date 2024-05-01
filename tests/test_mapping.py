import torch
from sparse import SparseTensor, Mapping

from .assert_sys import assert_no_out_arr
from .mock_tensor import MockTensor


@assert_no_out_arr
def test_mapping_repeat_last_dims():
    tensor_random = SparseTensor(torch.randint(0, 4, (4, 16)), shape=(4, 5, 6, 7))
    mapping = Mapping.repeat_last_dims(tensor_random, 3, 3)

    sorted_tensor = SparseTensor(mapping.target_indices, shape=mapping.target_shape)
    assert (mapping.target_indices == sorted_tensor.indices).all().item()
    assert mapping.target_shape == (4, 5, 6, 7, 5, 6, 7, 5, 6, 7)

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
    assert mapping_test.target_shape == (4, 4, 4, 4, 4)
    assert (
        mapping_test.target_indices
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


@assert_no_out_arr
def test_mapping_is():
    source = SparseTensor(MockTensor((3, 16), dtype=torch.long), shape=(128, 128, 128))
    target = SparseTensor(
        MockTensor((5, 32), dtype=torch.long), shape=(128, 128, 128, 128, 128)
    )
    batch = MockTensor((5, 32), dtype=torch.long)
    mapping = Mapping(source=source, target=target, mapping=batch)

    assert mapping.is_source(source)
    assert not mapping.is_source(target)

    assert not mapping.is_target(source)
    assert mapping.is_target(target)

    assert id(mapping.source_indices) == id(source.indices)
    assert mapping.source_shape == (128, 128, 128)
    assert id(mapping.target_indices) == id(target.indices)
    assert mapping.target_shape == (128, 128, 128, 128, 128)
    assert id(mapping.mapping) == id(batch)
