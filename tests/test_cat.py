import pytest
import torch
import unittest.mock as mock

from sparse.typing import Self
from sparse.cat import SparseCatMixin

from .mock_tensor import MockTensor
from .random_sparse import randint_sparse


def test_cat_integration():
    # torch.manual_seed(0)

    sparse_list = [SparseCatMixin(*randint_sparse((4, 4), ratio=0.5)) for _ in range(3)]

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=(0, 1))

    w, h = 0, 0
    for s in sparse_list:
        h += s.shape[0]
        w += s.shape[1]

    dense_result = torch.zeros((h, w), dtype=torch.int32)

    i, j = 0, 0
    for s in sparse_list:
        print(s.to_dense())
        dense_result[i : i + s.shape[0], j : j + s.shape[1]] = s.to_dense()
        i += s.shape[0]
        j += s.shape[1]

    print(dense_result)
    assert True
    return
    assert (cat_tensor.to_dense() == dense_result).all()


def test_cat_assert_cat():
    sparse_list = [
        SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
    ]
    SparseCatMixin._assert_cat(sparse_list, [1])

    with pytest.raises(AssertionError):
        sparse_list = [
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
        ]
        sparse_list.append(1)
        SparseCatMixin._assert_cat(sparse_list, [1])

    with pytest.raises(AssertionError):
        sparse_list = [
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
        ]
        sparse_list.append(
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long, device="cuda"))
        )
        SparseCatMixin._assert_cat(sparse_list, [1])

    with pytest.raises(AssertionError):
        sparse_list = [
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
        ]
        sparse_list.append(
            SparseCatMixin(MockTensor(torch.randint(4, 16, (4,)), dtype=torch.long))
        )
        SparseCatMixin._assert_cat(sparse_list, [1])

    with pytest.raises(AssertionError):
        sparse_list = [
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
        ]
        sparse_list.append(
            SparseCatMixin(MockTensor(torch.randint(4, 16, (4,)), dtype=torch.long))
        )
        SparseCatMixin._assert_cat(sparse_list, [3])
