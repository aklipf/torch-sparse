import pytest
import torch

from sparse.cat import SparseCatMixin

from .mock_tensor import MockTensor
from .random_sparse import randint_sparse
from .assert_sys import assert_no_out_arr


@assert_no_out_arr
def test_cat_integration_1d():
    torch.manual_seed(0)
    random = [randint_sparse((4,), ratio=0.5) for _ in range(3)]
    sparse_list = [SparseCatMixin(i, v.int(), shape=(4,)) for i, v in random]

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=0)

    dense_result = torch.zeros((12,), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[i * 4 : (i + 1) * 4] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()


@assert_no_out_arr
def test_cat_integration_2d():
    torch.manual_seed(0)
    random = [randint_sparse((4, 3), ratio=0.5) for _ in range(3)]
    sparse_list = [SparseCatMixin(i, v.int(), shape=(4, 3)) for i, v in random]

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=(0, 1))

    dense_result = torch.zeros((12, 9), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[i * 4 : (i + 1) * 4, i * 3 : (i + 1) * 3] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=1)

    dense_result = torch.zeros((4, 9), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[:, i * 3 : (i + 1) * 3] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=0)

    dense_result = torch.zeros((12, 3), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[i * 4 : (i + 1) * 4, :] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()

    sparse_list = [
        SparseCatMixin(
            torch.cat((torch.zeros_like(i[:1]), i)),
            v.int(),
            shape=(1, 4, 3),
        )
        for i, v in random
    ]
    cat_tensor = SparseCatMixin.cat(sparse_list, dim=0)

    dense_result = torch.zeros((3, 4, 3), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[i] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()


@assert_no_out_arr
def test_cat_integration_3d():
    torch.manual_seed(0)
    random = [randint_sparse((4, 3, 5), ratio=0.5) for _ in range(3)]
    sparse_list = [SparseCatMixin(i, v.int(), shape=(4, 3, 5)) for i, v in random]

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=(0, 1, 2))

    dense_result = torch.zeros((12, 9, 15), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[i * 4 : (i + 1) * 4, i * 3 : (i + 1) * 3, i * 5 : (i + 1) * 5] = (
            s.to_dense()
        )

    assert (cat_tensor.to_dense() == dense_result).all()

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=0)

    dense_result = torch.zeros((12, 3, 5), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[i * 4 : (i + 1) * 4] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=1)

    dense_result = torch.zeros((4, 9, 5), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[:, i * 3 : (i + 1) * 3] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=2)

    dense_result = torch.zeros((4, 3, 15), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[:, :, i * 5 : (i + 1) * 5] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=(0, 1))

    dense_result = torch.zeros((12, 9, 5), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[i * 4 : (i + 1) * 4, i * 3 : (i + 1) * 3] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=(1, 2))

    dense_result = torch.zeros((4, 9, 15), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[:, i * 3 : (i + 1) * 3, i * 5 : (i + 1) * 5] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=(0, 2))

    dense_result = torch.zeros((12, 3, 15), dtype=torch.int32)
    for i, s in enumerate(sparse_list):
        dense_result[i * 4 : (i + 1) * 4, :, i * 5 : (i + 1) * 5] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()


@assert_no_out_arr
def test_cat_assert_cat():
    sparse_list = [
        SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
    ]
    SparseCatMixin._assert_cat(sparse_list, [1])

    with pytest.raises(
        AssertionError, match=r"All inputs must be sparse tensors to be concatenated"
    ):
        sparse_list = [
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
        ]
        sparse_list.append(1)
        SparseCatMixin._assert_cat(sparse_list, [1])

    with pytest.raises(
        AssertionError,
        match=r"All sparse tensors must be on the same device to be concatenated",
    ):
        sparse_list = [
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
        ]
        sparse_list.append(
            SparseCatMixin(
                MockTensor((3, 16), dtype=torch.long, device="cuda"), sort=False
            )
        )
        SparseCatMixin._assert_cat(sparse_list, [1])

    with pytest.raises(
        AssertionError,
        match=r"All sparse tensors must have the same number of dimensions \(ndim\)",
    ):
        sparse_list = [
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
        ]
        sparse_list.append(SparseCatMixin(MockTensor((4, 16), dtype=torch.long)))
        SparseCatMixin._assert_cat(sparse_list, [1])

    with pytest.raises(
        AssertionError,
        match=r"The concatenation dimension must be less than the number of dimention \(ndim\)",
    ):
        sparse_list = [
            SparseCatMixin(MockTensor((3, 16), dtype=torch.long)) for _ in range(4)
        ]
        SparseCatMixin._assert_cat(sparse_list, [3])


@assert_no_out_arr
def test_cat_get_device_shape_ptr():
    sparse_list = [
        SparseCatMixin(
            MockTensor((3, 16), dtype=torch.long, device="cpu"), shape=(3, 5, 4)
        ),
        SparseCatMixin(
            MockTensor((3, 21), dtype=torch.long, device="cpu"), shape=(3, 6, 6)
        ),
        SparseCatMixin(
            MockTensor((3, 8), dtype=torch.long, device="cpu"), shape=(3, 3, 4)
        ),
    ]

    device, out_shape, ptr = SparseCatMixin._get_device_shape_ptr(sparse_list, [1, 2])
    assert device == "cpu"
    assert out_shape == (3, 14, 14)
    assert (
        ptr == torch.tensor([[0, 0], [5, 4], [11, 10], [14, 14]], dtype=torch.long)
    ).all()

    device, out_shape, ptr = SparseCatMixin._get_device_shape_ptr(sparse_list, [1])
    assert device == "cpu"
    assert out_shape == (3, 14, 6)
    assert (ptr == torch.tensor([[0], [5], [11], [14]], dtype=torch.long)).all()

    device, out_shape, ptr = SparseCatMixin._get_device_shape_ptr(sparse_list, [2])
    assert device == "cpu"
    assert out_shape == (3, 6, 14)
    assert (ptr == torch.tensor([[0], [4], [10], [14]], dtype=torch.long)).all()

    sparse_list = [
        SparseCatMixin(
            MockTensor((3, 16), dtype=torch.long, device="cpu"), shape=(3, 5, 4)
        ),
        SparseCatMixin(
            MockTensor((3, 21), dtype=torch.long, device="cpu"), shape=(3, 6, 6)
        ),
        SparseCatMixin(
            MockTensor((3, 8), dtype=torch.long, device="cpu"), shape=(3, 3, 4)
        ),
    ]
    device, out_shape, ptr = SparseCatMixin._get_device_shape_ptr(sparse_list, [0])
    assert device == "cpu"
    assert out_shape == (9, 6, 6)
    assert (ptr == torch.tensor([[0], [3], [6], [9]], dtype=torch.long)).all()

    sparse_list = [
        SparseCatMixin(
            MockTensor((3, 16), dtype=torch.long, device="cpu"), shape=(3, 5, 4)
        ),
        SparseCatMixin(
            MockTensor((3, 21), dtype=torch.long, device="cpu"), shape=(3, 6, 6)
        ),
        SparseCatMixin(
            MockTensor((3, 8), dtype=torch.long, device="cpu"), shape=(3, 3, 4)
        ),
    ]
    device, out_shape, ptr = SparseCatMixin._get_device_shape_ptr(sparse_list, [])
    assert device == "cpu"
    assert out_shape == (3, 6, 6)
    assert ptr is None

    sparse_list = [
        SparseCatMixin(
            MockTensor((3, 16), dtype=torch.long, device="cuda"),
            shape=(3, 5, 4),
            sort=False,
        ),
        SparseCatMixin(
            MockTensor((3, 21), dtype=torch.long, device="cuda"),
            shape=(3, 6, 6),
            sort=False,
        ),
        SparseCatMixin(
            MockTensor((3, 8), dtype=torch.long, device="cuda"),
            shape=(3, 3, 4),
            sort=False,
        ),
    ]
    device, out_shape, ptr = SparseCatMixin._get_device_shape_ptr(sparse_list, [])
    assert device == "cuda"
    assert out_shape == (3, 6, 6)
    assert ptr is None


@assert_no_out_arr
def test_cat_cat_sparse():
    index_list = [
        torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3],
                [2, 5, 8, 0, 1, 4, 2, 3, 1, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7],
                [2, 5, 8, 0, 1, 4, 2, 3, 1, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [8, 9, 10, 10, 10],
                [1, 6, 7, 8, 11],
            ],
            dtype=torch.long,
        ),
    ]
    values_list = [
        torch.tensor(
            [
                [5],
                [9],
                [0],
                [4],
                [6],
                [8],
                [3],
                [4],
                [7],
                [8],
                [6],
                [2],
            ],
            dtype=torch.int,
        ),
        torch.tensor(
            [[1], [7], [9], [3], [2], [2], [0], [5], [8], [7], [4], [9]],
            dtype=torch.int,
        ),
        torch.tensor([[0], [0], [1], [5], [3]], dtype=torch.int),
    ]
    sprase_list = [SparseCatMixin(indices) for indices in index_list]

    result, cat_size = SparseCatMixin._cat_sparse(sprase_list, (11, 12), "cpu")

    assert isinstance(result, SparseCatMixin)
    assert (result._indices == torch.cat(index_list, dim=1)).all()
    assert result._values is None
    assert result.shape == (11, 12)
    assert cat_size.tolist() == [indices.shape[1] for indices in index_list]

    sprase_list = [
        SparseCatMixin(indices, values)
        for indices, values in zip(index_list, values_list)
    ]

    result, cat_size = SparseCatMixin._cat_sparse(sprase_list, (11, 12), "cpu")

    assert isinstance(result, SparseCatMixin)
    assert (result._indices == torch.cat(index_list, dim=1)).all()
    assert (result._values == torch.cat(values_list, dim=0)).all()
    assert result.shape == (11, 12)
    assert cat_size.tolist() == [indices.shape[1] for indices in index_list]


@assert_no_out_arr
def test_cat_reindex_cat_dim():
    sparse = SparseCatMixin(
        torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3],
                [2, 5, 8, 0, 1, 4, 2, 3, 1, 3, 4, 5],
            ],
            dtype=torch.long,
        )
    )

    result = sparse._reindex_cat_dim_(
        [1],
        torch.tensor([[3], [2], [6], [0]], dtype=torch.long),
        torch.tensor([6, 2, 4], dtype=torch.long),
    )

    assert id(result) == id(sparse)
    assert (
        sparse._indices
        == torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3],
                [5, 8, 11, 3, 4, 7, 4, 5, 7, 9, 10, 11],
            ],
            dtype=torch.long,
        )
    ).all()


@assert_no_out_arr
def test_cat_empty():
    torch.manual_seed(0)
    random = [randint_sparse((4, 3), ratio=0.5) for _ in range(3)]
    sparse_list = [SparseCatMixin(i, v.int(), shape=(4, 3)) for i, v in random]
    sparse_list.append(
        SparseCatMixin(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.int),
            shape=(4, 3),
        )
    )

    cat_tensor = SparseCatMixin.cat(sparse_list, dim=(0, 1))

    dense_result = torch.zeros((16, 12), dtype=torch.int32)
    for i, s in enumerate(sparse_list[:-1]):
        dense_result[i * 4 : (i + 1) * 4, i * 3 : (i + 1) * 3] = s.to_dense()

    assert (cat_tensor.to_dense() == dense_result).all()
