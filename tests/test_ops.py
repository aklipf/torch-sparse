from unittest import mock

import pytest
import torch

from sparse.ops import SparseOpsMixin, _intersection_mask, _union_mask

from tests.utils.mock_tensor import MockTensor
from tests.utils.assert_sys import assert_no_out_arr
from tests.utils.assert_equals_tensors import assert_equal_tensors, assert_equal_bool_tensors

a_indices = torch.tensor(
    [[0, 3, 1, 1, 2, 2, 3], [0, 0, 1, 2, 1, 2, 3]], dtype=torch.long
)
a_values = torch.tensor([[1], [5], [1], [1], [1], [1], [1]], dtype=torch.int32)
a_values_mul = torch.tensor(
    [
        [[1, 1, 2], [1, 4, 8]],
        [[8, 4, 6], [1, 0, 1]],
        [[1, 4, 2], [2, 4, 6]],
        [[4, 3, 9], [5, 5, 8]],
        [[8, 5, 8], [7, 8, 2]],
        [[7, 9, 3], [1, 6, 6]],
        [[1, 8, 2], [3, 4, 3]],
    ],
    dtype=torch.int32,
)
a_bool_values = torch.tensor(
    [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, True, False],
        [False, False, True],
        [False, True, False],
        [True, True, False],
    ],
    dtype=torch.bool,
)
b_indices = torch.tensor([[0, 1, 1, 2, 3], [0, 1, 2, 2, 3]], dtype=torch.long)
b_values = torch.tensor([[1], [2], [1], [1], [1]], dtype=torch.int32)
b_values_mul = torch.tensor(
    [
        [[8, 1, 2], [4, 7, 8]],
        [[1, 11, 8], [5, 4, 2]],
        [[0, 5, 2], [3, 4, 5]],
        [[1, 3, 3], [2, 3, 8]],
        [[8, 2, 6], [8, 2, 0]],
    ],
    dtype=torch.int32,
)
b_bool_values = torch.tensor(
    [
        [False, True, False],
        [True, True, False],
        [False, True, True],
        [True, False, False],
        [False, False, False],
    ],
    dtype=torch.bool,
)
c_indices = torch.tensor([[0, 1, 1, 2, 2], [0, 1, 2, 2, 3]], dtype=torch.long)
c_values = torch.tensor([[-2], [-2], [1], [1], [1]], dtype=torch.int32)
c_values_mul = torch.tensor(
    [
        [[6, 1, 2], [1, 3, 5]],
        [[5, 4, 8], [5, 5, 2]],
        [[4, 7, 2], [5, 6, 4]],
        [[8, 4, 5], [2, 7, 8]],
        [[1, 1, 2], [0, 1, 2]],
    ],
    dtype=torch.int32,
)
c_bool_values = torch.tensor(
    [
        [True, False, False],
        [True, True, True],
        [False, True, True],
        [True, True, False],
        [False, True, False],
    ],
    dtype=torch.bool,
)

a = SparseOpsMixin(a_indices, a_values, shape=(4, 4))
a_mul = SparseOpsMixin(a_indices, a_values_mul, shape=(4, 4))
a_bool = SparseOpsMixin(a_indices, shape=(4, 4))
a_bool_mul = SparseOpsMixin(a_indices, a_bool_values, shape=(4, 4))
b = SparseOpsMixin(b_indices, b_values, shape=(4, 4))
b_mul = SparseOpsMixin(b_indices, b_values_mul, shape=(4, 4))
b_bool = SparseOpsMixin(b_indices, shape=(4, 4))
b_bool_mul = SparseOpsMixin(b_indices, b_bool_values, shape=(4, 4))
c = SparseOpsMixin(c_indices, c_values, shape=(4, 4))
c_mul = SparseOpsMixin(c_indices, c_values_mul, shape=(4, 4))
c_bool = SparseOpsMixin(c_indices, shape=(4, 4))
c_bool_mul = SparseOpsMixin(c_indices, c_bool_values, shape=(4, 4))


@assert_no_out_arr
def test_ops_intersection_mask():
    mask = _intersection_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 0, 1]], dtype=torch.long
        ),
        2,
    )
    assert_equal_bool_tensors(
        mask, torch.tensor([True, False, False, False, True, False, False, False])
    )

    mask = _intersection_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 1, 1]], dtype=torch.long
        ),
        2,
    )
    assert_equal_bool_tensors(
        mask, torch.tensor([True, False, False, False, True, False, True, False])
    )


@assert_no_out_arr
def test_ops_union_mask():
    mask = _union_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 0, 1]], dtype=torch.long
        ),
        2,
    )

    assert_equal_bool_tensors(
        mask, torch.tensor([True, False, True, True, True, False, True, True])
    )

    mask = _union_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 1, 1]], dtype=torch.long
        ),
        2,
    )

    assert_equal_bool_tensors(
        mask, torch.tensor([True, False, True, True, True, False, True, False])
    )


@assert_no_out_arr
def test_ops_apply():
    assert_equal_tensors(
        a.apply(torch.pow, exponent=2).values, torch.pow(a.values, exponent=2)
    )
    assert_equal_tensors(
        b.apply(torch.pow, exponent=2).values, torch.pow(b.values, exponent=2)
    )
    assert_equal_tensors(
        c.apply(torch.pow, exponent=2).values, torch.pow(c.values, exponent=2)
    )

    assert_equal_tensors(
        a_mul.apply(torch.pow, exponent=2).values, torch.pow(a_mul.values, exponent=2)
    )
    assert_equal_tensors(
        b_mul.apply(torch.pow, exponent=2).values, torch.pow(b_mul.values, exponent=2)
    )
    assert_equal_tensors(
        c_mul.apply(torch.pow, exponent=2).values, torch.pow(c_mul.values, exponent=2)
    )


@assert_no_out_arr
def test_ops_and():
    assert_equal_bool_tensors(
        (a_bool & b_bool).to_dense(),
        (a_bool.to_dense() & b_bool.to_dense()),
    )

    assert_equal_bool_tensors(
        (a_bool & b_bool & c_bool).to_dense(),
        (a_bool.to_dense() & b_bool.to_dense() & c_bool.to_dense()),
    )

    assert_equal_bool_tensors(
        (a_bool_mul & b_bool_mul).to_dense(),
        (a_bool_mul.to_dense() & b_bool_mul.to_dense()),
    )

    assert_equal_bool_tensors(
        (a_bool_mul & b_bool_mul & c_bool_mul).to_dense(),
        (a_bool_mul.to_dense() & b_bool_mul.to_dense() & c_bool_mul.to_dense()),
    )


@assert_no_out_arr
def test_ops_or():
    assert_equal_bool_tensors(
        (a_bool | b_bool).to_dense(),
        (a_bool.to_dense() | b_bool.to_dense()),
    )

    assert_equal_bool_tensors(
        (a_bool | b_bool | c_bool).to_dense(),
        (a_bool.to_dense() | b_bool.to_dense() | c_bool.to_dense()),
    )

    assert_equal_bool_tensors(
        (a_bool_mul | b_bool_mul).to_dense(),
        (a_bool_mul.to_dense() | b_bool_mul.to_dense()),
    )

    assert_equal_bool_tensors(
        (a_bool_mul | b_bool_mul | c_bool_mul).to_dense(),
        (a_bool_mul.to_dense() | b_bool_mul.to_dense() | c_bool_mul.to_dense()),
    )


@assert_no_out_arr
def test_ops_add():
    assert (
        (a + b + c).to_dense() == (a.to_dense() + b.to_dense() + c.to_dense())
    ).all()

    assert (
        (a_mul + b_mul + c_mul).to_dense()
        == (a_mul.to_dense() + b_mul.to_dense() + c_mul.to_dense())
    ).all()


@assert_no_out_arr
def test_ops_mul():
    assert (
        (a * b * c).to_dense() == (a.to_dense() * b.to_dense() * c.to_dense())
    ).all()

    assert (
        (a_mul * b_mul * c_mul).to_dense()
        == (a_mul.to_dense() * b_mul.to_dense() * c_mul.to_dense())
    ).all()


@assert_no_out_arr
def test_ops_mul_scalar():
    assert ((a * 3).to_dense() == (a.to_dense() * 3)).all()
    assert ((b * 3).to_dense() == (b.to_dense() * 3)).all()
    assert ((c * 3).to_dense() == (c.to_dense() * 3)).all()

    assert ((a_mul * 3).to_dense() == (a_mul.to_dense() * 3)).all()
    assert ((b_mul * 3).to_dense() == (b_mul.to_dense() * 3)).all()
    assert ((c_mul * 3).to_dense() == (c_mul.to_dense() * 3)).all()

    matrix = torch.tensor([[1, -2, 3], [-4, 5, -6]])

    assert ((a_mul * matrix).to_dense() == (a_mul.to_dense() * matrix)).all()
    assert ((b_mul * matrix).to_dense() == (b_mul.to_dense() * matrix)).all()
    assert ((c_mul * matrix).to_dense() == (c_mul.to_dense() * matrix)).all()


@assert_no_out_arr
def test_ops_rmul_scalar():
    assert ((3 * a).to_dense() == (3 * a.to_dense())).all()
    assert ((3 * b).to_dense() == (3 * b.to_dense())).all()
    assert ((3 * c).to_dense() == (3 * c.to_dense())).all()

    assert ((3 * a_mul).to_dense() == (3 * a_mul.to_dense())).all()
    assert ((3 * b_mul).to_dense() == (3 * b_mul.to_dense())).all()
    assert ((3 * c_mul).to_dense() == (3 * c_mul.to_dense())).all()

    matrix = torch.tensor([[1, -2, 3], [-4, 5, -6]])

    assert ((matrix * a_mul).to_dense() == (matrix * a_mul.to_dense())).all()
    assert ((matrix * b_mul).to_dense() == (matrix * b_mul.to_dense())).all()
    assert ((matrix * c_mul).to_dense() == (matrix * c_mul.to_dense())).all()


@assert_no_out_arr
def test_ops_truediv_scalar():
    assert ((a / 6).to_dense() == (a.to_dense() / 6)).all()
    assert ((b / 6).to_dense() == (b.to_dense() / 6)).all()
    assert ((c / 6).to_dense() == (c.to_dense() / 6)).all()

    assert ((a_mul / 6).to_dense() == (a_mul.to_dense() / 6)).all()
    assert ((b_mul / 6).to_dense() == (b_mul.to_dense() / 6)).all()
    assert ((c_mul / 6).to_dense() == (c_mul.to_dense() / 6)).all()

    matrix = torch.tensor([[1, -2, 3], [-4, 5, -6]])

    assert ((a_mul / matrix).to_dense() == (a_mul.to_dense() / matrix)).all()
    assert ((b_mul / matrix).to_dense() == (b_mul.to_dense() / matrix)).all()
    assert ((c_mul / matrix).to_dense() == (c_mul.to_dense() / matrix)).all()


@assert_no_out_arr
def test_ops_floordiv_scalar():
    assert ((a // 2).to_dense() == (a.to_dense() // 2)).all()
    assert ((b // 2).to_dense() == (b.to_dense() // 2)).all()
    assert ((c // 2).to_dense() == (c.to_dense() // 2)).all()

    assert ((a_mul // 2).to_dense() == (a_mul.to_dense() // 2)).all()
    assert ((b_mul // 2).to_dense() == (b_mul.to_dense() // 2)).all()
    assert ((c_mul // 2).to_dense() == (c_mul.to_dense() // 2)).all()

    matrix = torch.tensor([[1, -2, 3], [-4, 5, -6]])

    assert ((a_mul // matrix).to_dense() == (a_mul.to_dense() // matrix)).all()
    assert ((b_mul // matrix).to_dense() == (b_mul.to_dense() // matrix)).all()
    assert ((c_mul // matrix).to_dense() == (c_mul.to_dense() // matrix)).all()


@assert_no_out_arr
def test_ops_mod_scalar():
    assert ((a % 2).to_dense() == (a.to_dense() % 2)).all()
    assert ((b % 2).to_dense() == (b.to_dense() % 2)).all()
    assert ((c % 2).to_dense() == (c.to_dense() % 2)).all()

    assert ((a_mul % 2).to_dense() == (a_mul.to_dense() % 2)).all()
    assert ((b_mul % 2).to_dense() == (b_mul.to_dense() % 2)).all()
    assert ((c_mul % 2).to_dense() == (c_mul.to_dense() % 2)).all()

    matrix = torch.tensor([[1, -2, 3], [-4, 5, -6]])

    assert ((a_mul % matrix).to_dense() == (a_mul.to_dense() % matrix)).all()
    assert ((b_mul % matrix).to_dense() == (b_mul.to_dense() % matrix)).all()
    assert ((c_mul % matrix).to_dense() == (c_mul.to_dense() % matrix)).all()


@assert_no_out_arr
def test_ops_sub():
    assert (
        (a - b - c).to_dense() == (a.to_dense() - b.to_dense() - c.to_dense())
    ).all()

    assert (
        (a_mul - b_mul - c_mul).to_dense()
        == (a_mul.to_dense() - b_mul.to_dense() - c_mul.to_dense())
    ).all()


@assert_no_out_arr
def test_ops_neg():
    assert ((-a).to_dense() == -a.to_dense()).all()
    assert ((-b).to_dense() == -b.to_dense()).all()
    assert ((-c).to_dense() == -c.to_dense()).all()

    assert ((-a_mul).to_dense() == -a_mul.to_dense()).all()
    assert ((-b_mul).to_dense() == -b_mul.to_dense()).all()
    assert ((-c_mul).to_dense() == -c_mul.to_dense()).all()


@assert_no_out_arr
def test_ops_sparse_cart_prod():
    with pytest.raises(
        AssertionError, match="No input tensor, can't calculate the cartesian product"
    ):
        SparseOpsMixin._sparse_cart_prod()

    indices = torch.tensor([0, 2, 3, 6, 7, 8, 10, 15, 16], dtype=torch.long)
    result = SparseOpsMixin._sparse_cart_prod(indices)
    assert (indices.unsqueeze(0) == result).all()

    result = SparseOpsMixin._sparse_cart_prod(
        torch.tensor([0, 2, 3], dtype=torch.long),
        torch.tensor([1, 6], dtype=torch.long),
    )
    assert (
        result
        == torch.tensor([[0, 0, 2, 2, 3, 3], [1, 6, 1, 6, 1, 6]], dtype=torch.long)
    ).all()

    result = SparseOpsMixin._sparse_cart_prod(
        torch.tensor([0, 2, 3], dtype=torch.long),
        torch.tensor([1, 6], dtype=torch.long),
        torch.tensor([0, 2, 3], dtype=torch.long),
    )
    assert (
        result
        == torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                [1, 1, 1, 6, 6, 6, 1, 1, 1, 6, 6, 6, 1, 1, 1, 6, 6, 6],
                [0, 2, 3, 0, 2, 3, 0, 2, 3, 0, 2, 3, 0, 2, 3, 0, 2, 3],
            ],
            dtype=torch.long,
        )
    ).all()


@assert_no_out_arr
def test_ops_repeat_indices():
    indices = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 4, 4], [0, 0, 0, 0], [2, 1, 2, 3], [0, 0, 0, 0]]
    )

    tensor = SparseOpsMixin(indices, shape=(1, 8, 1, 4, 1))

    result = tensor._repeat_indices(
        [torch.tensor([0, 2, 3], dtype=torch.long)], [0], [4]
    )
    assert result.shape == (4, 8, 1, 4, 1)
    assert (
        result.indices
        == torch.tensor(
            [
                [0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3],
                [0, 1, 4, 4, 0, 1, 4, 4, 0, 1, 4, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    ).all()

    result = tensor._repeat_indices(
        [torch.tensor([0, 2, 3], dtype=torch.long)], [2], [4]
    )
    assert result.shape == (1, 8, 4, 4, 1)
    assert (
        result.indices
        == torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 4, 4, 4, 4, 4, 4],
                [0, 2, 3, 0, 2, 3, 0, 0, 2, 2, 3, 3],
                [2, 2, 2, 1, 1, 1, 2, 3, 2, 3, 2, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    ).all()

    result = tensor._repeat_indices(
        [torch.tensor([0, 2, 3], dtype=torch.long)], [4], [4]
    )

    assert result.shape == (1, 8, 1, 4, 4)
    assert (
        result.indices
        == torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 4, 4, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [0, 2, 3, 0, 2, 3, 0, 2, 3, 0, 2, 3],
            ]
        )
    ).all()

    result = tensor._repeat_indices(
        [
            torch.tensor([4, 6], dtype=torch.long),
            torch.tensor([0, 2], dtype=torch.long),
        ],
        [0, 4],
        [8, 4],
    )

    assert result.shape == (8, 8, 1, 4, 4)
    assert (
        result.indices
        == torch.tensor(
            [
                [4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6],
                [0, 0, 1, 1, 4, 4, 4, 4, 0, 0, 1, 1, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3],
                [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
            ]
        )
    ).all()

    result = tensor._repeat_indices(
        [
            torch.tensor([4, 6], dtype=torch.long),
            torch.tensor([3, 5], dtype=torch.long),
        ],
        [0, 2],
        [8, 6],
    )

    assert result.shape == (8, 8, 6, 4, 1)
    assert (
        result.indices
        == torch.tensor(
            [
                [4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6],
                [0, 0, 1, 1, 4, 4, 4, 4, 0, 0, 1, 1, 4, 4, 4, 4],
                [3, 5, 3, 5, 3, 3, 5, 5, 3, 5, 3, 5, 3, 3, 5, 5],
                [2, 2, 1, 1, 2, 3, 2, 3, 2, 2, 1, 1, 2, 3, 2, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    ).all()

    result = tensor._repeat_indices(
        [
            torch.tensor([4, 6], dtype=torch.long),
            torch.tensor([3, 5], dtype=torch.long),
        ],
        [0, 4],
        [8, 6],
    )

    assert result.shape == (8, 8, 1, 4, 6)
    assert (
        result.indices
        == torch.tensor(
            [
                [4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6],
                [0, 0, 1, 1, 4, 4, 4, 4, 0, 0, 1, 1, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3],
                [3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5],
            ]
        )
    ).all()


@assert_no_out_arr
def test_ops_cast_sparse_tensors():

    a = SparseOpsMixin(torch.tensor([[0, 1, 2]], dtype=torch.long), shape=(8,))
    b = SparseOpsMixin(torch.tensor([[2, 5, 7]], dtype=torch.long), shape=(8,))

    result = SparseOpsMixin._cast_sparse_tensors([a, b])
    assert id(result[0]) == id(a)
    assert id(result[1]) == id(b)

    a = SparseOpsMixin(
        torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long), shape=(3, 8)
    )
    b = SparseOpsMixin(
        torch.tensor([[0, 0, 0], [2, 5, 7]], dtype=torch.long), shape=(1, 8)
    )

    cast_a, cast_b = SparseOpsMixin._cast_sparse_tensors([a, b])
    assert cast_a.shape == cast_b.shape == (3, 8)
    assert (cast_a.indices == torch.tensor([[0, 1, 2], [0, 0, 0]])).all()
    assert (
        cast_b.indices
        == torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 5, 7, 2, 5, 7, 2, 5, 7]])
    ).all()

    a = SparseOpsMixin(
        torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long), shape=(3, 1)
    )
    b = SparseOpsMixin(
        torch.tensor([[0, 0, 0], [2, 5, 7]], dtype=torch.long), shape=(3, 8)
    )
    cast_a, cast_b = SparseOpsMixin._cast_sparse_tensors([a, b])

    assert cast_a.shape == cast_b.shape == (3, 8)
    assert (
        cast_a.indices
        == torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 5, 7, 2, 5, 7, 2, 5, 7]])
    ).all()
    assert (cast_b.indices == torch.tensor([[0, 0, 0], [2, 5, 7]])).all()

    a = SparseOpsMixin(
        torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long), shape=(3, 1)
    )
    b = SparseOpsMixin(
        torch.tensor([[0, 0, 0], [2, 5, 7]], dtype=torch.long), shape=(1, 8)
    )
    cast_a, cast_b = SparseOpsMixin._cast_sparse_tensors([a, b])

    assert cast_a.shape == cast_b.shape == (3, 8)
    assert (
        cast_a.indices
        == torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 5, 7, 2, 5, 7, 2, 5, 7]])
    ).all()
    assert (
        cast_b.indices
        == torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 5, 7, 2, 5, 7, 2, 5, 7]])
    ).all()

    a = SparseOpsMixin(
        torch.tensor([[0, 1, 2], [0, 0, 0], [0, 0, 0]], dtype=torch.long),
        shape=(3, 1, 1),
    )
    b = SparseOpsMixin(
        torch.tensor([[0, 0, 0], [2, 5, 7], [0, 0, 0]], dtype=torch.long),
        shape=(1, 8, 1),
    )
    cast_a, cast_b = SparseOpsMixin._cast_sparse_tensors([a, b])

    assert cast_a.shape == cast_b.shape == (3, 8, 1)
    assert (
        cast_a.indices
        == torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [2, 5, 7, 2, 5, 7, 2, 5, 7],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    ).all()
    assert (
        cast_b.indices
        == torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [2, 5, 7, 2, 5, 7, 2, 5, 7],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    ).all()

    a = SparseOpsMixin(
        torch.tensor([[0, 0, 0], [0, 1, 2], [0, 0, 0]], dtype=torch.long),
        shape=(1, 3, 1),
    )
    b = SparseOpsMixin(
        torch.tensor([[0, 0, 0], [0, 0, 0], [2, 5, 7]], dtype=torch.long),
        shape=(1, 1, 8),
    )
    cast_a, cast_b = SparseOpsMixin._cast_sparse_tensors([a, b])

    assert cast_a.shape == cast_b.shape == (1, 3, 8)
    assert (
        cast_a.indices
        == torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [2, 5, 7, 2, 5, 7, 2, 5, 7],
            ]
        )
    ).all()
    assert (
        cast_b.indices
        == torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [2, 5, 7, 2, 5, 7, 2, 5, 7],
            ]
        )
    ).all()


@assert_no_out_arr
def test_ops_generic_ops():
    indices = MockTensor((3, 32), dtype=torch.long)

    tensor1 = SparseOpsMixin(indices, shape=(16, 16, 16))
    tensor2 = SparseOpsMixin(indices, shape=(16, 16, 16))
    tensor3 = SparseOpsMixin(indices, shape=(16, 16, 16))

    with mock.patch(
        "sparse.ops.SparseOpsMixin._is_shared_indices",
        mock.MagicMock(return_value=True),
    ) as is_shared_mock, mock.patch(
        "sparse.ops.SparseOpsMixin._generic_shared_idx_ops",
        mock.MagicMock(return_value=tensor3),
    ) as generic_shared_mock:
        tensors = [tensor1, tensor2]
        result = SparseOpsMixin._generic_ops(tensors, None, "ops")

        is_shared_mock.assert_called_once_with(tensors)
        generic_shared_mock.assert_called_once_with(tensors, "ops")
        assert id(result) == id(tensor3)


@assert_no_out_arr
def test_ops_is_shared_indices():
    indices1 = MockTensor((3, 32), dtype=torch.long)
    indices2 = MockTensor((3, 32), dtype=torch.long)

    tensor1 = SparseOpsMixin(indices1, shape=(16, 16, 16))
    tensor2 = SparseOpsMixin(indices2, shape=(16, 16, 16))

    assert SparseOpsMixin._is_shared_indices([tensor1])
    assert SparseOpsMixin._is_shared_indices([tensor1, tensor1])
    assert SparseOpsMixin._is_shared_indices([tensor1, tensor1, tensor1])
    assert SparseOpsMixin._is_shared_indices([tensor2])
    assert not SparseOpsMixin._is_shared_indices([tensor1, tensor2])
    assert not SparseOpsMixin._is_shared_indices([tensor1, tensor1, tensor2, tensor1])
    assert not SparseOpsMixin._is_shared_indices([tensor1, tensor1, tensor1, tensor2])


@assert_no_out_arr
def test_ops_generic_shared_idx_ops():
    indices = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]], dtype=torch.long)
    values1 = torch.tensor([1, 2, 3, 4])
    values2 = torch.tensor([5, 6, 7, 8])

    tensor1 = SparseOpsMixin(indices, values1)
    tensor2 = SparseOpsMixin(indices, values2)

    result = SparseOpsMixin._generic_shared_idx_ops([tensor1, tensor2])

    assert id(result.indices) == id(indices)
    assert (result._values == torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8]])).all()

    result = SparseOpsMixin._generic_shared_idx_ops(
        [tensor1, tensor2], lambda x: x[:, 0] + x[:, 1]
    )

    assert id(result.indices) == id(indices)
    assert (result._values == torch.tensor([6, 8, 10, 12])).all()
