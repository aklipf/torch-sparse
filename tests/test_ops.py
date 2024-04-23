from unittest import mock

import torch

from sparse.ops import SparseOpsMixin, _intersection_mask, _union_mask

from .mock_tensor import MockTensor
from .assert_sys import assert_no_out_arr

a_indices = torch.tensor(
    [[0, 3, 1, 1, 2, 2, 3], [0, 0, 1, 2, 1, 2, 3]], dtype=torch.long
)
a_values = torch.tensor([[1], [5], [1], [1], [1], [1], [1]], dtype=torch.int32)
b_indices = torch.tensor([[0, 1, 1, 2, 3], [0, 1, 2, 2, 3]], dtype=torch.long)
b_values = torch.tensor([[1], [2], [1], [1], [1]], dtype=torch.int32)
c_indices = torch.tensor([[0, 1, 1, 2, 2], [0, 1, 2, 2, 3]], dtype=torch.long)
c_values = torch.tensor([[-2], [-2], [1], [1], [1]], dtype=torch.int32)

a = SparseOpsMixin(a_indices, a_values, shape=(4, 4))
a_bool = SparseOpsMixin(a_indices, shape=(4, 4))
b = SparseOpsMixin(b_indices, b_values, shape=(4, 4))
b_bool = SparseOpsMixin(b_indices, shape=(4, 4))
c = SparseOpsMixin(c_indices, c_values, shape=(4, 4))
c_bool = SparseOpsMixin(c_indices, shape=(4, 4))


@assert_no_out_arr
def test_ops_intersection_mask():
    mask = _intersection_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 0, 1]], dtype=torch.long
        ),
        2,
    )
    assert (
        mask == torch.tensor([True, False, False, False, True, False, False, False])
    ).all()

    mask = _intersection_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 1, 1]], dtype=torch.long
        ),
        2,
    )
    assert (
        mask == torch.tensor([True, False, False, False, True, False, True, False])
    ).all()


@assert_no_out_arr
def test_ops_union_mask():
    mask = _union_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 0, 1]], dtype=torch.long
        ),
        2,
    )

    assert (
        mask == torch.tensor([True, False, True, True, True, False, True, True])
    ).all()

    mask = _union_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 1, 1]], dtype=torch.long
        ),
        2,
    )

    assert (
        mask == torch.tensor([True, False, True, True, True, False, True, False])
    ).all()


@assert_no_out_arr
def test_ops_and():
    assert (
        (a_bool & b_bool & c_bool).to_dense()
        == (a_bool.to_dense() & b_bool.to_dense() & c_bool.to_dense())
    ).all()


@assert_no_out_arr
def test_ops_or():
    assert (
        (a_bool | b_bool | c_bool).to_dense()
        == (a_bool.to_dense() | b_bool.to_dense() | c_bool.to_dense())
    ).all()


@assert_no_out_arr
def test_ops_add():
    assert (
        (a + b + c).to_dense() == (a.to_dense() + b.to_dense() + c.to_dense())
    ).all()


@assert_no_out_arr
def test_ops_mul():
    assert (
        (a * b * c).to_dense() == (a.to_dense() * b.to_dense() * c.to_dense())
    ).all()


@assert_no_out_arr
def test_ops_sub():
    assert (
        (a - b - c).to_dense() == (a.to_dense() - b.to_dense() - c.to_dense())
    ).all()
