from unittest import mock

import torch

from sparse.ops import _intersection_mask, _union_mask

from .mock_tensor import MockTensor
from .assert_sys import assert_no_out_arr


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
    print(mask)
    assert (
        mask == torch.tensor([True, False, True, True, True, False, True, True])
    ).all()

    mask = _union_mask(
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 0, 0, 1, 1]], dtype=torch.long
        ),
        2,
    )
    print(mask)
    assert (
        mask == torch.tensor([True, False, True, True, True, False, True, False])
    ).all()
