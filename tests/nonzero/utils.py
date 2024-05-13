from unittest import mock

import torch
from sparse import nonzero, SparseTensor

from tests.utils.mock_tensor import MockTensor
from tests.utils.assert_sys import assert_no_out_arr


def assert_nonzero_ops(op_name: str, *args, **kwargs):
    indices = MockTensor((1, 16), dtype=torch.long)
    values = MockTensor((16,))
    values_result = MockTensor((16,))

    input = SparseTensor(indices, values, shape=(1024,))
    expected_result = SparseTensor(indices, values_result, shape=(1024,))

    input.create_shared = mock.MagicMock("create_shared", return_value=expected_result)
    setattr(values, op_name, mock.MagicMock(op_name, return_value=values_result))

    result = getattr(nonzero, op_name)(input, *args, **kwargs)

    assert id(result) == id(expected_result)
    getattr(values, op_name).assert_called_once_with(*args, **kwargs)
    input.create_shared.assert_called_once_with(values_result)
