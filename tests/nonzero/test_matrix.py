from tests.utils.assert_sys import assert_no_out_arr
from tests.nonzero.utils import assert_nonzero_ops


@assert_no_out_arr
def test_nonzero_matrix_diag():
    assert_nonzero_ops("diag", 3)


@assert_no_out_arr
def test_nonzero_matrix_det():
    assert_nonzero_ops("det")


@assert_no_out_arr
def test_nonzero_matrix_trace():
    assert_nonzero_ops("trace")
