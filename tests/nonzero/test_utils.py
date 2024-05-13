from tests.utils.assert_sys import assert_no_out_arr
from tests.nonzero.utils import assert_nonzero_ops


@assert_no_out_arr
def test_nonzero_utils_view():
    assert_nonzero_ops("view", 2, 6, 8)


@assert_no_out_arr
def test_nonzero_utils_reshape():
    assert_nonzero_ops("reshape", 2, 6, 8)


@assert_no_out_arr
def test_nonzero_utils_clip():
    assert_nonzero_ops("clip", min=5, max=7)


@assert_no_out_arr
def test_nonzero_utils_clamp():
    assert_nonzero_ops("clamp", min=3, max=6)
