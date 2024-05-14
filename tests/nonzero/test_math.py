from tests.utils.assert_sys import assert_no_out_arr
from tests.nonzero.utils import assert_nonzero_ops


@assert_no_out_arr
def test_nonzero_math_pow():
    assert_nonzero_ops("pow", 3)


@assert_no_out_arr
def test_nonzero_math_exp():
    assert_nonzero_ops("exp")


@assert_no_out_arr
def test_nonzero_math_exp2():
    assert_nonzero_ops("exp2")


@assert_no_out_arr
def test_nonzero_math_log():
    assert_nonzero_ops("log")


@assert_no_out_arr
def test_nonzero_math_log2():
    assert_nonzero_ops("log2")


@assert_no_out_arr
def test_nonzero_math_log10():
    assert_nonzero_ops("log10")


@assert_no_out_arr
def test_nonzero_math_sqrt():
    assert_nonzero_ops("sqrt")


@assert_no_out_arr
def test_nonzero_math_rsqrt():
    assert_nonzero_ops("rsqrt")
