from tests.utils.assert_sys import assert_no_out_arr
from tests.nonzero.utils import assert_nonzero_ops


@assert_no_out_arr
def test_nonzero_trig_sin():
    assert_nonzero_ops("sin")


@assert_no_out_arr
def test_nonzero_trig_cos():
    assert_nonzero_ops("cos")


@assert_no_out_arr
def test_nonzero_trig_tan():
    assert_nonzero_ops("tan")


@assert_no_out_arr
def test_nonzero_trig_sinh():
    assert_nonzero_ops("sinh")


@assert_no_out_arr
def test_nonzero_trig_cosh():
    assert_nonzero_ops("cosh")


@assert_no_out_arr
def test_nonzero_trig_tanh():
    assert_nonzero_ops("tanh")


@assert_no_out_arr
def test_nonzero_trig_asin():
    assert_nonzero_ops("asin")


@assert_no_out_arr
def test_nonzero_trig_acos():
    assert_nonzero_ops("acos")


@assert_no_out_arr
def test_nonzero_trig_atan():
    assert_nonzero_ops("atan")


@assert_no_out_arr
def test_nonzero_trig_asinh():
    assert_nonzero_ops("asinh")


@assert_no_out_arr
def test_nonzero_trig_acosh():
    assert_nonzero_ops("acosh")


@assert_no_out_arr
def test_nonzero_trig_atanh():
    assert_nonzero_ops("atanh")
