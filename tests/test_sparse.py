from sparse import SparseTensor, __version__
from sparse.base import BaseSparse
from sparse.type import SparseTypeMixin
from sparse.mask import SparseMaskMixin
from sparse.shape import SparseShapeMixin
from sparse.scatter import SparseScatterMixin
from sparse.cat import SparseCatMixin
from sparse.ops import SparseOpsMixin

from .assert_sys import assert_no_out_arr


@assert_no_out_arr
def test_sparse_version():
    assert __version__ == "0.0.1"


@assert_no_out_arr
def test_sparse():
    assert issubclass(SparseTensor, BaseSparse)
    assert issubclass(SparseTensor, SparseTypeMixin)
    assert issubclass(SparseTensor, SparseMaskMixin)
    assert issubclass(SparseTensor, SparseShapeMixin)
    assert issubclass(SparseTensor, SparseScatterMixin)
    assert issubclass(SparseTensor, SparseCatMixin)
    assert issubclass(SparseTensor, SparseOpsMixin)
