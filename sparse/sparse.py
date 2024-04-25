from .base import BaseSparse
from .shape import SparseShapeMixin
from .type import SparseTypeMixin
from .scatter import SparseScatterMixin
from .indexing import SparseIndexingMixin
from .cat import SparseCatMixin
from .mask import SparseMaskMixin
from .ops import SparseOpsMixin


class SparseTensor(
    SparseOpsMixin,
    SparseCatMixin,
    SparseIndexingMixin,
    SparseScatterMixin,
    SparseShapeMixin,
    SparseMaskMixin,
    SparseTypeMixin,
    BaseSparse,
):
    pass
