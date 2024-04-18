from .base import BaseSparse
from .shape import SparseShapeMixin
from .type import SparseTypeMixin
from .scatter import SparseScatterMixin
from .cat import SparseCatMixin
from .prod import SparseProdMixin
from .mask import SparseMaskMixin


class SparseTensor(
    SparseProdMixin,
    SparseCatMixin,
    SparseScatterMixin,
    SparseShapeMixin,
    SparseMaskMixin,
    SparseTypeMixin,
    BaseSparse,
):
    pass
