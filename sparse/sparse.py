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
    SparseShapeMixin,
    SparseScatterMixin,
    SparseMaskMixin,
    SparseTypeMixin,
    BaseSparse,
):
    pass
