from .base import BaseSparse
from .shape import SparseShapeMixin
from .type import SparseTypeMixin
from .scatter import SparseScatterMixin
from .cat import SparseCatMixin
from .prod import SparseProdMixin
from .indexing import SparseIndexingMixin


class SparseTensor(
    SparseProdMixin,
    SparseCatMixin,
    SparseShapeMixin,
    SparseScatterMixin,
    SparseTypeMixin,
    SparseIndexingMixin,
    BaseSparse,
):
    pass
