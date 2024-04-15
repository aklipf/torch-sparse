from __future__ import annotations
from typing import Iterable, List, Tuple, Literal, Any

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from .base import BaseSparse
from .shape import SparseShapeMixin
from .type import SparseTypeMixin
from .scatter import SparseScatterMixin
from .cat import SparseCatMixin


class SparseTensor(
    SparseCatMixin, SparseShapeMixin, SparseTypeMixin, SparseScatterMixin, BaseSparse
):
    pass
