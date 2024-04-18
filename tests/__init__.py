import pytest

pytestmark = pytest.mark.filterwarnings("error")

from .test_base import *
from .test_type import *
from .test_mask import *
from .test_shape import *
