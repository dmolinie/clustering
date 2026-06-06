""" Self-Parameterized Recursively Assessed Decomposition Algorithm (SPRADA) """

from . import _recursive, _sprada
from ._recursive import *
from ._sprada import *

__all__ = _recursive.__all__.copy()
__all__ += _sprada.__all__.copy()
