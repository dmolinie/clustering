""" Distances, Linkages, Kernels, Statistics & Characterizing Metrics """

from . import (
    _distances, _volumes, _kernels, _linkages, _scaling, _statistics, _quantifiers)

from ._distances import *
from ._volumes import *
from ._kernels import *
from ._linkages import *
from ._scaling import *
from ._statistics import *
from ._quantifiers import *

__all__ = _distances.__all__.copy()
__all__ += _volumes.__all__.copy()
__all__ += _kernels.__all__.copy()
__all__ += _linkages.__all__.copy()
__all__ += _scaling.__all__.copy()
__all__ += _statistics.__all__.copy()
__all__ += _quantifiers.__all__.copy()
