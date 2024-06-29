# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\__init__.py`

```py
"""
Unstructured triangular grid functions.
"""

# 从内部模块中导入各种功能和类
from ._triangulation import Triangulation
from ._tricontour import TriContourSet, tricontour, tricontourf
from ._trifinder import TriFinder, TrapezoidMapTriFinder
from ._triinterpolate import (TriInterpolator, LinearTriInterpolator,
                              CubicTriInterpolator)
from ._tripcolor import tripcolor
from ._triplot import triplot
from ._trirefine import TriRefiner, UniformTriRefiner
from ._tritools import TriAnalyzer

# 定义导出的公共接口，即模块中的所有公共对象
__all__ = ["Triangulation",
           "TriContourSet", "tricontour", "tricontourf",
           "TriFinder", "TrapezoidMapTriFinder",
           "TriInterpolator", "LinearTriInterpolator", "CubicTriInterpolator",
           "tripcolor",
           "triplot",
           "TriRefiner", "UniformTriRefiner",
           "TriAnalyzer"]
```