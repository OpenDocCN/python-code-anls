# `.\pytorch\torch\_numpy\__init__.py`

```
# 忽略类型检查错误，适用于类型检查工具如mypy
# 从当前包导入fft, linalg, random模块
from . import fft, linalg, random
# 导入_dtypes模块中的所有内容，忽略F403错误（与import *相关的错误）
from ._dtypes import *  # noqa: F403
# 导入_funcs模块中的所有内容，忽略F403错误
from ._funcs import *  # noqa: F403
# 导入_getlimits模块中的finfo和iinfo对象
from ._getlimits import finfo, iinfo
# 导入_ndarray模块中的特定函数和类
from ._ndarray import (
    array,
    asarray,
    ascontiguousarray,
    can_cast,
    from_dlpack,
    ndarray,
    newaxis,
    result_type,
)
# 导入_ufuncs模块中的所有内容，忽略F403错误
from ._ufuncs import *  # noqa: F403
# 导入_util模块中的AxisError和UFuncTypeError异常类
from ._util import AxisError, UFuncTypeError

# 从math模块导入常数pi和e，忽略usort: skip注释
from math import pi, e  # usort: skip

# 将内置函数all赋值给alltrue，用于简化语法
alltrue = all
# 将内置函数any赋值给sometrue，用于简化语法
sometrue = any

# 定义inf为浮点数正无穷大
inf = float("inf")
# 定义nan为浮点数NaN（非数值）
nan = float("nan")

# 定义False_为布尔值False（可以避免与内置关键字False冲突）
False_ = False
# 定义True_为布尔值True（可以避免与内置关键字True冲突）
True_ = True
```