# `D:\src\scipysrc\scipy\scipy\interpolate\__init__.py`

```
# 导入所有 _interpolate 模块中的内容，通常用于直接访问插值相关函数和类
from ._interpolate import *

# 导入 _fitpack_py 模块中的所有内容，这些内容提供了旧版 FITPACK 接口的实现
from ._fitpack_py import *

# 导入 _fitpack2 模块中的所有内容，引入了 FITPACK 的新接口
from ._fitpack2 import *
# 导入 Rbf 类，用于径向基函数插值
from ._rbf import Rbf

# 导入 _rbfinterp 命名空间中的所有内容
from ._rbfinterp import *

# 导入 _polyint 命名空间中的所有内容
from ._polyint import *

# 导入 _cubic 命名空间中的所有内容
from ._cubic import *

# 导入 _ndgriddata 命名空间中的所有内容
from ._ndgriddata import *

# 导入 _bsplines 命名空间中的所有内容
from ._bsplines import *

# 导入 _pade 命名空间中的所有内容
from ._pade import *

# 导入 _rgi 命名空间中的所有内容
from ._rgi import *

# 导入 NdBSpline 类，用于 N 维 B 样条插值
from ._ndbspline import NdBSpline

# 弃用的命名空间，将在 v2.0.0 版本中移除
# 导入 fitpack、fitpack2、interpolate、ndgriddata、polyint、rbf 命名空间
from . import fitpack, fitpack2, interpolate, ndgriddata, polyint, rbf

# 定义 __all__ 列表，包含所有非下划线开头的命名空间成员
__all__ = [s for s in dir() if not s.startswith('_')]

# 导入 PytestTester 类并创建 test 对象，用于运行测试
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
# 删除 PytestTester 类的引用，确保不再使用该名称
del PytestTester

# 向后兼容性声明：pchip 等同于 PchipInterpolator 类
pchip = PchipInterpolator
```