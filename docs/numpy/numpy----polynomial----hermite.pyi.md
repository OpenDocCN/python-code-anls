# `.\numpy\numpy\polynomial\hermite.pyi`

```py
# 导入必要的类型注解
from typing import Any

# 导入所需的 numpy 类型
from numpy import int_, float64

# 导入 numpy 的类型注解
from numpy.typing import NDArray

# 导入多项式基类和系数修剪函数
from numpy.polynomial._polybase import ABCPolyBase
from numpy.polynomial.polyutils import trimcoef

# 声明 __all__ 变量，并初始化为空列表
__all__: list[str]

# 将 trimcoef 函数别名为 hermtrim
hermtrim = trimcoef

# 定义两个函数 poly2herm 和 herm2poly，但具体实现部分未显示，用 ... 表示
def poly2herm(pol): ...
def herm2poly(c): ...

# 声明一些全局变量，类型为 numpy 数组
hermdomain: NDArray[int_]
hermzero: NDArray[int_]
hermone: NDArray[int_]
hermx: NDArray[float64]

# 定义一系列 Hermite 多项式的操作函数
def hermline(off, scl): ...
def hermfromroots(roots): ...
def hermadd(c1, c2): ...
def hermsub(c1, c2): ...
def hermmulx(c): ...
def hermmul(c1, c2): ...
def hermdiv(c1, c2): ...
def hermpow(c, pow, maxpower=...): ...
def hermder(c, m=..., scl=..., axis=...): ...
def hermint(c, m=..., k = ..., lbnd=..., scl=..., axis=...): ...
def hermval(x, c, tensor=...): ...
def hermval2d(x, y, c): ...
def hermgrid2d(x, y, c): ...
def hermval3d(x, y, z, c): ...
def hermgrid3d(x, y, z, c): ...
def hermvander(x, deg): ...
def hermvander2d(x, y, deg): ...
def hermvander3d(x, y, z, deg): ...
def hermfit(x, y, deg, rcond=..., full=..., w=...): ...
def hermcompanion(c): ...
def hermroots(c): ...
def hermgauss(deg): ...
def hermweight(x): ...

# 定义 Hermite 类，继承自 ABCPolyBase 类
class Hermite(ABCPolyBase):
    domain: Any      # 定义 domain 属性
    window: Any      # 定义 window 属性
    basis_name: Any  # 定义 basis_name 属性
```