# `D:\src\scipysrc\numpy\numpy\polynomial\polynomial.pyi`

```
# 从 typing 模块导入 Any 类型
from typing import Any

# 从 numpy 中导入 int_ 类型和 NDArray 类型
from numpy import int_
from numpy.typing import NDArray

# 从 numpy.polynomial._polybase 模块导入 ABCPolyBase 类
from numpy.polynomial._polybase import ABCPolyBase

# 从 numpy.polynomial.polyutils 模块导入 trimcoef 函数
from numpy.polynomial.polyutils import trimcoef

# 定义一个包含 __all__ 属性的列表，用于模块的公开接口
__all__: list[str]

# 将 trimcoef 函数赋值给 polytrim 变量，用于多项式的系数修剪
polytrim = trimcoef

# 定义以下四个变量，均为 int_ 类型的一维数组 NDArray[int_]
polydomain: NDArray[int_]
polyzero: NDArray[int_]
polyone: NDArray[int_]
polyx: NDArray[int_]

# 定义以下函数，用于多项式操作，具体实现略去，可能包括多项式创建、加减乘除、求导积分、值计算等
def polyline(off, scl): ...

def polyfromroots(roots): ...

def polyadd(c1, c2): ...

def polysub(c1, c2): ...

def polymulx(c): ...

def polymul(c1, c2): ...

def polydiv(c1, c2): ...

def polypow(c, pow, maxpower=...): ...

def polyder(c, m=..., scl=..., axis=...): ...

def polyint(c, m=..., k=..., lbnd=..., scl=..., axis=...): ...

def polyval(x, c, tensor=...): ...

def polyvalfromroots(x, r, tensor=...): ...

def polyval2d(x, y, c): ...

def polygrid2d(x, y, c): ...

def polyval3d(x, y, z, c): ...

def polygrid3d(x, y, z, c): ...

def polyvander(x, deg): ...

def polyvander2d(x, y, deg): ...

def polyvander3d(x, y, z, deg): ...

def polyfit(x, y, deg, rcond=..., full=..., w=...): ...

def polyroots(c): ...

# 定义一个继承自 ABCPolyBase 的 Polynomial 类，表示多项式
class Polynomial(ABCPolyBase):
    # domain、window 和 basis_name 是该类的成员变量，类型为 Any，用于描述多项式的领域、窗口和基础名称
    domain: Any
    window: Any
    basis_name: Any
```