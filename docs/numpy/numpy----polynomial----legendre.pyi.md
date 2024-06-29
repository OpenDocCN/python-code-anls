# `D:\src\scipysrc\numpy\numpy\polynomial\legendre.pyi`

```
# 引入类型提示中的 Any 类型
from typing import Any

# 引入整数类型 NDArray 以及 int_ 别名
from numpy import int_
# 引入 NDArray 类型别名
from numpy.typing import NDArray
# 引入 ABCPolyBase 类
from numpy.polynomial._polybase import ABCPolyBase
# 引入 trimcoef 函数
from numpy.polynomial.polyutils import trimcoef

# 定义 __all__ 列表，用于模块的导出
__all__: list[str]

# 将 trimcoef 函数赋值给 legtrim 变量
legtrim = trimcoef

# 定义 poly2leg 函数，用于将多项式转换为勒让德多项式
def poly2leg(pol): ...

# 定义 leg2poly 函数，用于将勒让德多项式转换为多项式
def leg2poly(c): ...

# 定义数组 legdomain，存储勒让德多项式的定义域
legdomain: NDArray[int_]
# 定义数组 legzero，存储勒让德多项式的零多项式
legzero: NDArray[int_]
# 定义数组 legone，存储勒让德多项式的单位多项式
legone: NDArray[int_]
# 定义数组 legx，存储勒让德多项式的 x 多项式
legx: NDArray[int_]

# 定义 legline 函数，生成从 off 到 scl 的勒让德多项式
def legline(off, scl): ...

# 定义 legfromroots 函数，根据给定的根生成勒让德多项式
def legfromroots(roots): ...

# 定义 legadd 函数，实现勒让德多项式的加法
def legadd(c1, c2): ...

# 定义 legsub 函数，实现勒让德多项式的减法
def legsub(c1, c2): ...

# 定义 legmulx 函数，实现勒让德多项式与 x 的乘法
def legmulx(c): ...

# 定义 legmul 函数，实现勒让德多项式的乘法
def legmul(c1, c2): ...

# 定义 legdiv 函数，实现勒让德多项式的除法
def legdiv(c1, c2): ...

# 定义 legpow 函数，实现勒让德多项式的幂次运算
def legpow(c, pow, maxpower=...): ...

# 定义 legder 函数，实现勒让德多项式的导数计算
def legder(c, m=..., scl=..., axis=...): ...

# 定义 legint 函数，实现勒让德多项式的积分计算
def legint(c, m=..., k = ..., lbnd=..., scl=..., axis=...): ...

# 定义 legval 函数，用于计算勒让德多项式在给定点 x 处的值
def legval(x, c, tensor=...): ...

# 定义 legval2d 函数，用于计算二维勒让德多项式在给定点 (x, y) 处的值
def legval2d(x, y, c): ...

# 定义 leggrid2d 函数，用于计算二维勒让德多项式在网格点 (x, y) 上的值
def leggrid2d(x, y, c): ...

# 定义 legval3d 函数，用于计算三维勒让德多项式在给定点 (x, y, z) 处的值
def legval3d(x, y, z, c): ...

# 定义 leggrid3d 函数，用于计算三维勒让德多项式在网格点 (x, y, z) 上的值
def leggrid3d(x, y, z, c): ...

# 定义 legvander 函数，用于计算 Vandermonde 矩阵，生成勒让德多项式的基函数
def legvander(x, deg): ...

# 定义 legvander2d 函数，用于计算二维 Vandermonde 矩阵，生成二维勒让德多项式的基函数
def legvander2d(x, y, deg): ...

# 定义 legvander3d 函数，用于计算三维 Vandermonde 矩阵，生成三维勒让德多项式的基函数
def legvander3d(x, y, z, deg): ...

# 定义 legfit 函数，用于拟合勒让德多项式到给定的数据点 (x, y)
def legfit(x, y, deg, rcond=..., full=..., w=...): ...

# 定义 legcompanion 函数，用于生成勒让德多项式的伴随矩阵
def legcompanion(c): ...

# 定义 legroots 函数，用于计算勒让德多项式的根
def legroots(c): ...

# 定义 leggauss 函数，用于计算勒让德多项式的 Gauss 积分点和权重
def leggauss(deg): ...

# 定义 legweight 函数，用于计算勒让德多项式在给定点上的权重
def legweight(x): ...

# 定义 Legendre 类，继承 ABCPolyBase 基类，表示勒让德多项式
class Legendre(ABCPolyBase):
    # domain 属性，表示勒让德多项式的定义域
    domain: Any
    # window 属性，表示勒让德多项式的窗口
    window: Any
    # basis_name 属性，表示勒让德多项式的基函数名称
    basis_name: Any
```