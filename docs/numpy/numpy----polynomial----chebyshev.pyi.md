# `D:\src\scipysrc\numpy\numpy\polynomial\chebyshev.pyi`

```py
# 导入必要的类型引用
from typing import Any
# 导入必要的函数和类
from numpy import int_
from numpy.typing import NDArray
from numpy.polynomial._polybase import ABCPolyBase
from numpy.polynomial.polyutils import trimcoef

# 指定在模块中公开的对象列表
__all__: list[str]

# 将 trimcoef 函数赋值给 chebtrim 变量
chebtrim = trimcoef

# 定义 poly2cheb 函数，将多项式转换为切比雪夫多项式
def poly2cheb(pol): ...

# 定义 cheb2poly 函数，将切比雪夫多项式转换为多项式
def cheb2poly(c): ...

# 定义切比雪夫多项式的定义域、零多项式和单位多项式的数组
chebdomain: NDArray[int_]
chebzero: NDArray[int_]
chebone: NDArray[int_]
chebx: NDArray[int_]

# 定义生成线性切比雪夫多项式的函数
def chebline(off, scl): ...

# 定义根据给定根生成切比雪夫多项式的函数
def chebfromroots(roots): ...

# 定义切比雪夫多项式加法运算
def chebadd(c1, c2): ...

# 定义切比雪夫多项式减法运算
def chebsub(c1, c2): ...

# 定义切比雪夫多项式乘以 x 的函数
def chebmulx(c): ...

# 定义切比雪夫多项式乘法运算
def chebmul(c1, c2): ...

# 定义切比雪夫多项式除法运算
def chebdiv(c1, c2): ...

# 定义切比雪夫多项式的幂运算
def chebpow(c, pow, maxpower=...): ...

# 定义切比雪夫多项式的导数计算
def chebder(c, m=..., scl=..., axis=...): ...

# 定义切比雪夫多项式的积分计算
def chebint(c, m=..., k = ..., lbnd=..., scl=..., axis=...): ...

# 定义对切比雪夫多项式进行求值的函数
def chebval(x, c, tensor=...): ...

# 定义在二维平面上对切比雪夫多项式进行求值的函数
def chebval2d(x, y, c): ...

# 定义在二维平面上生成切比雪夫网格的函数
def chebgrid2d(x, y, c): ...

# 定义在三维空间上对切比雪夫多项式进行求值的函数
def chebval3d(x, y, z, c): ...

# 定义在三维空间上生成切比雪夫网格的函数
def chebgrid3d(x, y, z, c): ...

# 定义生成切比雪夫范德蒙德矩阵的函数
def chebvander(x, deg): ...

# 定义在二维平面上生成切比雪夫范德蒙德矩阵的函数
def chebvander2d(x, y, deg): ...

# 定义在三维空间上生成切比雪夫范德蒙德矩阵的函数
def chebvander3d(x, y, z, deg): ...

# 定义使用最小二乘法拟合数据生成切比雪夫多项式的函数
def chebfit(x, y, deg, rcond=..., full=..., w=...): ...

# 定义生成切比雪夫多项式的伴随矩阵的函数
def chebcompanion(c): ...

# 定义计算切比雪夫多项式的根的函数
def chebroots(c): ...

# 定义使用插值法生成切比雪夫多项式的函数
def chebinterpolate(func, deg, args = ...): ...

# 定义生成切比雪夫-高斯积分节点和权重的函数
def chebgauss(deg): ...

# 定义计算切比雪夫多项式在给定点的权重的函数
def chebweight(x): ...

# 定义生成一维切比雪夫积分节点的函数
def chebpts1(npts): ...

# 定义生成二维切比雪夫积分节点的函数
def chebpts2(npts): ...

# 定义一个继承自 ABCPolyBase 类的切比雪夫多项式类
class Chebyshev(ABCPolyBase):
    # 根据函数插值生成切比雪夫多项式的类方法
    @classmethod
    def interpolate(cls, func, deg, domain=..., args = ...): ...
    
    # 切比雪夫多项式的定义域属性
    domain: Any
    # 切比雪夫多项式的窗口属性
    window: Any
    # 切比雪夫多项式的基函数名称属性
    basis_name: Any
```