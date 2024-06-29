# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_triplot.pyi`

```py
# 导入绘图所需的类和函数
from matplotlib.tri._triangulation import Triangulation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# 导入函数重载相关的类型注解
from typing import overload
from numpy.typing import ArrayLike

# 函数重载：接受 Triangulation 对象参数的 triplot 函数定义
@overload
def triplot(
    ax: Axes, triangulation: Triangulation, *args, **kwargs
) -> tuple[Line2D, Line2D]: ...

# 函数重载：接受数组参数的 triplot 函数定义
@overload
def triplot(
    ax: Axes, x: ArrayLike, y: ArrayLike, triangles: ArrayLike = ..., *args, **kwargs
) -> tuple[Line2D, Line2D]: ...
```