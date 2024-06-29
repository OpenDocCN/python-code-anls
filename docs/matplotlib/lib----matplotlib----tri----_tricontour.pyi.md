# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_tricontour.pyi`

```py
# 从 matplotlib 库中导入 Axes 类，用于绘制图形的坐标轴
from matplotlib.axes import Axes
# 从 matplotlib.contour 模块中导入 ContourSet 类，用于绘制等高线图
from matplotlib.contour import ContourSet
# 从 matplotlib.tri._triangulation 模块中导入 Triangulation 类，用于处理三角剖分

# 从 numpy.typing 模块中导入 ArrayLike 类型，用于表示类似数组的对象
from numpy.typing import ArrayLike
# 从 typing 模块中导入 overload 装饰器，用于定义函数重载

# TODO: 是否需要更明确的参数和关键字参数（对于本模块中的所有内容）？

# 定义 TriContourSet 类，继承自 ContourSet 类，用于处理三角剖分的等高线集合
class TriContourSet(ContourSet):
    # 构造函数，初始化 TriContourSet 对象
    def __init__(self, ax: Axes, *args, **kwargs) -> None: ...

# 以下是函数重载定义，用于绘制三角剖分的等高线或填充等高线图

# 函数重载：根据三角剖分对象绘制等高线图
@overload
def tricontour(
    ax: Axes,
    triangulation: Triangulation,
    z: ArrayLike,
    levels: int | ArrayLike = ...,
    **kwargs
) -> TriContourSet: ...

# 函数重载：根据 x, y, z 数据绘制三角剖分的等高线图
@overload
def tricontour(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    levels: int | ArrayLike = ...,
    *,
    triangles: ArrayLike = ...,
    mask: ArrayLike = ...,
    **kwargs
) -> TriContourSet: ...

# 函数重载：根据三角剖分对象绘制填充等高线图
@overload
def tricontourf(
    ax: Axes,
    triangulation: Triangulation,
    z: ArrayLike,
    levels: int | ArrayLike = ...,
    **kwargs
) -> TriContourSet: ...

# 函数重载：根据 x, y, z 数据绘制三角剖分的填充等高线图
@overload
def tricontourf(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    levels: int | ArrayLike = ...,
    *,
    triangles: ArrayLike = ...,
    mask: ArrayLike = ...,
    **kwargs
) -> TriContourSet: ...
```