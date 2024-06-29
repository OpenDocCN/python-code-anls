# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_tripcolor.pyi`

```
# 导入绘图所需的模块和函数
from matplotlib.axes import Axes  # 导入 Axes 类，用于创建图表的坐标系
from matplotlib.collections import PolyCollection, TriMesh  # 导入 PolyCollection 和 TriMesh 类，用于绘制多边形集合和三角网格
from matplotlib.colors import Normalize, Colormap  # 导入 Normalize 和 Colormap 类，用于颜色映射和归一化
from matplotlib.tri._triangulation import Triangulation  # 导入 Triangulation 类，用于处理三角网格的数据结构

from numpy.typing import ArrayLike  # 导入 ArrayLike 类型提示，用于指定数组类似的类型

from typing import overload, Literal  # 导入 overload 和 Literal 类型提示，用于函数重载和指定字面值类型

@overload
def tripcolor(
    ax: Axes,
    triangulation: Triangulation,
    c: ArrayLike = ...,
    *,
    alpha: float = ...,
    norm: str | Normalize | None = ...,
    cmap: str | Colormap | None = ...,
    vmin: float | None = ...,
    vmax: float | None = ...,
    shading: Literal["flat"] = ...,
    facecolors: ArrayLike | None = ...,
    **kwargs
) -> PolyCollection: ...
# 函数签名重载：在给定的坐标系 ax 上，根据给定的三角网格 triangulation 和颜色数据 c，绘制一个填充的颜色图案。
# 返回 PolyCollection 对象，用于表示多边形集合。

@overload
def tripcolor(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    c: ArrayLike = ...,
    *,
    alpha: float = ...,
    norm: str | Normalize | None = ...,
    cmap: str | Colormap | None = ...,
    vmin: float | None = ...,
    vmax: float | None = ...,
    shading: Literal["flat"] = ...,
    facecolors: ArrayLike | None = ...,
    **kwargs
) -> PolyCollection: ...
# 函数签名重载：在给定的坐标系 ax 上，根据给定的坐标数组 x 和 y、颜色数据 c，绘制一个填充的颜色图案。
# 返回 PolyCollection 对象，用于表示多边形集合。

@overload
def tripcolor(
    ax: Axes,
    triangulation: Triangulation,
    c: ArrayLike = ...,
    *,
    alpha: float = ...,
    norm: str | Normalize | None = ...,
    cmap: str | Colormap | None = ...,
    vmin: float | None = ...,
    vmax: float | None = ...,
    shading: Literal["gouraud"],
    facecolors: ArrayLike | None = ...,
    **kwargs
) -> TriMesh: ...
# 函数签名重载：在给定的坐标系 ax 上，根据给定的三角网格 triangulation 和颜色数据 c，绘制一个平滑着色的三角网格图案。
# 返回 TriMesh 对象，用于表示三角网格。

@overload
def tripcolor(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    c: ArrayLike = ...,
    *,
    alpha: float = ...,
    norm: str | Normalize | None = ...,
    cmap: str | Colormap | None = ...,
    vmin: float | None = ...,
    vmax: float | None = ...,
    shading: Literal["gouraud"],
    facecolors: ArrayLike | None = ...,
    **kwargs
) -> TriMesh: ...
# 函数签名重载：在给定的坐标系 ax 上，根据给定的坐标数组 x 和 y、颜色数据 c，绘制一个平滑着色的三角网格图案。
# 返回 TriMesh 对象，用于表示三角网格。
```