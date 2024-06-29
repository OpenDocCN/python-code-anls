# `D:\src\scipysrc\matplotlib\lib\matplotlib\streamplot.pyi`

```py
# 从 matplotlib 库中导入需要使用的类和函数
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, Colormap
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import ArrowStyle
from matplotlib.transforms import Transform

# 导入类型提示相关的模块和类
from typing import Literal
from numpy.typing import ArrayLike
from .typing import ColorType

# 定义 streamplot 函数，用于绘制流场图
def streamplot(
    axes: Axes,
    x: ArrayLike,
    y: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    density: float | tuple[float, float] = ...,
    linewidth: float | ArrayLike | None = ...,
    color: ColorType | ArrayLike | None = ...,
    cmap: str | Colormap | None = ...,
    norm: str | Normalize | None = ...,
    arrowsize: float = ...,
    arrowstyle: str | ArrowStyle = ...,
    minlength: float = ...,
    transform: Transform | None = ...,
    zorder: float | None = ...,
    start_points: ArrayLike | None = ...,
    maxlength: float = ...,
    integration_direction: Literal["forward", "backward", "both"] = ...,
    broken_streamlines: bool = ...
) -> StreamplotSet:
    ...

# 定义 StreamplotSet 类，包含流场图的线段和箭头集合
class StreamplotSet:
    lines: LineCollection  # 用于存储流场图的线段集合
    arrows: PatchCollection  # 用于存储流场图的箭头集合

    def __init__(self, lines: LineCollection, arrows: PatchCollection) -> None:
        ...

# 定义 DomainMap 类，用于管理网格和流场的映射关系
class DomainMap:
    grid: Grid  # 网格对象
    mask: StreamMask  # 流场掩码对象
    x_grid2mask: float  # 网格到掩码的 x 方向缩放比例
    y_grid2mask: float  # 网格到掩码的 y 方向缩放比例
    x_mask2grid: float  # 掩码到网格的 x 方向缩放比例
    y_mask2grid: float  # 掩码到网格的 y 方向缩放比例
    x_data2grid: float  # 数据到网格的 x 方向缩放比例
    y_data2grid: float  # 数据到网格的 y 方向缩放比例

    def __init__(self, grid: Grid, mask: StreamMask) -> None:
        ...
    
    def grid2mask(self, xi: float, yi: float) -> tuple[int, int]:
        ...
    
    def mask2grid(self, xm: float, ym: float) -> tuple[float, float]:
        ...
    
    def data2grid(self, xd: float, yd: float) -> tuple[float, float]:
        ...
    
    def grid2data(self, xg: float, yg: float) -> tuple[float, float]:
        ...
    
    def start_trajectory(self, xg: float, yg: float, broken_streamlines: bool = ...) -> None:
        ...
    
    def reset_start_point(self, xg: float, yg: float) -> None:
        ...
    
    def update_trajectory(self, xg, yg, broken_streamlines: bool = ...) -> None:
        ...
    
    def undo_trajectory(self) -> None:
        ...

# 定义 Grid 类，描述流场图的网格信息
class Grid:
    nx: int  # 网格 x 方向的数量
    ny: int  # 网格 y 方向的数量
    dx: float  # 网格 x 方向的间距
    dy: float  # 网格 y 方向的间距
    x_origin: float  # 网格起点的 x 坐标
    y_origin: float  # 网格起点的 y 坐标
    width: float  # 网格的宽度
    height: float  # 网格的高度

    def __init__(self, x: ArrayLike, y: ArrayLike) -> None:
        ...
    
    @property
    def shape(self) -> tuple[int, int]:
        ...
    
    def within_grid(self, xi: float, yi: float) -> bool:
        ...

# 定义 StreamMask 类，用于描述流场的掩码
class StreamMask:
    nx: int  # 掩码 x 方向的数量
    ny: int  # 掩码 y 方向的数量
    shape: tuple[int, int]  # 掩码的形状信息

    def __init__(self, density: float | tuple[float, float]) -> None:
        ...
    
    def __getitem__(self, args):
        ...

# 定义异常类 InvalidIndexError，用于表示索引错误
class InvalidIndexError(Exception):
    ...

# 定义异常类 TerminateTrajectory，用于表示终止轨迹的异常
class TerminateTrajectory(Exception):
    ...

# 定义异常类 OutOfBounds，用于表示超出边界的异常
class OutOfBounds(IndexError):
    ...
```