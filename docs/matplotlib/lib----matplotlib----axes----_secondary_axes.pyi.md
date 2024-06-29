# `D:\src\scipysrc\matplotlib\lib\matplotlib\axes\_secondary_axes.pyi`

```py
from matplotlib.axes._base import _AxesBase
from matplotlib.axis import Tick

from matplotlib.transforms import Transform

from collections.abc import Callable, Iterable
from typing import Literal
from numpy.typing import ArrayLike
from matplotlib.typing import ColorType

class SecondaryAxis(_AxesBase):
    # 初始化函数，用于创建 SecondaryAxis 实例
    def __init__(
        self,
        parent: _AxesBase,  # 父级 AxesBase 对象，表示次要坐标轴所属的主要坐标轴
        orientation: Literal["x", "y"],  # 次要坐标轴的方向，可以是水平 ("x") 或垂直 ("y")
        location: Literal["top", "bottom", "right", "left"] | float,  # 次要坐标轴的位置，可以是边缘位置名称或浮点数值
        functions: tuple[  # 函数或变换对象，用于将数据转换到次要坐标轴上
            Callable[[ArrayLike], ArrayLike],  # 用于数据向次要坐标轴方向转换的函数
            Callable[[ArrayLike], ArrayLike]   # 用于数据从次要坐标轴方向转换回原始坐标轴方向的函数
        ] | Transform,  # 或者直接是变换对象 Transform
        transform: Transform | None = ...,
        **kwargs
    ) -> None: ...
    
    # 设置次要坐标轴的对齐方式
    def set_alignment(
        self, align: Literal["top", "bottom", "right", "left"]
    ) -> None: ...
    
    # 设置次要坐标轴的位置和相应的变换对象
    def set_location(
        self,
        location: Literal["top", "bottom", "right", "left"] | float,  # 次要坐标轴的位置或浮点数值
        transform: Transform | None = ...  # 可选的变换对象，用于数据转换
    ) -> None: ...
    
    # 设置次要坐标轴的刻度位置和标签
    def set_ticks(
        self,
        ticks: ArrayLike,  # 刻度的位置
        labels: Iterable[str] | None = ...,  # 刻度的标签，可选
        *,
        minor: bool = ...,  # 是否为次要刻度
        **kwargs
    ) -> list[Tick]: ...
    
    # 设置次要坐标轴的转换函数或变换对象
    def set_functions(
        self,
        functions: tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]] | Transform,
    ) -> None: ...
    
    # 设置次要坐标轴的纵横比
    def set_aspect(self, *args, **kwargs) -> None: ...
    
    # 设置次要坐标轴的颜色
    def set_color(self, color: ColorType) -> None: ...
```