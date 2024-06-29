# `D:\src\scipysrc\matplotlib\lib\matplotlib\container.pyi`

```
# 从 matplotlib.artist 模块导入 Artist 类
from matplotlib.artist import Artist
# 从 matplotlib.lines 模块导入 Line2D 类
from matplotlib.lines import Line2D
# 从 matplotlib.collections 模块导入 LineCollection 类
from matplotlib.collections import LineCollection
# 从 matplotlib.patches 模块导入 Rectangle 类
from matplotlib.patches import Rectangle

# 从 collections.abc 模块导入 Callable 类型
from collections.abc import Callable
# 从 typing 模块导入 Any 和 Literal 类型
from typing import Any, Literal
# 从 numpy.typing 模块导入 ArrayLike 类型
from numpy.typing import ArrayLike

# 定义一个继承自 tuple 的类 Container
class Container(tuple):
    # 定义 __new__ 方法，但没有具体实现
    def __new__(cls, *args, **kwargs): ...
    # 定义初始化方法 __init__
    def __init__(self, kl, label: Any | None = ...) -> None: ...
    # 定义 remove 方法，用于移除子元素
    def remove(self) -> None: ...
    # 定义 get_children 方法，返回子元素列表
    def get_children(self) -> list[Artist]: ...
    # 定义 get_label 方法，返回标签字符串或 None
    def get_label(self) -> str | None: ...
    # 定义 set_label 方法，设置标签字符串
    def set_label(self, s: Any) -> None: ...
    # 定义 add_callback 方法，添加回调函数
    def add_callback(self, func: Callable[[Artist], Any]) -> int: ...
    # 定义 remove_callback 方法，移除指定的回调函数
    def remove_callback(self, oid: int) -> None: ...
    # 定义 pchanged 方法，表示内容变化
    def pchanged(self) -> None: ...

# 定义一个继承自 Container 的类 BarContainer
class BarContainer(Container):
    # 定义 patches 属性，表示矩形列表
    patches: list[Rectangle]
    # 定义 errorbar 属性，表示误差线容器或 None
    errorbar: None | ErrorbarContainer
    # 定义 datavalues 属性，表示数据值或 None
    datavalues: None | ArrayLike
    # 定义 orientation 属性，表示方向或 None
    orientation: None | Literal["vertical", "horizontal"]
    # 定义初始化方法 __init__
    def __init__(
        self,
        patches: list[Rectangle],
        errorbar: ErrorbarContainer | None = ...,
        *,
        datavalues: ArrayLike | None = ...,
        orientation: Literal["vertical", "horizontal"] | None = ...,
        **kwargs
    ) -> None: ...

# 定义一个继承自 Container 的类 ErrorbarContainer
class ErrorbarContainer(Container):
    # 定义 lines 属性，表示线条元组
    lines: tuple[Line2D, tuple[Line2D, ...], tuple[LineCollection, ...]]
    # 定义 has_xerr 属性，表示是否有 x 方向误差
    has_xerr: bool
    # 定义 has_yerr 属性，表示是否有 y 方向误差
    has_yerr: bool
    # 定义初始化方法 __init__
    def __init__(
        self,
        lines: tuple[Line2D, tuple[Line2D, ...], tuple[LineCollection, ...]],
        has_xerr: bool = ...,
        has_yerr: bool = ...,
        **kwargs
    ) -> None: ...

# 定义一个继承自 Container 的类 StemContainer
class StemContainer(Container):
    # 定义 markerline 属性，表示标记线
    markerline: Line2D
    # 定义 stemlines 属性，表示茎线集合
    stemlines: LineCollection
    # 定义 baseline 属性，表示基线
    baseline: Line2D
    # 定义初始化方法 __init__
    def __init__(
        self,
        markerline_stemlines_baseline: tuple[Line2D, LineCollection, Line2D],
        **kwargs
    ) -> None: ...
```