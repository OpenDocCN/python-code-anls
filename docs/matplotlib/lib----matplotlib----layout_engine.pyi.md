# `D:\src\scipysrc\matplotlib\lib\matplotlib\layout_engine.pyi`

```py
from matplotlib.figure import Figure  # 导入Figure类，用于创建和管理图形对象

from typing import Any  # 导入Any类型，用于灵活类型注解

class LayoutEngine:
    def __init__(self, **kwargs: Any) -> None: ...  # 初始化函数，接受任意关键字参数

    def set(self) -> None: ...  # 设置函数，用于设置布局引擎的配置

    @property
    def colorbar_gridspec(self) -> bool: ...  # 属性函数，返回布尔值，表示是否支持colorbar和gridspec

    @property
    def adjust_compatible(self) -> bool: ...  # 属性函数，返回布尔值，表示是否支持adjust_compatible特性

    def get(self) -> dict[str, Any]: ...  # 获取函数，返回一个字典，包含布局引擎的相关配置信息

    def execute(self, fig: Figure) -> None: ...  # 执行函数，接受一个Figure对象参数，用于执行布局操作

class PlaceHolderLayoutEngine(LayoutEngine):
    def __init__(
        self, adjust_compatible: bool, colorbar_gridspec: bool, **kwargs: Any
    ) -> None: ...  # 初始化函数，接受布尔型参数和任意关键字参数

    def execute(self, fig: Figure) -> None: ...  # 执行函数，重写父类方法，接受一个Figure对象参数

class TightLayoutEngine(LayoutEngine):
    def __init__(
        self,
        *,
        pad: float = ...,  # 浮点型参数，表示边界填充
        h_pad: float | None = ...,  # 可选的浮点型参数，表示水平边界填充
        w_pad: float | None = ...,  # 可选的浮点型参数，表示垂直边界填充
        rect: tuple[float, float, float, float] = ...,  # 四元组参数，表示矩形区域的位置和大小
        **kwargs: Any
    ) -> None: ...  # 初始化函数，接受多个命名参数和任意关键字参数

    def execute(self, fig: Figure) -> None: ...  # 执行函数，重写父类方法，接受一个Figure对象参数

    def set(
        self,
        *,
        pad: float | None = ...,  # 可选的浮点型参数，表示边界填充
        w_pad: float | None = ...,  # 可选的浮点型参数，表示水平边界填充
        h_pad: float | None = ...,  # 可选的浮点型参数，表示垂直边界填充
        rect: tuple[float, float, float, float] | None = ...  # 可选的四元组参数，表示矩形区域的位置和大小
    ) -> None: ...  # 设置函数，用于设置布局引擎的配置

class ConstrainedLayoutEngine(LayoutEngine):
    def __init__(
        self,
        *,
        h_pad: float | None = ...,  # 可选的浮点型参数，表示水平边界填充
        w_pad: float | None = ...,  # 可选的浮点型参数，表示垂直边界填充
        hspace: float | None = ...,  # 可选的浮点型参数，表示水平间距
        wspace: float | None = ...,  # 可选的浮点型参数，表示垂直间距
        rect: tuple[float, float, float, float] = ...,  # 四元组参数，表示矩形区域的位置和大小
        compress: bool = ...,  # 布尔型参数，表示是否压缩布局
        **kwargs: Any
    ) -> None: ...  # 初始化函数，接受多个命名参数和任意关键字参数

    def execute(self, fig: Figure) -> Any: ...  # 执行函数，接受一个Figure对象参数，返回任意类型

    def set(
        self,
        *,
        h_pad: float | None = ...,  # 可选的浮点型参数，表示水平边界填充
        w_pad: float | None = ...,  # 可选的浮点型参数，表示垂直边界填充
        hspace: float | None = ...,  # 可选的浮点型参数，表示水平间距
        wspace: float | None = ...,  # 可选的浮点型参数，表示垂直间距
        rect: tuple[float, float, float, float] | None = ...  # 可选的四元组参数，表示矩形区域的位置和大小
    ) -> None: ...  # 设置函数，用于设置布局引擎的配置
```