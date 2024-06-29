# `D:\src\scipysrc\matplotlib\lib\matplotlib\typing.py`

```py
"""
Typing support for Matplotlib

This module contains Type aliases which are useful for Matplotlib and potentially
downstream libraries.

.. admonition:: Provisional status of typing

    The ``typing`` module and type stub files are considered provisional and may change
    at any time without a deprecation period.
"""

# 导入必要的模块和类型定义
from collections.abc import Hashable, Sequence  # 导入 Hashable 和 Sequence 抽象基类
import pathlib  # 导入 pathlib 模块
from typing import Any, Literal, TypeVar, Union  # 导入类型定义 Any, Literal, TypeVar, Union

# 从相对路径导入指定模块和对象
from . import path  # 从当前包的 path 模块导入
from ._enums import JoinStyle, CapStyle  # 从当前包的 _enums 模块导入 JoinStyle 和 CapStyle 枚举类型
from .markers import MarkerStyle  # 从当前包的 markers 模块导入 MarkerStyle 类

# 下面是类型别名的定义。一旦 Python 3.9 被废弃，应使用 typing.TypeAlias 进行注释，
# 并将 Union 转换为使用 | 语法。

# RGB 颜色类型别名
RGBColorType = Union[tuple[float, float, float], str]
# RGBA 颜色类型别名
RGBAColorType = Union[
    str,  # "none" 或 "#RRGGBBAA"/"#RGBA" 十六进制字符串
    tuple[float, float, float, float],  # 4 元组表示颜色和透明度
    tuple[RGBColorType, float],  # (颜色, 透明度) 表示
    tuple[tuple[float, float, float, float], float],  # 奇怪的 (4 元组, 浮点数) 表示
]
# 颜色类型别名
ColorType = Union[RGBColorType, RGBAColorType]

# RGB 颜色类型别名（英式拼写）
RGBColourType = RGBColorType
# RGBA 颜色类型别名（英式拼写）
RGBAColourType = RGBAColorType
# 颜色类型别名（英式拼写）
ColourType = ColorType

# 线条样式类型别名
LineStyleType = Union[str, tuple[float, Sequence[float]]]
# 绘制样式类型别名
DrawStyleType = Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
# 每个标记类型别名
MarkEveryType = Union[
    None, int, tuple[int, int], slice, list[int], float, tuple[float, float], list[bool]
]

# 标记类型别名
MarkerType = Union[str, path.Path, MarkerStyle]
# 填充样式类型别名
FillStyleType = Literal["full", "left", "right", "bottom", "top", "none"]
# 连接样式类型别名
JoinStyleType = Union[JoinStyle, Literal["miter", "round", "bevel"]]
# 端点样式类型别名
CapStyleType = Union[CapStyle, Literal["butt", "projecting", "round"]]

# 风格类型别名
RcStyleType = Union[
    str,
    dict[str, Any],
    pathlib.Path,
    Sequence[Union[str, pathlib.Path, dict[str, Any]]],
]

# 哈希列表类型别名，包含哈希值的嵌套列表
_HT = TypeVar("_HT", bound=Hashable)
HashableList = list[Union[_HT, "HashableList[_HT]"]]
"""A nested list of Hashable values."""
```