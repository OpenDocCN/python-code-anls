# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\style.py`

```
from __future__ import annotations
# 导入必要的模块和类型定义
from collections.abc import (
    Collection,
    Iterator,
    Sequence,
)
import itertools
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)
import warnings

import matplotlib as mpl
import matplotlib.colors
import numpy as np

from pandas._typing import MatplotlibColor as Color
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like

import pandas.core.common as com

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

@overload
def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = ...,
    color_type: str = ...,
    *,
    color: dict[str, Color],
) -> dict[str, Color]: ...

@overload
def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = ...,
    color_type: str = ...,
    *,
    color: Color | Sequence[Color] | None = ...,
) -> list[Color]: ...

@overload
def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = ...,
    color_type: str = ...,
    *,
    color: dict[str, Color] | Color | Sequence[Color] | None = ...,
) -> dict[str, Color] | list[Color]: ...

def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = None,
    color_type: str = "default",
    *,
    color: dict[str, Color] | Color | Sequence[Color] | None = None,
) -> dict[str, Color] | list[Color]:
    """
    Get standard colors based on `colormap`, `color_type` or `color` inputs.

    Parameters
    ----------
    num_colors : int
        Minimum number of colors to be returned.
        Ignored if `color` is a dictionary.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap` are not None.
    color : dict or str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a dictionary, or a single color (single color string,
        or sequence of floats representing a single color),
        or a sequence of colors.

    Returns
    -------
    dict or list
        Standard colors. Can either be a mapping if `color` was a dictionary,
        or a list of colors with a length of `num_colors` or more.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    """
    if isinstance(color, dict):
        return color  # 如果 color 参数是字典，则直接返回该字典作为标准颜色

    # 调用 _derive_colors 函数获取颜色信息
    colors = _derive_colors(
        color=color,
        colormap=colormap,
        color_type=color_type,
        num_colors=num_colors,
    )

    # 返回循环处理后的颜色列表
    return list(_cycle_colors(colors, num_colors=num_colors))


def _derive_colors(
    *,
    color: Color | Collection[Color] | None,
    colormap: str | Colormap | None,
    color_type: str,
    num_colors: int,
) -> Collection[Color]:
    """
    Derive colors based on input parameters.

    Parameters
    ----------
    color : Color or Collection[Color] or None
        Color(s) to be used for deriving sequence of colors.
    colormap : str or :py:class:`matplotlib.colors.Colormap` or None
        Matplotlib colormap.
    color_type : str
        Type of colors to derive.
    num_colors : int
        Minimum number of colors to be returned.

    Returns
    -------
    Collection[Color]
        Derived colors.
    """
    # 实际的颜色推导过程
    # 略去具体推导过程的细节，根据输入参数生成合适的颜色集合
    pass

def _cycle_colors(colors: Collection[Color], num_colors: int) -> Iterator[Color]:
    """
    Cycle through colors to ensure enough colors are available.

    Parameters
    ----------
    colors : Collection[Color]
        Colors to cycle through.
    num_colors : int
        Minimum number of colors needed.

    Yields
    ------
    Iterator[Color]
        Yields a color iterator.
    """
    # 循环处理颜色以确保有足够的颜色可用
    # 这里返回一个颜色的迭代器，用于处理颜色循环
    pass
    num_colors: int,


注释：


# 定义一个变量 num_colors，类型为整数
# 从 `colormap`, `color_type` 或 `color` 输入中获取颜色列表

def get_colors(
    color: str | sequence, optional
    Color(s) to be used for deriving sequence of colors.
    Can be either be a single color (single color string, or sequence of floats
    representing a single color), or a sequence of colors.
    
    colormap: matplotlib.colors.Colormap, optional
    Matplotlib colormap.
    When provided, the resulting colors will be derived from the colormap.
    
    color_type: {"default", "random"}, optional
    Type of colors to derive. Used if provided `color` and `colormap` are None.
    Ignored if either `color` or `colormap`` are not None.
    
    num_colors: int
    Number of colors to be extracted.
    
    Returns
    -------
    list
    List of colors extracted.
    
    Warns
    -----
    UserWarning
    If both `colormap` and `color` are provided.
    Parameter `color` will override.
"""
if color is None and colormap is not None:
    return _get_colors_from_colormap(colormap, num_colors=num_colors)
elif color is not None:
    if colormap is not None:
        warnings.warn(
            "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
            stacklevel=find_stack_level(),
        )
    return _get_colors_from_color(color)
else:
    return _get_colors_from_color_type(color_type, num_colors=num_colors)


def _cycle_colors(colors: list[Color], num_colors: int) -> Iterator[Color]:
    """循环使用颜色，直至达到 `num_colors` 的最大值或颜色列表的长度。

    如果颜色超出所需数量，matplotlib 将会忽略多余的颜色，此处不需处理。
    """
    max_colors = max(num_colors, len(colors))
    yield from itertools.islice(itertools.cycle(colors), max_colors)


def _get_colors_from_colormap(
    colormap: str | Colormap,
    num_colors: int,
) -> list[Color]:
    """从 colormap 中获取颜色。"""
    cmap = _get_cmap_instance(colormap)
    return [cmap(num) for num in np.linspace(0, 1, num=num_colors)]


def _get_cmap_instance(colormap: str | Colormap) -> Colormap:
    """获取 matplotlib colormap 的实例。"""
    if isinstance(colormap, str):
        cmap = colormap
        colormap = mpl.colormaps[colormap]
        if colormap is None:
            raise ValueError(f"Colormap {cmap} is not recognized")
    return colormap


def _get_colors_from_color(
    color: Color | Collection[Color],
) -> list[Color]:
    """从用户输入的颜色获取颜色列表。"""
    if len(color) == 0:
        raise ValueError(f"Invalid color argument: {color}")

    if _is_single_color(color):
        color = cast(Color, color)
        return [color]

    color = cast(Collection[Color], color)
    # 调用名为 _gen_list_of_colors_from_iterable 的函数，将 color 参数作为可迭代对象传入，并将结果转换为列表后返回
    return list(_gen_list_of_colors_from_iterable(color))
# 检查颜色是否为单一颜色，而不是颜色序列
# 单一颜色可以是以下类型之一：
#   - 命名颜色如 "red", "C0", "firebrick"
#   - 别名如 "g"
#   - 一组浮点数，例如 (0.1, 0.2, 0.3) 或 (0.1, 0.2, 0.3, 0.4)
def _is_single_color(color: Color | Collection[Color]) -> bool:
    if isinstance(color, str) and _is_single_string_color(color):
        # GH #36972
        return True
    
    if _is_floats_color(color):
        return True
    
    return False


# 从可迭代对象中生成颜色列表的生成器
def _gen_list_of_colors_from_iterable(color: Collection[Color]) -> Iterator[Color]:
    for x in color:
        if _is_single_color(x):
            yield x
        else:
            raise ValueError(f"Invalid color {x}")


# 检查颜色是否由表示颜色的浮点数序列组成
def _is_floats_color(color: Color | Collection[Color]) -> bool:
    return bool(
        is_list_like(color)
        and (len(color) == 3 or len(color) == 4)
        and all(isinstance(x, (int, float)) for x in color)
    )


# 从用户输入的颜色类型中获取颜色列表
def _get_colors_from_color_type(color_type: str, num_colors: int) -> list[Color]:
    if color_type == "default":
        return _get_default_colors(num_colors)
    elif color_type == "random":
        return _get_random_colors(num_colors)
    else:
        raise ValueError("color_type must be either 'default' or 'random'")


# 从 matplotlib 的 rc 参数中获取默认颜色列表的前 `num_colors` 个颜色
def _get_default_colors(num_colors: int) -> list[Color]:
    colors = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
    return colors[0:num_colors]


# 获取 `num_colors` 个随机颜色的列表
def _get_random_colors(num_colors: int) -> list[Color]:
    return [_random_color(num) for num in range(num_colors)]


# 获取用于表示随机颜色的列表，长度为 3 的随机浮点数列表
def _random_color(column: int) -> list[float]:
    # GH17525 use common._random_state to avoid resetting the seed
    rs = com.random_state(column)
    return rs.rand(3).tolist()


# 检查颜色是否为单一字符串颜色
# 单一字符串颜色的示例包括：
#   - 'r'
#   - 'g'
#   - 'red'
#   - 'green'
#   - 'C3'
#   - 'firebrick'
def _is_single_string_color(color: Color) -> bool:
    conv = matplotlib.colors.ColorConverter()
    try:
        conv.to_rgba(color)  # type: ignore[arg-type]
    except ValueError:
        return False
    else:
        return True
```