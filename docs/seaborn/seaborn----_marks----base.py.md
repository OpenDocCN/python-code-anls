# `D:\src\scipysrc\seaborn\seaborn\_marks\base.py`

```
from __future__ import annotations
# 导入用于类型注解的特性

from dataclasses import dataclass, fields, field
# 导入用于数据类的装饰器和字段操作的工具

import textwrap
# 导入用于文本包装的模块

from typing import Any, Callable, Union
# 导入用于类型提示的工具，包括 Any、Callable 和 Union

from collections.abc import Generator
# 导入生成器抽象基类

import numpy as np
# 导入 NumPy 库，并使用 np 别名

import pandas as pd
# 导入 Pandas 库，并使用 pd 别名

import matplotlib as mpl
# 导入 Matplotlib 库，并使用 mpl 别名

from numpy import ndarray
# 导入 NumPy 的 ndarray 类型

from pandas import DataFrame
# 导入 Pandas 的 DataFrame 类型

from matplotlib.artist import Artist
# 从 Matplotlib 中导入 Artist 类

from seaborn._core.scales import Scale
# 从 Seaborn 库中导入 Scale 类

from seaborn._core.properties import (
    PROPERTIES,
    Property,
    RGBATuple,
    DashPattern,
    DashPatternWithOffset,
)
# 从 Seaborn 库中导入多个属性和类型

from seaborn._core.exceptions import PlotSpecError
# 导入 Seaborn 库中的绘图规范错误类

class Mappable:
    def __init__(
        self,
        val: Any = None,
        depend: str | None = None,
        rc: str | None = None,
        auto: bool = False,
        grouping: bool = True,
    ):
        """
        Property that can be mapped from data or set directly, with flexible defaults.

        Parameters
        ----------
        val : Any
            Use this value as the default.
        depend : str
            Use the value of this feature as the default.
        rc : str
            Use the value of this rcParam as the default.
        auto : bool
            The default value will depend on other parameters at compile time.
        grouping : bool
            If True, use the mapped variable to define groups.

        """
        if depend is not None:
            assert depend in PROPERTIES
        if rc is not None:
            assert rc in mpl.rcParams

        self._val = val
        self._rc = rc
        self._depend = depend
        self._auto = auto
        self._grouping = grouping

    def __repr__(self):
        """Nice formatting for when object appears in Mark init signature."""
        if self._val is not None:
            s = f"<{repr(self._val)}>"
        elif self._depend is not None:
            s = f"<depend:{self._depend}>"
        elif self._rc is not None:
            s = f"<rc:{self._rc}>"
        elif self._auto:
            s = "<auto>"
        else:
            s = "<undefined>"
        return s
    # 返回对象的字符串表示形式

    @property
    def depend(self) -> Any:
        """Return the name of the feature to source a default value from."""
        return self._depend
    # 返回用于获取默认值的特性名称

    @property
    def grouping(self) -> bool:
        return self._grouping
    # 返回是否使用映射变量来定义组的布尔值

    @property
    def default(self) -> Any:
        """Get the default value for this feature, or access the relevant rcParam."""
        if self._val is not None:
            return self._val
        elif self._rc is not None:
            return mpl.rcParams.get(self._rc)
    # 获取此特性的默认值，或访问相关的 rcParam

# TODO where is the right place to put this kind of type aliasing?

MappableBool = Union[bool, Mappable]
# 创建 MappableBool 类型别名，表示可以是布尔值或 Mappable 对象

MappableString = Union[str, Mappable]
# 创建 MappableString 类型别名，表示可以是字符串或 Mappable 对象

MappableFloat = Union[float, Mappable]
# 创建 MappableFloat 类型别名，表示可以是浮点数或 Mappable 对象

MappableColor = Union[str, tuple, Mappable]
# 创建 MappableColor 类型别名，表示可以是字符串、元组或 Mappable 对象

MappableStyle = Union[str, DashPattern, DashPatternWithOffset, Mappable]
# 创建 MappableStyle 类型别名，表示可以是字符串、DashPattern、DashPatternWithOffset 或 Mappable 对象

@dataclass
class Mark:
    """Base class for objects that visually represent data."""

    artist_kws: dict = field(default_factory=dict)
    # 定义 Mark 类的 artist_kws 属性，用于存储艺术家关键字参数

    @property
    # 将以下方法装饰为属性方法
    # 返回一个字典，包含所有具有默认值为 Mappable 类型的属性名及其对应的属性值
    def _mappable_props(self):
        return {
            f.name: getattr(self, f.name) for f in fields(self)
            if isinstance(f.default, Mappable)
        }

    @property
    def _grouping_props(self):
        # 返回一个列表，包含所有具有默认值为 Mappable 类型且属性 grouping 为 True 的属性名
        # TODO 在 Mark 对象的属性中，是否有意义使其在分组时具有变化？
        return [
            f.name for f in fields(self)
            if isinstance(f.default, Mappable) and f.default.grouping
        ]

    # TODO 是否应该将此方法私有化？扩展程序是否需要直接调用它？
    def _resolve(
        self,
        data: DataFrame | dict[str, Any],
        name: str,
        scales: dict[str, Scale] | None = None,
    ) -> Any:
        """
        Obtain default, specified, or mapped value for a named feature.

        Parameters
        ----------
        data : DataFrame or dict with scalar values
            Container with data values for features that will be semantically mapped.
        name : string
            Identity of the feature / semantic.
        scales: dict
            Mapping from variable to corresponding scale object.

        Returns
        -------
        value or array of values
            Outer return type depends on whether `data` is a dict (implying that
            we want a single value) or DataFrame (implying that we want an array
            of values with matching length).
        """

        # Obtain the mappable property corresponding to the feature name
        feature = self._mappable_props[name]

        # Get the property definition from global properties or create a new one
        prop = PROPERTIES.get(name, Property(name))

        # Check if the feature is directly specified (not an instance of Mappable)
        directly_specified = not isinstance(feature, Mappable)

        # Determine if the return type should be an array (for DataFrame)
        return_multiple = isinstance(data, pd.DataFrame)

        # Determine if the returned array should be numpy array (for DataFrame)
        return_array = return_multiple and not name.endswith("style")

        # Special handling for 'width' feature to ensure it's properly resolved and added
        if name == "width":
            directly_specified = directly_specified and name not in data

        if directly_specified:
            # Standardize the feature property value
            feature = prop.standardize(feature)

            # If returning multiple values (DataFrame), replicate the feature value
            if return_multiple:
                feature = [feature] * len(data)

            # If returning as an array, convert to numpy array
            if return_array:
                feature = np.array(feature)

            return feature

        # If the feature name exists in the data container
        if name in data:
            if scales is None or name not in scales:
                # If no scaling is required, directly use the data value
                feature = data[name]
            else:
                # Apply scaling to the data value using the specified scale function
                scale = scales[name]
                value = data[name]
                try:
                    feature = scale(value)
                except Exception as err:
                    # Handle errors during scaling operation
                    raise PlotSpecError._during("Scaling operation", name) from err

            # If returning as an array, convert to numpy array
            if return_array:
                feature = np.asarray(feature)

            return feature

        # If the feature has a dependency, resolve it recursively
        if feature.depend is not None:
            return self._resolve(data, feature.depend, scales)

        # Use the default value from the property definition
        default = prop.standardize(feature.default)

        # If returning multiple values (DataFrame), replicate the default value
        if return_multiple:
            default = [default] * len(data)

        # If returning as an array, convert to numpy array
        if return_array:
            default = np.array(default)

        return default
    # 推断数据的方向（orient），基于给定的比例尺（scales）
    def _infer_orient(self, scales: dict) -> str:  # TODO type scales

        # TODO 此函数在 seaborn._base 的原始版本中进行了更多的检查。
        # 在原型阶段，这里简化了一些内容，以确定哪些限制是有意义的。

        # TODO 重新考虑是否可以从比例尺类型映射到“DV优先级”，然后使用它？
        # 例如，标称（Nominal）> 离散（Discrete）> 连续（Continuous）

        x = 0 if "x" not in scales else scales["x"]._priority
        y = 0 if "y" not in scales else scales["y"]._priority

        # 根据优先级比较 x 和 y 的值，决定返回 "x" 还是 "y"
        if y > x:
            return "y"
        else:
            return "x"

    def _plot(
        self,
        split_generator: Callable[[], Generator],
        scales: dict[str, Scale],
        orient: str,
    ) -> None:
        """创建绘图的主要接口."""
        raise NotImplementedError()

    def _legend_artist(
        self, variables: list[str], value: Any, scales: dict[str, Scale],
    ) -> Artist | None:
        # 返回空，因为此函数尚未实现具体的图例元素生成逻辑
        return None
def resolve_properties(
    mark: Mark, data: DataFrame, scales: dict[str, Scale]
) -> dict[str, Any]:
    """
    Resolve properties for a given mark using provided data and scales.

    Parameters
    ----------
    mark :
        The mark object for which properties are resolved.
    data :
        DataFrame or dict containing data values.
    scales :
        Dictionary mapping property names to corresponding Scale objects.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping property names to resolved values.
    """

    # Resolve each property by iterating over the mappable properties of the mark
    props = {
        name: mark._resolve(data, name, scales) for name in mark._mappable_props
    }
    return props


def resolve_color(
    mark: Mark,
    data: DataFrame | dict,
    prefix: str = "",
    scales: dict[str, Scale] | None = None,
) -> RGBATuple | ndarray:
    """
    Resolve color and potentially alpha value for a mark.

    This function supports both specified and mapped color values, including alpha.

    Parameters
    ----------
    mark :
        The mark object for which color is resolved.
    data :
        DataFrame or dict containing data values.
    prefix :
        Prefix indicating the type of color (e.g., "color", "fillcolor").
    scales :
        Dictionary mapping property names to corresponding Scale objects, or None.

    Returns
    -------
    RGBATuple | ndarray
        Tuple or array representing RGBA color values.

    Notes
    -----
    This function handles visibility checks for colors and alpha values.
    """

    # Resolve the color using the specified prefix and scales
    color = mark._resolve(data, f"{prefix}color", scales)

    # Determine alpha value, prioritizing prefix-specific alpha if available
    if f"{prefix}alpha" in mark._mappable_props:
        alpha = mark._resolve(data, f"{prefix}alpha", scales)
    else:
        alpha = mark._resolve(data, "alpha", scales)

    def visible(x, axis=None):
        """Detect 'invisible' colors to set alpha appropriately."""
        # Check if all values in x are finite floats
        return np.array(x).dtype.kind != "f" or np.isfinite(x).all(axis)

    # Handle different cases for color and alpha values
    if np.ndim(color) < 2 and all(isinstance(x, float) for x in color):
        # Handle RGBA tuple case
        if len(color) == 4:
            return mpl.colors.to_rgba(color)
        # Handle alpha setting based on visibility of color
        alpha = alpha if visible(color) else np.nan
        return mpl.colors.to_rgba(color, alpha)
    else:
        # Handle RGBA array case
        if np.ndim(color) == 2 and color.shape[1] == 4:
            return mpl.colors.to_rgba_array(color)
        # Adjust alpha values based on visibility
        alpha = np.where(visible(color, axis=1), alpha, np.nan)
        return mpl.colors.to_rgba_array(color, alpha)

    # TODO should we be implementing fill here too?
    # (i.e. set fillalpha to 0 when fill=False)


def document_properties(mark):
    """
    Document the mappable properties of a mark in its docstring.

    Parameters
    ----------
    mark :
        The mark object to document.

    Returns
    -------
    mark
        The modified mark object with updated docstring.
    """

    # Gather mappable properties from the mark
    properties = [f.name for f in fields(mark) if isinstance(f.default, Mappable)]

    # Prepare documentation text
    text = [
        "",
        "    This mark defines the following properties:",
        textwrap.fill(
            ", ".join([f"|{p}|" for p in properties]),
            width=78, initial_indent=" " * 8, subsequent_indent=" " * 8,
        ),
    ]

    # Update mark's docstring with new documentation
    docstring_lines = mark.__doc__.split("\n")
    new_docstring = "\n".join([
        *docstring_lines[:2],
        *text,
        *docstring_lines[2:],
    ])
    mark.__doc__ = new_docstring
    return mark
```