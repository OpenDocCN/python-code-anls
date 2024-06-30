# `D:\src\scipysrc\seaborn\seaborn\_core\properties.py`

```
from __future__ import annotations
import itertools  # 导入 itertools 模块，提供了用于迭代的函数
import warnings  # 导入 warnings 模块，用于警告处理

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.typing import ArrayLike  # 导入 NumPy 提供的类型定义 ArrayLike

from pandas import Series  # 从 Pandas 库中导入 Series 类
import matplotlib as mpl  # 导入 Matplotlib 库，并使用 mpl 别名
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array  # 从 Matplotlib 中导入颜色转换函数
from matplotlib.markers import MarkerStyle  # 导入 Matplotlib 的 MarkerStyle 类
from matplotlib.path import Path  # 导入 Matplotlib 的 Path 类

from seaborn._core.scales import Scale, Boolean, Continuous, Nominal, Temporal  # 从 Seaborn 库中导入不同的数据尺度类
from seaborn._core.rules import categorical_order, variable_type  # 从 Seaborn 中导入数据规则处理函数
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette  # 从 Seaborn 中导入颜色调色板相关函数
from seaborn.utils import get_color_cycle  # 从 Seaborn 中导入获取颜色循环的实用函数

from typing import Any, Callable, Tuple, List, Union, Optional  # 导入类型定义

RGBTuple = Tuple[float, float, float]  # 定义 RGBTuple 类型为三个浮点数元组
RGBATuple = Tuple[float, float, float, float]  # 定义 RGBATuple 类型为四个浮点数元组
ColorSpec = Union[RGBTuple, RGBATuple, str]  # 定义 ColorSpec 类型为 RGBTuple、RGBATuple 或字符串的联合类型

DashPattern = Tuple[float, ...]  # 定义 DashPattern 类型为浮点数元组
DashPatternWithOffset = Tuple[float, Optional[DashPattern]]  # 定义 DashPatternWithOffset 类型为包含浮点数元组和可选 DashPattern 的元组

MarkerPattern = Union[  # 定义 MarkerPattern 类型为多种类型的联合类型，用于表示标记样式
    float,
    str,
    Tuple[int, int, float],
    List[Tuple[float, float]],
    Path,
    MarkerStyle,
]

Mapping = Callable[[ArrayLike], ArrayLike]  # 定义 Mapping 类型为接受 ArrayLike 参数并返回 ArrayLike 的可调用类型
    # 给定数据和缩放参数，初始化相应的缩放类
    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """Given data and a scaling argument, initialize appropriate scale class."""
        # TODO 将这些放在外部某处进行验证
        # TODO 如果子类定义了 infer_scale（例如颜色），在此处放置它们将无法捕捉到。
        # 最佳处理方法是在处理特定于属性的可能性后（例如对颜色进行检查，确保参数不是有效的调色板名称）调用 super，但这可能会变得棘手。
        
        # 可能的转换参数列表
        trans_args = ["log", "symlog", "logit", "pow", "sqrt"]
        # 如果参数是字符串类型
        if isinstance(arg, str):
            # 检查参数是否以转换参数列表中的任何一个开头
            if any(arg.startswith(k) for k in trans_args):
                # TODO 验证数值类型？这应该在某个地方集中进行
                return Continuous(trans=arg)
            else:
                # 抛出值错误异常，指示未知的参数值
                msg = f"Unknown magic arg for {self.variable} scale: '{arg}'."
                raise ValueError(msg)
        else:
            # 获取参数的类型名称
            arg_type = type(arg).__name__
            # 抛出类型错误异常，指示参数类型错误
            msg = f"Magic arg for {self.variable} scale must be str, not {arg_type}."
            raise TypeError(msg)

    # 返回一个函数，用于将数据域映射到属性范围
    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """Return a function that maps from data domain to property range."""
        # 定义一个恒等函数
        def identity(x):
            return x
        # 返回恒等函数
        return identity

    # 将灵活的属性值转换为标准化表示
    def standardize(self, val: Any) -> Any:
        """Coerce flexible property value to standardized representation."""
        # 直接返回输入值
        return val

    # 当以字典形式提供值时进行输入检查
    def _check_dict_entries(self, levels: list, values: dict) -> None:
        """Input check when values are provided as a dictionary."""
        # 计算缺失的条目
        missing = set(levels) - set(values)
        # 如果存在缺失的条目
        if missing:
            # 格式化缺失条目的字符串表示
            formatted = ", ".join(map(repr, sorted(missing, key=str)))
            # 构造值错误异常，指示缺少的条目
            err = f"No entry in {self.variable} dictionary for {formatted}"
            raise ValueError(err)

    # 当以列表形式提供值时进行输入检查
    def _check_list_length(self, levels: list, values: list) -> list:
        """Input check when values are provided as a list."""
        # 初始化消息字符串为空
        message = ""
        # 如果级别数多于值的数量
        if len(levels) > len(values):
            # 构造消息字符串，指示列表值不足，将会循环使用
            message = " ".join([
                f"\nThe {self.variable} list has fewer values ({len(values)})",
                f"than needed ({len(levels)}) and will cycle, which may",
                "produce an uninterpretable plot."
            ])
            # 使用 itertools.cycle 来填充值，直到达到与级别数相同的数量
            values = [x for _, x in zip(levels, itertools.cycle(values))]

        # 如果值的数量多于级别数
        elif len(values) > len(levels):
            # 构造消息字符串，指示列表值过多，可能不是预期的行为
            message = " ".join([
                f"The {self.variable} list has more values ({len(values)})",
                f"than needed ({len(levels)}), which may not be intended.",
            ])
            # 截断值列表，使其长度与级别数相同
            values = values[:len(levels)]

        # TODO 探索具有更好格式的自定义 PlotSpecWarning
        # 如果存在消息字符串，则发出用户警告
        if message:
            warnings.warn(message, UserWarning)

        # 返回经过处理后的值列表
        return values
# =================================================================================== #
# Properties relating to spatial position of marks on the plotting axes
# =================================================================================== #

# Coordinate类，表示视觉标记相对于绘图坐标轴的位置属性
class Coordinate(Property):
    """The position of visual marks with respect to the axes of the plot."""
    legend = False  # 是否显示在图例中，默认为False，不显示
    normed = False   # 是否标准化，默认为False，不进行标准化

# =================================================================================== #
# Properties with numeric values where scale range can be defined as an interval
# =================================================================================== #

# IntervalProperty类，表示数值属性，其比例范围可以定义为一个区间
class IntervalProperty(Property):
    """A numeric property where scale range can be defined as an interval."""
    legend = True   # 是否显示在图例中，默认为True，显示在图例中
    normed = True   # 是否标准化，默认为True，进行标准化

    _default_range: tuple[float, float] = (0, 1)  # 默认的数值范围，作为元组表示 (最小值, 最大值)

    @property
    def default_range(self) -> tuple[float, float]:
        """Min and max values used by default for semantic mapping."""
        return self._default_range
        # 返回默认的数值范围，用于语义映射的默认最小和最大值

    def _forward(self, values: ArrayLike) -> ArrayLike:
        """Transform applied to native values before linear mapping into interval."""
        return values
        # 在将值线性映射到区间之前，应用于原始值的转换

    def _inverse(self, values: ArrayLike) -> ArrayLike:
        """Transform applied to results of mapping that returns to native values."""
        return values
        # 应用于映射结果并返回原始值的转换

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """Given data and a scaling argument, initialize appropriate scale class."""

        # TODO infer continuous based on log/sqrt etc?
        # 根据对数、平方根等推断连续型变量的比例尺类型

        var_type = variable_type(data, boolean_type="boolean", strict_boolean=True)
        # 确定数据的变量类型

        if var_type == "boolean":
            return Boolean(arg)
            # 如果数据是布尔类型，返回Boolean比例尺对象
        elif isinstance(arg, (list, dict)):
            return Nominal(arg)
            # 如果参数是列表或字典，返回Nominal（名义）比例尺对象
        elif var_type == "categorical":
            return Nominal(arg)
            # 如果数据是分类变量，返回Nominal（名义）比例尺对象
        elif var_type == "datetime":
            return Temporal(arg)
            # 如果数据是日期时间变量，返回Temporal（时间）比例尺对象
        # TODO other variable types
        else:
            return Continuous(arg)
            # 对于其他类型的变量，返回Continuous（连续）比例尺对象
    # 返回一个函数，将数据域映射到属性范围的函数
    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """Return a function that maps from data domain to property range."""
        # 如果规模是名义尺度（Nominal），调用私有方法获取名义映射
        if isinstance(scale, Nominal):
            return self._get_nominal_mapping(scale, data)
        # 如果规模是布尔尺度（Boolean），调用私有方法获取布尔映射
        elif isinstance(scale, Boolean):
            return self._get_boolean_mapping(scale, data)

        # 如果尺度值为空，则使用默认范围进行正向映射
        if scale.values is None:
            vmin, vmax = self._forward(self.default_range)
        # 如果尺度值是二元组且长度为2，则使用尺度值进行正向映射
        elif isinstance(scale.values, tuple) and len(scale.values) == 2:
            vmin, vmax = self._forward(scale.values)
        else:
            # 如果尺度值是二元组但长度不为2，记录实际类型
            if isinstance(scale.values, tuple):
                actual = f"{len(scale.values)}-tuple"
            else:
                actual = str(type(scale.values))
            scale_class = scale.__class__.__name__
            # 抛出类型错误，要求尺度值为二元组
            err = " ".join([
                f"Values for {self.variable} variables with {scale_class} scale",
                f"must be 2-tuple; not {actual}.",
            ])
            raise TypeError(err)

        # 定义映射函数，将数据从数据域映射到属性范围
        def mapping(x):
            return self._inverse(np.multiply(x, vmax - vmin) + vmin)

        return mapping

    # 获取名义尺度（Nominal）的映射函数
    def _get_nominal_mapping(self, scale: Nominal, data: Series) -> Mapping:
        """Identify evenly-spaced values using interval or explicit mapping."""
        # 按照类别顺序获取数据的水平
        levels = categorical_order(data, scale.order)
        # 获取名义尺度的值
        values = self._get_values(scale, levels)

        # 定义映射函数，将数据从数据域映射到属性范围
        def mapping(x):
            ixs = np.asarray(x, np.intp)
            out = np.full(len(x), np.nan)
            use = np.isfinite(x)
            out[use] = np.take(values, ixs[use])
            return out

        return mapping

    # 获取布尔尺度（Boolean）的映射函数
    def _get_boolean_mapping(self, scale: Boolean, data: Series) -> Mapping:
        """Identify evenly-spaced values using interval or explicit mapping."""
        # 获取布尔尺度的值
        values = self._get_values(scale, [True, False])

        # 定义映射函数，将数据从数据域映射到属性范围
        def mapping(x):
            out = np.full(len(x), np.nan)
            use = np.isfinite(x)
            out[use] = np.where(x[use], *values)
            return out

        return mapping
    def _get_values(self, scale: Scale, levels: list) -> list:
        """Validate scale.values and identify a value for each level."""
        # 检查 scale.values 的类型是否为字典
        if isinstance(scale.values, dict):
            # 如果是字典，则检查字典的条目，并根据 levels 列表提取对应的值
            self._check_dict_entries(levels, scale.values)
            values = [scale.values[x] for x in levels]
        elif isinstance(scale.values, list):
            # 如果是列表，则检查列表的长度是否符合 levels 的长度，并直接使用列表作为 values
            values = self._check_list_length(levels, scale.values)
        else:
            # 如果 scale.values 不是字典或列表，处理其他可能的情况
            if scale.values is None:
                # 如果为 None，则使用默认的范围
                vmin, vmax = self.default_range
            elif isinstance(scale.values, tuple):
                # 如果是元组，则解包 vmin 和 vmax
                vmin, vmax = scale.values
            else:
                # 如果既不是字典、列表、None，也不是元组，则抛出类型错误
                scale_class = scale.__class__.__name__
                err = " ".join([
                    f"Values for {self.variable} variables with {scale_class} scale",
                    f"must be a dict, list or tuple; not {type(scale.values)}",
                ])
                raise TypeError(err)

            # 对 vmin 和 vmax 进行正向变换，并使用反向变换生成与 levels 长度相等的值列表
            vmin, vmax = self._forward([vmin, vmax])
            values = list(self._inverse(np.linspace(vmax, vmin, len(levels))))

        # 返回处理后的 values 列表
        return values
class PointSize(IntervalProperty):
    """Size (diameter) of a point mark, in points, with scaling by area."""
    _default_range = 2, 8  # TODO use rcparams?

    def _forward(self, values):
        """Square native values to implement linear scaling of point area."""
        return np.square(values)

    def _inverse(self, values):
        """Invert areal values back to point diameter."""
        return np.sqrt(values)


class LineWidth(IntervalProperty):
    """Thickness of a line mark, in points."""
    
    @property
    def default_range(self) -> tuple[float, float]:
        """Min and max values used by default for semantic mapping."""
        base = mpl.rcParams["lines.linewidth"]  # 获取当前 matplotlib 设置中的线条宽度基础值
        return base * .5, base * 2  # 返回基础值的一半到两倍范围内的元组


class EdgeWidth(IntervalProperty):
    """Thickness of the edges on a patch mark, in points."""
    
    @property
    def default_range(self) -> tuple[float, float]:
        """Min and max values used by default for semantic mapping."""
        base = mpl.rcParams["patch.linewidth"]  # 获取当前 matplotlib 设置中的补丁边缘宽度基础值
        return base * .5, base * 2  # 返回基础值的一半到两倍范围内的元组


class Stroke(IntervalProperty):
    """Thickness of lines that define point glyphs."""
    _default_range = .25, 2.5  # 定义点标记边界线的默认粗细范围


class Alpha(IntervalProperty):
    """Opacity of the color values for an arbitrary mark."""
    _default_range = .3, .95  # 定义任意标记的颜色不透明度范围
    # TODO validate / enforce that output is in [0, 1]  # TODO：验证/强制确保输出在[0, 1]范围内


class Offset(IntervalProperty):
    """Offset for edge-aligned text, in point units."""
    _default_range = 0, 5  # 定义边缘对齐文本的偏移量范围，单位为点
    _legend = False  # 设置不在图例中显示


class FontSize(IntervalProperty):
    """Font size for textual marks, in points."""
    _legend = False  # 设置不在图例中显示

    @property
    def default_range(self) -> tuple[float, float]:
        """Min and max values used by default for semantic mapping."""
        base = mpl.rcParams["font.size"]  # 获取当前 matplotlib 设置中的字体大小基础值
        return base * .5, base * 2  # 返回基础值的一半到两倍范围内的元组


# =================================================================================== #
# Properties defined by arbitrary objects with inherently nominal scaling
# =================================================================================== #

class ObjectProperty(Property):
    """A property defined by arbitrary an object, with inherently nominal scaling."""
    legend = True  # 属性可以显示在图例中
    normed = False  # 标准化设置为假

    # Object representing null data, should appear invisible when drawn by matplotlib
    # Note that we now drop nulls in Plot._plot_layer and thus may not need this
    null_value: Any = None  # 用于表示空数据的对象，绘制时应该是不可见的，matplotlib中可能不再需要这个

    def _default_values(self, n: int) -> list:
        raise NotImplementedError()  # 抛出未实现错误，需要子类实现

    def default_scale(self, data: Series) -> Scale:
        var_type = variable_type(data, boolean_type="boolean", strict_boolean=True)  # 根据数据推断类型
        return Boolean() if var_type == "boolean" else Nominal()  # 如果是布尔型数据返回布尔类型，否则返回名义类型

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        var_type = variable_type(data, boolean_type="boolean", strict_boolean=True)  # 根据数据推断类型
        return Boolean(arg) if var_type == "boolean" else Nominal(arg)  # 根据输入参数和数据类型推断比例尺
    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """Define mapping as lookup into list of object values."""
        # 检查 scale 是否是 Boolean 类型
        boolean_scale = isinstance(scale, Boolean)
        # 获取 scale 的 order 属性，如果是 Boolean 类型则默认为 [True, False]
        order = getattr(scale, "order", [True, False] if boolean_scale else None)
        # 根据数据和 order 顺序获取分类变量的顺序
        levels = categorical_order(data, order)
        # 获取 scale 对应的值列表
        values = self._get_values(scale, levels)

        # 如果 scale 是 Boolean 类型，反转 values 列表
        if boolean_scale:
            values = values[::-1]

        # 定义 mapping 函数，将输入的 x 映射为对应的 values 值
        def mapping(x):
            # 将 x 转换为 numpy 数组并处理 NaN 值为 0
            ixs = np.asarray(np.nan_to_num(x), np.intp)
            # 根据索引 ix 返回对应的 values 值，若 x_i 是有限的则返回 self.null_value
            return [
                values[ix] if np.isfinite(x_i) else self.null_value
                for x_i, ix in zip(x, ixs)
            ]

        # 返回 mapping 函数
        return mapping

    def _get_values(self, scale: Scale, levels: list) -> list:
        """Validate scale.values and identify a value for each level."""
        # 获取 levels 的长度
        n = len(levels)
        # 如果 scale.values 是 dict 类型，校验并获取每个 level 对应的值
        if isinstance(scale.values, dict):
            self._check_dict_entries(levels, scale.values)
            values = [scale.values[x] for x in levels]
        # 如果 scale.values 是 list 类型，校验其长度并返回
        elif isinstance(scale.values, list):
            values = self._check_list_length(levels, scale.values)
        # 如果 scale.values 是 None，则使用默认值
        elif scale.values is None:
            values = self._default_values(n)
        # 如果 scale.values 是其他类型，则抛出类型错误异常
        else:
            msg = " ".join([
                f"Scale values for a {self.variable} variable must be provided",
                f"in a dict or list; not {type(scale.values)}."
            ])
            raise TypeError(msg)

        # 标准化 values 列表中的每个值并返回
        values = [self.standardize(x) for x in values]
        return values
class Marker(ObjectProperty):
    """Shape of points in scatter-type marks or lines with data points marked."""
    # 定义空值常量，用于标记样式
    null_value = MarkerStyle("")

    # TODO should we have named marker "palettes"? (e.g. see d3 options)

    # TODO need some sort of "require_scale" functionality
    # to raise when we get the wrong kind explicitly specified

    def standardize(self, val: MarkerPattern) -> MarkerStyle:
        """Standardize marker pattern to MarkerStyle object."""
        return MarkerStyle(val)

    def _default_values(self, n: int) -> list[MarkerStyle]:
        """Build an arbitrarily long list of unique marker styles.

        Parameters
        ----------
        n : int
            Number of unique marker specs to generate.

        Returns
        -------
        markers : list of string or tuples
            Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
            All markers will be filled.

        """
        # Start with marker specs that are well distinguishable
        markers = [
            "o", "X", (4, 0, 45), "P", (4, 0, 0), (4, 1, 0), "^", (4, 1, 45), "v",
        ]

        # Now generate more from regular polygons of increasing order
        s = 5
        while len(markers) < n:
            a = 360 / (s + 1) / 2
            markers.extend([(s + 1, 1, a), (s + 1, 0, a), (s, 1, 0), (s, 0, 0)])
            s += 1

        # Convert marker specs to MarkerStyle objects
        markers = [MarkerStyle(m) for m in markers[:n]]

        return markers


class LineStyle(ObjectProperty):
    """Dash pattern for line-type marks."""
    # 定义空值常量，用于标记线型
    null_value = ""

    def standardize(self, val: str | DashPattern) -> DashPatternWithOffset:
        """Standardize dash pattern to DashPatternWithOffset object."""
        return self._get_dash_pattern(val)
    def _default_values(self, n: int) -> list[DashPatternWithOffset]:
        """Build an arbitrarily long list of unique dash styles for lines.

        Parameters
        ----------
        n : int
            Number of unique dash specs to generate.

        Returns
        -------
        dashes : list of strings or tuples
            Valid arguments for the ``dashes`` parameter on
            :class:`matplotlib.lines.Line2D`. The first spec is a solid
            line (``""``), the remainder are sequences of long and short
            dashes.

        """
        # Start with dash specs that are well distinguishable
        dashes: list[str | DashPattern] = [
            "-", (4, 1.5), (1, 1), (3, 1.25, 1.5, 1.25), (5, 1, 1, 1),
        ]

        # Now programmatically build as many as we need
        p = 3
        while len(dashes) < n:

            # Take combinations of long and short dashes
            a = itertools.combinations_with_replacement([3, 1.25], p)
            b = itertools.combinations_with_replacement([4, 1], p)

            # Interleave the combinations, reversing one of the streams
            segment_list = itertools.chain(*zip(list(a)[1:-1][::-1], list(b)[1:-1]))

            # Now insert the gaps
            for segments in segment_list:
                gap = min(segments)
                spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
                dashes.append(spec)

            p += 1

        # Return the list of dash patterns with calculated offsets
        return [self._get_dash_pattern(x) for x in dashes]
    def _get_dash_pattern(style: str | DashPattern) -> DashPatternWithOffset:
        """Convert linestyle arguments to dash pattern with offset."""
        # 从 Matplotlib 3.4 复制并修改的代码
        # 将简略风格转换为完整字符串风格
        ls_mapper = {"-": "solid", "--": "dashed", "-.": "dashdot", ":": "dotted"}
        
        # 如果 style 是字符串类型
        if isinstance(style, str):
            # 使用映射表 ls_mapper 将风格转换为完整字符串风格
            style = ls_mapper.get(style, style)
            
            # 对于不带虚线的风格
            if style in ["solid", "none", "None"]:
                offset = 0
                dashes = None
            # 对于带虚线的风格
            elif style in ["dashed", "dashdot", "dotted"]:
                offset = 0
                # 从 Matplotlib 配置中获取对应风格的虚线模式
                dashes = tuple(mpl.rcParams[f"lines.{style}_pattern"])
            else:
                # 如果风格不在预定义的风格中，抛出 ValueError 异常
                options = [*ls_mapper.values(), *ls_mapper.keys()]
                msg = f"Linestyle string must be one of {options}, not {repr(style)}."
                raise ValueError(msg)

        # 如果 style 是元组类型
        elif isinstance(style, tuple):
            # 如果风格长度大于 1 并且第二个元素也是元组
            if len(style) > 1 and isinstance(style[1], tuple):
                offset, dashes = style
            # 如果风格长度大于 1 并且第二个元素为 None
            elif len(style) > 1 and style[1] is None:
                offset, dashes = style
            else:
                offset = 0
                dashes = style
        else:
            # 如果 style 不是字符串也不是元组类型，抛出 TypeError 异常
            val_type = type(style).__name__
            msg = f"Linestyle must be str or tuple, not {val_type}."
            raise TypeError(msg)

        # 将偏移量规范化为正数，并确保偏移小于虚线周期
        if dashes is not None:
            try:
                dsum = sum(dashes)
            except TypeError as err:
                # 如果虚线模式不是可迭代的，抛出 TypeError 异常
                msg = f"Invalid dash pattern: {dashes}"
                raise TypeError(msg) from err
            # 对偏移量进行取模操作，确保偏移小于虚线周期
            if dsum:
                offset %= dsum

        # 返回规范化后的偏移量和虚线模式
        return offset, dashes
class TextAlignment(ObjectProperty):
    # TextAlignment 类，继承自 ObjectProperty
    legend = False  # 设置 legend 属性为 False


class HorizontalAlignment(TextAlignment):
    # HorizontalAlignment 类，继承自 TextAlignment

    def _default_values(self, n: int) -> list:
        # 返回长度为 n 的列表，其中元素循环取自 ["left", "right"]
        vals = itertools.cycle(["left", "right"])
        return [next(vals) for _ in range(n)]


class VerticalAlignment(TextAlignment):
    # VerticalAlignment 类，继承自 TextAlignment

    def _default_values(self, n: int) -> list:
        # 返回长度为 n 的列表，其中元素循环取自 ["top", "bottom"]
        vals = itertools.cycle(["top", "bottom"])
        return [next(vals) for _ in range(n)]


# =================================================================================== #
# Properties with RGB(A) color values
# =================================================================================== #


class Color(Property):
    """Color, as RGB(A), scalable with nominal palettes or continuous gradients."""
    legend = True  # 设置 legend 属性为 True
    normed = True  # 设置 normed 属性为 True

    def standardize(self, val: ColorSpec) -> RGBTuple | RGBATuple:
        # 根据输入的 ColorSpec 规范化颜色，如果有 alpha 通道则保留
        # RGBA 颜色可以覆盖 Alpha 属性
        if to_rgba(val) != to_rgba(val, 1):
            return to_rgba(val)
        else:
            return to_rgb(val)

    def _standardize_color_sequence(self, colors: ArrayLike) -> ArrayLike:
        """Convert color sequence to RGB(A) array, preserving but not adding alpha."""
        def has_alpha(x):
            # 检查颜色 x 是否有 alpha 通道
            return to_rgba(x) != to_rgba(x, 1)

        if isinstance(colors, np.ndarray):
            needs_alpha = colors.shape[1] == 4  # 判断是否需要 alpha 通道
        else:
            needs_alpha = any(has_alpha(x) for x in colors)

        if needs_alpha:
            return to_rgba_array(colors)  # 转换为 RGBA 数组
        else:
            return to_rgba_array(colors)[:, :3]  # 转换为 RGB 数组（去除 alpha 通道）
    # 推断变量的数据规模（量表）
    def infer_scale(self, arg: Any, data: Series) -> Scale:
        # TODO 当推断连续变量但没有数据时，需要验证类型

        # TODO 需要重新考虑变量类型系统
        # （例如布尔值、有序类别作为序数等）。。

        # 调用函数获取数据的变量类型，严格要求布尔类型
        var_type = variable_type(data, boolean_type="boolean", strict_boolean=True)

        # 如果变量类型是布尔值，则返回布尔量表对象
        if var_type == "boolean":
            return Boolean(arg)

        # 如果参数是字典或列表，则返回名义量表对象
        if isinstance(arg, (dict, list)):
            return Nominal(arg)

        # 如果参数是元组
        if isinstance(arg, tuple):
            # 如果数据是分类的，则考虑允许用于名义量表的渐变映射
            # 但这似乎在技术上不太正确。应该用有序数据推断序数量表吗？需要验证其有序性。
            if var_type == "categorical":
                return Nominal(arg)
            # 否则返回连续量表对象
            return Continuous(arg)

        # 如果参数是可调用对象，则返回连续量表对象
        if callable(arg):
            return Continuous(arg)

        # TODO 对于字符串参数如 "log"、"pow" 等，我们接受其语义吗？

        # 如果参数不是字符串，则抛出类型错误异常
        if not isinstance(arg, str):
            msg = " ".join([
                f"A single scale argument for {self.variable} variables must be",
                f"a string, dict, tuple, list, or callable, not {type(arg)}."
            ])
            raise TypeError(msg)

        # 如果参数是预定义的调色板名称，则返回名义量表对象
        if arg in QUAL_PALETTES:
            return Nominal(arg)
        # 如果数据类型是数值型，则返回连续量表对象
        elif var_type == "numeric":
            return Continuous(arg)
        # TODO 实现日期变量和其他变量的量表
        else:
            return Nominal(arg)
    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """Return a function that maps from data domain to color values."""
        # TODO what is best way to do this conditional?
        # Should it be class-based or should classes have behavioral attributes?
        if isinstance(scale, Nominal):
            # 如果缩放对象是名义型，则调用 _get_nominal_mapping 方法处理
            return self._get_nominal_mapping(scale, data)
        elif isinstance(scale, Boolean):
            # 如果缩放对象是布尔型，则调用 _get_boolean_mapping 方法处理
            return self._get_boolean_mapping(scale, data)

        if scale.values is None:
            # 如果缩放值为 None，则使用默认的色彩渐变调色板
            # TODO Rethink best default continuous color gradient
            mapping = color_palette("ch:", as_cmap=True)
        elif isinstance(scale.values, tuple):
            # 如果缩放值是元组，则使用 blend_palette 方法创建调色板
            # TODO blend_palette will strip alpha, but we should support
            # interpolation on all four channels
            mapping = blend_palette(scale.values, as_cmap=True)
        elif isinstance(scale.values, str):
            # 如果缩放值是字符串，则使用指定名称的 matplotlib 调色板
            # TODO for matplotlib colormaps this will clip extremes, which is
            # different from what using the named colormap directly would do
            # This may or may not be desireable.
            mapping = color_palette(scale.values, as_cmap=True)
        elif callable(scale.values):
            # 如果缩放值是可调用对象，则直接使用它作为映射函数
            mapping = scale.values
        else:
            # 如果缩放值类型不符合预期，则抛出类型错误异常
            scale_class = scale.__class__.__name__
            msg = " ".join([
                f"Scale values for {self.variable} with a {scale_class} mapping",
                f"must be string, tuple, or callable; not {type(scale.values)}."
            ])
            raise TypeError(msg)

        def _mapping(x):
            # Remove alpha channel so it does not override alpha property downstream
            # TODO this will need to be more flexible to support RGBA tuples (see above)
            invalid = ~np.isfinite(x)
            out = mapping(x)[:, :3]
            out[invalid] = np.nan
            return out

        return _mapping

    def _get_nominal_mapping(self, scale: Nominal, data: Series) -> Mapping:
        # 获取名义型缩放的映射函数

        levels = categorical_order(data, scale.order)
        colors = self._get_values(scale, levels)

        def mapping(x):
            ixs = np.asarray(np.nan_to_num(x), np.intp)
            use = np.isfinite(x)
            out = np.full((len(ixs), colors.shape[1]), np.nan)
            out[use] = np.take(colors, ixs[use], axis=0)
            return out

        return mapping

    def _get_boolean_mapping(self, scale: Boolean, data: Series) -> Mapping:
        # 获取布尔型缩放的映射函数

        colors = self._get_values(scale, [True, False])

        def mapping(x):
            use = np.isfinite(x)
            x = np.asarray(np.nan_to_num(x)).astype(bool)
            out = np.full((len(x), colors.shape[1]), np.nan)
            out[x & use] = colors[0]
            out[~x & use] = colors[1]
            return out

        return mapping
    # 定义一个方法 _get_values，用于根据指定的比例尺和级别列表获取值数组
    def _get_values(self, scale: Scale, levels: list) -> ArrayLike:
        """Validate scale.values and identify a value for each level."""
        # 计算级别列表的长度
        n = len(levels)
        # 从比例尺对象中获取值数组
        values = scale.values
        # 如果值数组是字典类型
        if isinstance(values, dict):
            # 检查字典中的键是否与级别列表相符
            self._check_dict_entries(levels, values)
            # 根据级别列表中的键顺序获取对应的值作为颜色值列表
            colors = [values[x] for x in levels]
        # 如果值数组是列表类型
        elif isinstance(values, list):
            # 检查列表长度是否与级别列表一致，并返回调整后的颜色值列表
            colors = self._check_list_length(levels, values)
        # 如果值数组是元组类型
        elif isinstance(values, tuple):
            # 使用 blend_palette 函数根据元组定义的颜色值混合生成 n 个颜色值
            colors = blend_palette(values, n)
        # 如果值数组是字符串类型
        elif isinstance(values, str):
            # 使用 color_palette 函数根据字符串定义的调色板生成 n 个颜色值
            colors = color_palette(values, n)
        # 如果值数组为 None
        elif values is None:
            # 如果级别数小于等于全局默认调色环的长度，则使用当前全局默认调色板
            if n <= len(get_color_cycle()):
                colors = color_palette(n_colors=n)
            # 否则使用 "husl" 调色板生成 n 个颜色值
            else:
                colors = color_palette("husl", n)
        # 如果值数组不是以上任何一种类型，抛出类型错误异常
        else:
            scale_class = scale.__class__.__name__
            msg = " ".join([
                f"Scale values for {self.variable} with a {scale_class} mapping",
                f"must be string, list, tuple, or dict; not {type(scale.values)}."
            ])
            raise TypeError(msg)

        # 调用 _standardize_color_sequence 方法，标准化颜色序列并返回
        return self._standardize_color_sequence(colors)
# =================================================================================== #
# Properties that can take only two states
# =================================================================================== #

# Fill 类继承自 Property 类，表示图中点、条形图或图形对象可以填充或轮廓化的布尔属性。
class Fill(Property):
    """Boolean property of points/bars/patches that can be solid or outlined."""
    
    # 图例默认为 True
    legend = True
    
    # normed 默认为 False
    normed = False

    # 根据数据推断默认的比例尺类型
    def default_scale(self, data: Series) -> Scale:
        var_type = variable_type(data, boolean_type="boolean", strict_boolean=True)
        return Boolean() if var_type == "boolean" else Nominal()

    # 推断比例尺类型，支持用户指定的参数 arg
    def infer_scale(self, arg: Any, data: Series) -> Scale:
        var_type = variable_type(data, boolean_type="boolean", strict_boolean=True)
        return Boolean(arg) if var_type == "boolean" else Nominal(arg)

    # 标准化值为布尔类型
    def standardize(self, val: Any) -> bool:
        return bool(val)

    # 返回默认值列表，包含 n 个值，交替为 True 和 False
    def _default_values(self, n: int) -> list:
        """Return a list of n values, alternating True and False."""
        if n > 2:
            msg = " ".join([
                f"The variable assigned to {self.variable} has more than two levels,",
                f"so {self.variable} values will cycle and may be uninterpretable",
            ])
            # TODO 在一个“友好”的方式中触发警告（参见上文）
            warnings.warn(msg, UserWarning)
        return [x for x, _ in zip(itertools.cycle([True, False]), range(n))]

    # 获取数据映射函数，将每个数据值映射为 True 或 False
    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """Return a function that maps each data value to True or False."""
        boolean_scale = isinstance(scale, Boolean)
        order = getattr(scale, "order", [True, False] if boolean_scale else None)
        levels = categorical_order(data, order)
        values = self._get_values(scale, levels)

        if boolean_scale:
            values = values[::-1]

        # 定义映射函数
        def mapping(x):
            ixs = np.asarray(np.nan_to_num(x), np.intp)
            return [
                values[ix] if np.isfinite(x_i) else False
                for x_i, ix in zip(x, ixs)
            ]

        return mapping

    # 获取值列表，验证比例尺的值，并为每个级别识别一个值
    def _get_values(self, scale: Scale, levels: list) -> list:
        """Validate scale.values and identify a value for each level."""
        if isinstance(scale.values, list):
            values = [bool(x) for x in scale.values]
        elif isinstance(scale.values, dict):
            values = [bool(scale.values[x]) for x in levels]
        elif scale.values is None:
            values = self._default_values(len(levels))
        else:
            msg = " ".join([
                f"Scale values for {self.variable} must be passed in",
                f"a list or dict; not {type(scale.values)}."
            ])
            raise TypeError(msg)

        return values


# =================================================================================== #
# Enumeration of properties for use by Plot and Mark classes
# =================================================================================== #
# TODO turn this into a property registry with hooks, etc.
# TODO Users do not interact directly with properties, so how to document them?

# 定义属性类与其对应的名称的映射关系，用于属性注册表
PROPERTY_CLASSES = {
    "x": Coordinate,               # 属性 "x" 对应 Coordinate 类
    "y": Coordinate,               # 属性 "y" 对应 Coordinate 类
    "color": Color,                # 属性 "color" 对应 Color 类
    "alpha": Alpha,                # 属性 "alpha" 对应 Alpha 类
    "fill": Fill,                  # 属性 "fill" 对应 Fill 类
    "marker": Marker,              # 属性 "marker" 对应 Marker 类
    "pointsize": PointSize,        # 属性 "pointsize" 对应 PointSize 类
    "stroke": Stroke,              # 属性 "stroke" 对应 Stroke 类
    "linewidth": LineWidth,        # 属性 "linewidth" 对应 LineWidth 类
    "linestyle": LineStyle,        # 属性 "linestyle" 对应 LineStyle 类
    "fillcolor": Color,            # 属性 "fillcolor" 对应 Color 类
    "fillalpha": Alpha,            # 属性 "fillalpha" 对应 Alpha 类
    "edgewidth": EdgeWidth,        # 属性 "edgewidth" 对应 EdgeWidth 类
    "edgestyle": LineStyle,        # 属性 "edgestyle" 对应 LineStyle 类
    "edgecolor": Color,            # 属性 "edgecolor" 对应 Color 类
    "edgealpha": Alpha,            # 属性 "edgealpha" 对应 Alpha 类
    "text": Property,              # 属性 "text" 对应 Property 类
    "halign": HorizontalAlignment,# 属性 "halign" 对应 HorizontalAlignment 类
    "valign": VerticalAlignment,  # 属性 "valign" 对应 VerticalAlignment 类
    "offset": Offset,              # 属性 "offset" 对应 Offset 类
    "fontsize": FontSize,          # 属性 "fontsize" 对应 FontSize 类
    "xmin": Coordinate,            # 属性 "xmin" 对应 Coordinate 类
    "xmax": Coordinate,            # 属性 "xmax" 对应 Coordinate 类
    "ymin": Coordinate,            # 属性 "ymin" 对应 Coordinate 类
    "ymax": Coordinate,            # 属性 "ymax" 对应 Coordinate 类
    "group": Property,             # 属性 "group" 对应 Property 类
    # TODO pattern?                 # TODO 待补充: 是否需要添加 pattern 属性？
    # TODO gradient?                # TODO 待补充: 是否需要添加 gradient 属性？
}

# 创建属性实例字典，将每个属性名和对应类的实例关联起来
PROPERTIES = {var: cls(var) for var, cls in PROPERTY_CLASSES.items()}
```