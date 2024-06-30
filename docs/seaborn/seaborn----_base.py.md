# `D:\src\scipysrc\seaborn\seaborn\_base.py`

```
from __future__ import annotations
# 导入必要的模块和类
import warnings  # 引入警告模块
import itertools  # 引入迭代工具模块
from copy import copy  # 从复制模块中引入复制函数
from collections import UserString  # 从集合模块中引入用户字符串类
from collections.abc import Iterable, Sequence, Mapping  # 从集合抽象基类中引入可迭代、序列和映射类
from numbers import Number  # 从数值模块中引入数值类
from datetime import datetime  # 从日期时间模块中引入日期时间类

import numpy as np  # 导入数值计算模块numpy
import pandas as pd  # 导入数据处理模块pandas
import matplotlib as mpl  # 导入绘图模块matplotlib

from seaborn._core.data import PlotData  # 从seaborn核心数据模块中引入PlotData类
from seaborn.palettes import (  # 从seaborn调色板模块中引入QUAL_PALETTES和color_palette函数
    QUAL_PALETTES,
    color_palette,
)
from seaborn.utils import (  # 从seaborn工具模块中引入各种函数
    _check_argument,
    _version_predates,
    desaturate,
    locator_to_legend_entries,
    get_color_cycle,
    remove_na,
)


class SemanticMapping:
    """Base class for mapping data values to plot attributes."""

    # -- Default attributes that all SemanticMapping subclasses must set

    # Whether the mapping is numeric, categorical, or datetime
    map_type: str | None = None  # 映射类型，可以是数字、分类或日期时间

    # Ordered list of unique values in the input data
    levels = None  # 输入数据中唯一值的有序列表

    # A mapping from the data values to corresponding plot attributes
    lookup_table = None  # 数据值到对应绘图属性的映射表

    def __init__(self, plotter):
        # 构造函数，初始化plotter对象，用于绘图逻辑
        self.plotter = plotter

    def _check_list_length(self, levels, values, variable):
        """Input check when values are provided as a list."""
        # 检查当values作为列表提供时的输入合法性
        # 从_core/properties中复制的函数，最终将用新的函数替换。
        message = ""
        if len(levels) > len(values):
            message = " ".join([
                f"\nThe {variable} list has fewer values ({len(values)})",
                f"than needed ({len(levels)}) and will cycle, which may",
                "produce an uninterpretable plot."
            ])
            values = [x for _, x in zip(levels, itertools.cycle(values))]

        elif len(values) > len(levels):
            message = " ".join([
                f"The {variable} list has more values ({len(values)})",
                f"than needed ({len(levels)}), which may not be intended.",
            ])
            values = values[:len(levels)]

        if message:
            warnings.warn(message, UserWarning, stacklevel=6)

        return values

    def _lookup_single(self, key):
        """Apply the mapping to a single data value."""
        # 将映射应用于单个数据值
        return self.lookup_table[key]

    def __call__(self, key, *args, **kwargs):
        """Get the attribute(s) values for the data key."""
        # 获取数据键对应的属性值
        if isinstance(key, (list, np.ndarray, pd.Series)):
            return [self._lookup_single(k, *args, **kwargs) for k in key]
        else:
            return self._lookup_single(key, *args, **kwargs)


class HueMapping(SemanticMapping):
    """Mapping that sets artist colors according to data values."""
    # 根据数据值设置艺术家颜色的映射类
    # 指定应出现在图中的颜色规范
    palette = None

    # 用于归一化数据值到 [0, 1] 范围，用于颜色映射
    norm = None

    # 用于在数值上下文中进行插值的连续颜色映射对象
    cmap = None

    def __init__(
        self, plotter, palette=None, order=None, norm=None, saturation=1,
    ):
        """Map the levels of the `hue` variable to distinct colors.

        Parameters
        ----------
        plotter : object
            绘图对象，包含绘图所需的数据和设置
        palette : list or None, optional
            颜色调色板，用于映射不同的 `hue` 变量到不同的颜色
        order : list or None, optional
            `hue` 变量的顺序列表，用于排序
        norm : object or None, optional
            数据归一化对象，用于将数据归一化到特定范围
        saturation : float, default 1
            饱和度参数，用于调整颜色的饱和度
        """
        super().__init__(plotter)

        # 获取绘图数据中的 `hue` 列数据
        data = plotter.plot_data.get("hue", pd.Series(dtype=float))

        # 如果传入的 palette 是 numpy 数组，则转换为列表，发出警告信息
        if isinstance(palette, np.ndarray):
            msg = (
                "Numpy array is not a supported type for `palette`. "
                "Please convert your palette to a list. "
                "This will become an error in v0.14"
            )
            warnings.warn(msg, stacklevel=4)
            palette = palette.tolist()

        # 如果 `data` 中所有值都是 NaN，则根据情况发出警告信息
        if data.isna().all():
            if palette is not None:
                msg = "Ignoring `palette` because no `hue` variable has been assigned."
                warnings.warn(msg, stacklevel=4)
        else:
            # 推断映射类型（numeric、categorical 或 datetime）
            map_type = self.infer_map_type(
                palette, norm, plotter.input_format, plotter.var_types["hue"]
            )

            # 根据推断的映射类型进行不同的映射操作

            # --- Option 1: numeric mapping with a matplotlib colormap
            if map_type == "numeric":
                # 将 `data` 转换为数值类型
                data = pd.to_numeric(data)
                # 进行数值映射，获取 levels、lookup_table、norm 和 cmap
                levels, lookup_table, norm, cmap = self.numeric_mapping(
                    data, palette, norm,
                )

            # --- Option 2: categorical mapping using seaborn palette
            elif map_type == "categorical":
                # 对于分类数据，不需要 colormap 和 norm
                cmap = norm = None
                # 进行分类映射，获取 levels 和 lookup_table
                levels, lookup_table = self.categorical_mapping(
                    data, palette, order,
                )

            # --- Option 3: datetime mapping
            else:
                # TODO 实际上还需要实现 datetime 的映射
                cmap = norm = None
                # 对 datetime 数据进行分类映射，获取 levels 和 lookup_table
                levels, lookup_table = self.categorical_mapping(
                    list(data), palette, order,
                )

            # 将映射结果和参数保存到对象的属性中
            self.saturation = saturation
            self.map_type = map_type
            self.lookup_table = lookup_table
            self.palette = palette
            self.levels = levels
            self.norm = norm
            self.cmap = cmap
    def _lookup_single(self, key):
        """Get the color for a single value, using colormap to interpolate."""
        try:
            # 尝试从查找表中获取键对应的颜色值
            value = self.lookup_table[key]
        except KeyError:
            # 如果键不在查找表中

            if self.norm is None:
                # 当没有归一化函数时，通常发生在散点图中使用 hue_order，
                # 因为散点图不将 hue 视为分组变量
                # 因此未使用的 hue 级别存在于数据中，但不在查找表中
                return (0, 0, 0, 0)

            # 使用色彩映射函数插值生成颜色
            try:
                normed = self.norm(key)
            except TypeError as err:
                # 处理类型错误异常，通常发生在输入的键是 NaN 的情况下
                if np.isnan(key):
                    value = (0, 0, 0, 0)
                else:
                    raise err
            else:
                # 如果归一化结果是掩码值，将其视为 NaN
                if np.ma.is_masked(normed):
                    normed = np.nan
                # 使用 colormap 生成归一化后的颜色值
                value = self.cmap(normed)

        if self.saturation < 1:
            # 如果饱和度小于1，对颜色进行去饱和处理
            value = desaturate(value, self.saturation)

        return value

    def infer_map_type(self, palette, norm, input_format, var_type):
        """Determine how to implement the mapping."""
        if palette in QUAL_PALETTES:
            # 如果调色板在预定义的合格调色板列表中，说明是分类映射
            map_type = "categorical"
        elif norm is not None:
            # 如果存在归一化函数，说明是数值映射
            map_type = "numeric"
        elif isinstance(palette, (dict, list)):
            # 如果调色板是字典或列表类型，也是分类映射
            map_type = "categorical"
        elif input_format == "wide":
            # 如果输入格式是 wide，说明是分类映射
            map_type = "categorical"
        else:
            # 否则，根据变量类型来确定映射类型
            map_type = var_type

        return map_type

    def categorical_mapping(self, data, palette, order):
        """Determine colors when the hue mapping is categorical."""
        # -- Identify the order and name of the levels
        # 确定分类变量的顺序和名称

        levels = categorical_order(data, order)
        n_colors = len(levels)

        # -- Identify the set of colors to use
        # 确定要使用的颜色集合

        if isinstance(palette, dict):
            # 如果调色板是字典类型

            missing = set(levels) - set(palette)
            if any(missing):
                err = "The palette dictionary is missing keys: {}"
                raise ValueError(err.format(missing))

            lookup_table = palette

        else:
            # 如果调色板不是字典类型

            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                colors = self._check_list_length(levels, palette, "palette")
            else:
                colors = color_palette(palette, n_colors)

            lookup_table = dict(zip(levels, colors))

        return levels, lookup_table
    def numeric_mapping(self, data, palette, norm):
        """Determine colors when the hue variable is quantitative."""
        # 如果调色板是一个字典，则考虑传入的 norm 对象，以确定数值映射关系。
        if isinstance(palette, dict):
            # 获取字典中的键，并按字母顺序排序作为 levels
            levels = list(sorted(palette))
            # 根据排序后的键获取对应的颜色列表
            colors = [palette[k] for k in sorted(palette)]
            # 创建一个基于颜色列表的 ListedColormap 对象
            cmap = mpl.colors.ListedColormap(colors)
            # 复制一份调色板作为查找表
            lookup_table = palette.copy()

        else:
            # 如果调色板不是字典，则根据数据中的唯一值进行排序后的列表作为 levels
            levels = list(np.sort(remove_na(data.unique())))

            # --- 从调色板参数中确定要使用的色图

            # 默认的数值调色板是我们的默认 cubehelix 调色板
            # TODO 是否需要复杂处理以确保对比度？
            palette = "ch:" if palette is None else palette

            # 如果 palette 是 Colormap 对象，则直接使用
            if isinstance(palette, mpl.colors.Colormap):
                cmap = palette
            else:
                # 否则根据调色板创建一个色图对象
                cmap = color_palette(palette, as_cmap=True)

            # 现在处理数据的归一化
            if norm is None:
                norm = mpl.colors.Normalize()
            elif isinstance(norm, tuple):
                norm = mpl.colors.Normalize(*norm)
            elif not isinstance(norm, mpl.colors.Normalize):
                err = "``hue_norm`` must be None, tuple, or Normalize object."
                raise ValueError(err)

            # 如果归一化对象未进行缩放，则将数据的非空值转换为 NumPy 数组并进行归一化
            if not norm.scaled():
                norm(np.asarray(data.dropna()))

            # 使用色图对象对 levels 进行归一化处理，并将结果作为字典构成查找表
            lookup_table = dict(zip(levels, cmap(norm(levels))))

        # 返回 levels（唯一值列表）、查找表、归一化对象和色图对象
        return levels, lookup_table, norm, cmap
class SizeMapping(SemanticMapping):
    """Mapping that sets artist sizes according to data values."""

    # An object that normalizes data values to [0, 1] range
    norm = None

    def __init__(
        self, plotter, sizes=None, order=None, norm=None,
    ):
        """Map the levels of the `size` variable to distinct values.

        Parameters
        ----------
        plotter : Plotter
            An instance of the Plotter class used for plotting.
        sizes : dict or list, optional
            Dictionary or list specifying size mappings for categorical data.
        order : list, optional
            Specifies the order of categorical data.
        norm : object, optional
            Object responsible for normalizing data values to [0, 1] range.
        """

        super().__init__(plotter)

        # Get the 'size' data from plot_data; default to empty float Series if not present
        data = plotter.plot_data.get("size", pd.Series(dtype=float))

        if data.notna().any():
            # Determine the type of mapping based on input parameters and data characteristics
            map_type = self.infer_map_type(
                norm, sizes, plotter.var_types["size"]
            )

            # --- Option 1: numeric mapping
            if map_type == "numeric":
                # Perform numeric mapping based on data, sizes, and normalization object
                levels, lookup_table, norm, size_range = self.numeric_mapping(
                    data, sizes, norm,
                )

            # --- Option 2: categorical mapping
            elif map_type == "categorical":
                # Perform categorical mapping based on data, sizes, and order
                levels, lookup_table = self.categorical_mapping(
                    data, sizes, order,
                )
                size_range = None

            # --- Option 3: datetime mapping
            # Placeholder for future implementation
            else:
                # Perform categorical mapping for datetime data (casting data to list for compatibility)
                levels, lookup_table = self.categorical_mapping(
                    list(data), sizes, order,
                )
                size_range = None

            # Assign computed values to object attributes
            self.map_type = map_type
            self.levels = levels
            self.norm = norm
            self.sizes = sizes
            self.size_range = size_range
            self.lookup_table = lookup_table

    def infer_map_type(self, norm, sizes, var_type):
        """Infer the mapping type based on provided parameters.

        Parameters
        ----------
        norm : object
            Object responsible for normalizing data values to [0, 1] range.
        sizes : dict or list
            Dictionary or list specifying size mappings for categorical data.
        var_type : str
            Type of variable ('size' in this context).

        Returns
        -------
        str
            Type of mapping ('numeric', 'categorical', or specified var_type).
        """

        if norm is not None:
            map_type = "numeric"
        elif isinstance(sizes, (dict, list)):
            map_type = "categorical"
        else:
            map_type = var_type

        return map_type

    def _lookup_single(self, key):
        """Lookup a single key in the lookup_table and return the corresponding value.

        Parameters
        ----------
        key : hashable
            Key to lookup in the lookup_table.

        Returns
        -------
        object
            Corresponding value from the lookup_table or computed based on normalization.
        """

        try:
            value = self.lookup_table[key]
        except KeyError:
            # If key is not found, normalize the key and compute corresponding value
            normed = self.norm(key)
            if np.ma.is_masked(normed):
                normed = np.nan
            # Calculate value based on normalized key and size range
            value = self.size_range[0] + normed * np.ptp(self.size_range)
        return value
    # 定义一个方法，用于将分类数据映射到尺寸值
    def categorical_mapping(self, data, sizes, order):

        # 调用辅助函数 `categorical_order` 来获取按照指定顺序排列的分类级别
        levels = categorical_order(data, order)

        # 如果 `sizes` 参数是字典类型
        if isinstance(sizes, dict):

            # 字典输入将现有数据值映射到尺寸属性
            missing = set(levels) - set(sizes)
            if any(missing):
                err = f"Missing sizes for the following levels: {missing}"
                raise ValueError(err)
            lookup_table = sizes.copy()

        elif isinstance(sizes, list):

            # 如果 `sizes` 参数是列表类型，尺寸值与分类级别的顺序相同
            sizes = self._check_list_length(levels, sizes, "sizes")
            lookup_table = dict(zip(levels, sizes))

        else:

            # 如果 `sizes` 参数不是字典也不是列表

            if isinstance(sizes, tuple):

                # 如果 `sizes` 参数是元组类型，设置最小和最大尺寸值
                if len(sizes) != 2:
                    err = "A `sizes` tuple must have only 2 values"
                    raise ValueError(err)

            elif sizes is not None:

                # 如果 `sizes` 参数不是 `None`，报告参数值不可理解的错误
                err = f"Value for `sizes` not understood: {sizes}"
                raise ValueError(err)

            else:

                # 否则，从我们附加到的绘图对象中获取最小和最大尺寸值
                # TODO 此处可能会在以后造成麻烦，因为我们希望重新构造逻辑，
                # 使绘图器在数据的视觉表示方面更通用。但是在这一点上，
                # 我们不知道视觉表示。目前的做法是最清晰的。
                sizes = self.plotter._default_size_range

            # 对于分类尺寸，使用最小和最大尺寸之间的等间隔线性步骤
            # 并反转渐变，以便最大值用于 `size_order` 的第一个条目等。
            # 这是因为“有序”分类通常被认为是按降序优先级排序的。
            sizes = np.linspace(*sizes, len(levels))[::-1]
            lookup_table = dict(zip(levels, sizes))

        # 返回分类级别和相应的尺寸映射表
        return levels, lookup_table
class StyleMapping(SemanticMapping):
    """Mapping that sets artist style according to data values."""

    # Style mapping is always treated as categorical
    map_type = "categorical"

    def __init__(self, plotter, markers=None, dashes=None, order=None):
        """Map the levels of the `style` variable to distinct values.

        Parameters
        ----------
        plotter : object
            The plotter object responsible for generating the plot.
        markers : dict or None, optional
            Dictionary mapping style levels to marker types.
        dashes : dict or None, optional
            Dictionary mapping style levels to dash types.
        order : list or None, optional
            The order in which style levels should be processed.
        """
        super().__init__(plotter)

        # Extract 'style' data from plot_data attribute of plotter
        data = plotter.plot_data.get("style", pd.Series(dtype=float))

        if data.notna().any():

            # Cast to list to handle numpy/pandas datetime quirks
            if variable_type(data) == "datetime":
                data = list(data)

            # Find ordered unique values of 'style' data
            levels = categorical_order(data, order)

            # Map markers and dashes to style levels
            markers = self._map_attributes(
                markers, levels, unique_markers(len(levels)), "markers",
            )
            dashes = self._map_attributes(
                dashes, levels, unique_dashes(len(levels)), "dashes",
            )

            # Build marker paths for matplotlib
            paths = {}
            filled_markers = []
            for k, m in markers.items():
                if not isinstance(m, mpl.markers.MarkerStyle):
                    m = mpl.markers.MarkerStyle(m)
                paths[k] = m.get_path().transformed(m.get_transform())
                filled_markers.append(m.is_filled())

            # Ensure consistent marker type (filled or line art)
            if any(filled_markers) and not all(filled_markers):
                err = "Filled and line art markers cannot be mixed"
                raise ValueError(err)

            # Create lookup table for style levels
            lookup_table = {}
            for key in levels:
                lookup_table[key] = {}
                if markers:
                    lookup_table[key]["marker"] = markers[key]
                    lookup_table[key]["path"] = paths[key]
                if dashes:
                    lookup_table[key]["dashes"] = dashes[key]

            # Assign instance variables
            self.levels = levels
            self.lookup_table = lookup_table

    def _lookup_single(self, key, attr=None):
        """Get attribute(s) for a given data point.

        Parameters
        ----------
        key : object
            The style level for which attributes are queried.
        attr : str or None, optional
            Specific attribute to retrieve ('marker', 'path', or 'dashes').

        Returns
        -------
        object or dict
            Attribute value for the specified style level or dictionary of attributes.
        """
        if attr is None:
            value = self.lookup_table[key]
        else:
            value = self.lookup_table[key][attr]
        return value
    # 处理给定样式属性的规范化过程
    def _map_attributes(self, arg, levels, defaults, attr):
        """Handle the specification for a given style attribute."""
        # 如果 arg 是 True，则创建一个从 levels 到 defaults 的查找表
        if arg is True:
            lookup_table = dict(zip(levels, defaults))
        # 如果 arg 是 dict 类型，则使用 arg 作为查找表
        elif isinstance(arg, dict):
            # 检查 arg 中是否缺少 levels 中的任何键
            missing = set(levels) - set(arg)
            if missing:
                err = f"These `{attr}` levels are missing values: {missing}"
                raise ValueError(err)
            lookup_table = arg
        # 如果 arg 是 Sequence（序列），则根据 levels 和 arg 创建查找表
        elif isinstance(arg, Sequence):
            # 检查 arg 的长度是否与 levels 的长度一致
            arg = self._check_list_length(levels, arg, attr)
            lookup_table = dict(zip(levels, arg))
        # 如果 arg 为真值但不是上述类型，则抛出错误
        elif arg:
            err = f"This `{attr}` argument was not understood: {arg}"
            raise ValueError(err)
        # 如果 arg 是假值，则返回空的查找表
        else:
            lookup_table = {}

        # 返回最终生成的查找表
        return lookup_table
# =========================================================================== #

class VectorPlotter:
    """Base class for objects underlying *plot functions."""

    # 默认的宽格式结构，用于数据可视化，包含 x 轴、y 轴、颜色和样式的映射
    wide_structure = {
        "x": "@index", "y": "@values", "hue": "@columns", "style": "@columns",
    }
    
    # 默认的扁平格式结构，用于数据可视化，包含 x 轴和 y 轴的映射
    flat_structure = {"x": "@index", "y": "@values"}

    _default_size_range = 1, 2  # 未使用，但在测试中需要存在，略显烦人

    def __init__(self, data=None, variables={}):
        # 存储每个变量的级别信息的字典
        self._var_levels = {}

        # 用于有序分类轴变量的标记，通过 scale_* 方法进行处理
        self._var_ordered = {"x": False, "y": False}  # 或者，使用了 DefaultDict
        self.assign_variables(data, variables)

        # TODO 许多测试假设这些方法在类初始化时被调用以初始化映射到默认值的设置。
        #      我更希望摆脱这一点，并且只在显式调用时才有映射。
        # 根据传入的变量，如果存在 hue、size、style 等变量，则调用相应的映射方法
        for var in ["hue", "size", "style"]:
            if var in variables:
                getattr(self, f"map_{var}")()

    @property
    def has_xy_data(self):
        """判断是否至少定义了 x 轴或 y 轴的数据。"""
        return bool({"x", "y"} & set(self.variables))

    @property
    def var_levels(self):
        """获取变量级别的属性接口。

        每次访问时，更新 var_levels 字典，其中包含当前语义映射中的级别列表。
        同时也允许字典保持持久性，因此可以通过键设置级别。
        这用于跟踪使用附加的 FacetGrid 对象来设置的 col/row 级别列表，
        但这有些混乱，最好通过改进分面逻辑来更好地与现代绘图变量跟踪方法结合使用。
        
        """
        for var in self.variables:
            # 如果存在对应的映射对象，则更新 _var_levels 中的级别信息
            if (map_obj := getattr(self, f"_{var}_map", None)) is not None:
                self._var_levels[var] = map_obj.levels
        return self._var_levels
    def assign_variables(self, data=None, variables={}):
        """定义绘图变量，可选地使用`data`中的查找。"""
        # 从variables字典中获取变量"x"和"y"的值，若不存在则设为None
        x = variables.get("x", None)
        y = variables.get("y", None)

        # 如果x和y都为None，则设定输入格式为"wide"，并调用_assign_variables_wideform方法
        if x is None and y is None:
            self.input_format = "wide"
            frame, names = self._assign_variables_wideform(data, **variables)
        else:
            # 处理长格式输入时，使用新的PlotData对象来集中/标准化数据消耗逻辑
            self.input_format = "long"
            plot_data = PlotData(data, variables)
            frame = plot_data.frame
            names = plot_data.names

        # 将计算得到的frame赋给plot_data属性，将names赋给variables属性
        self.plot_data = frame
        self.variables = names

        # 根据变量名构建类型字典，包括数值型和分类型
        self.var_types = {
            v: variable_type(
                frame[v],
                boolean_type="numeric" if v in "xy" else "categorical"
            )
            for v in names
        }

        # 返回当前对象的引用
        return self

    def map_hue(self, palette=None, order=None, norm=None, saturation=1):
        # 创建HueMapping对象，设置hue映射
        mapping = HueMapping(self, palette, order, norm, saturation)
        self._hue_map = mapping

    def map_size(self, sizes=None, order=None, norm=None):
        # 创建SizeMapping对象，设置size映射
        mapping = SizeMapping(self, sizes, order, norm)
        self._size_map = mapping

    def map_style(self, markers=None, dashes=None, order=None):
        # 创建StyleMapping对象，设置style映射
        mapping = StyleMapping(self, markers, dashes, order)
        self._style_map = mapping

    def iter_data(
        self, grouping_vars=None, *,
        reverse=False, from_comp_data=False,
        by_facet=True, allow_empty=False, dropna=True,
    @property
    # 返回经过单位转换和对数缩放后的数值 x 和 y 的数据框(Dataframe)
    def comp_data(self):
        if not hasattr(self, "ax"):
            # 如果没有附加 Axes 对象，则返回原始的 plot_data
            # 可能需要更新一些测试用例，以使用外部接口，然后可以重新启用此功能
            return self.plot_data

        if not hasattr(self, "_comp_data"):
            # 复制 plot_data 的非深层副本，并丢弃列 "x" 和 "y"（如果存在）
            comp_data = (
                self.plot_data
                .copy(deep=False)
                .drop(["x", "y"], axis=1, errors="ignore")
            )

            # 针对 "x" 和 "y" 进行循环处理
            for var in "yx":
                if var not in self.variables:
                    continue

                parts = []
                # 根据转换器分组 plot_data[var]，不排序
                grouped = self.plot_data[var].groupby(self.converters[var], sort=False)
                for converter, orig in grouped:
                    # 将无穷大和无穷小替换为 NaN，然后删除 NaN 值
                    orig = orig.mask(orig.isin([np.inf, -np.inf]), np.nan)
                    orig = orig.dropna()
                    if var in self.var_levels:
                        # TODO 应该在某个集中位置处理这个步骤
                        # 支持分类图中 `order` 的处理比较棘手
                        orig = orig[orig.isin(self.var_levels[var])]
                    # 将原始数据 orig 转换为数值类型，再转换为浮点型
                    comp = pd.to_numeric(converter.convert_units(orig)).astype(float)
                    # 获取转换器的转换函数，并对 comp 应用该函数
                    transform = converter.get_transform().transform
                    parts.append(pd.Series(transform(comp), orig.index, name=orig.name))
                if parts:
                    comp_col = pd.concat(parts)
                else:
                    comp_col = pd.Series(dtype=float, name=var)
                # 在 comp_data 的开头插入处理后的数据列
                comp_data.insert(0, var, comp_col)

            self._comp_data = comp_data

        # 返回处理后的数据
        return self._comp_data

    # 根据子变量的存在情况返回 Axes 对象
    def _get_axes(self, sub_vars):
        row = sub_vars.get("row", None)
        col = sub_vars.get("col", None)
        if row is not None and col is not None:
            return self.facets.axes_dict[(row, col)]
        elif row is not None:
            return self.facets.axes_dict[row]
        elif col is not None:
            return self.facets.axes_dict[col]
        elif self.ax is None:
            return self.facets.ax
        else:
            return self.ax

    # 附加对象到当前对象，允许指定类型和对数缩放
    def _attach(
        self,
        obj,
        allowed_types=None,
        log_scale=None,
    def _get_scale_transforms(self, axis):
        """Return a function implementing the scale transform (or its inverse)."""
        # 如果未指定特定轴，则获取所有子图轴对象中特定轴的列表
        if self.ax is None:
            axis_list = [getattr(ax, f"{axis}axis") for ax in self.facets.axes.flat]
            # 获取所有轴对象中的刻度尺度，存储在集合中
            scales = {axis.get_scale() for axis in axis_list}
            # 如果发现不止一种尺度，则抛出运行时错误
            if len(scales) > 1:
                err = "Cannot determine transform with mixed scales on faceted axes."
                raise RuntimeError(err)
            # 获取第一个轴对象的变换对象
            transform_obj = axis_list[0].get_transform()
        else:
            # 如果指定了轴对象，则直接获取该轴对象的变换对象
            transform_obj = getattr(self.ax, f"{axis}axis").get_transform()

        return transform_obj.transform, transform_obj.inverted().transform

    def _add_axis_labels(self, ax, default_x="", default_y=""):
        """Add axis labels if not present, set visibility to match ticklabels."""
        # 如果 X 轴没有标签，则根据是否有刻度标签决定是否显示标签，并添加默认标签
        if not ax.get_xlabel():
            x_visible = any(t.get_visible() for t in ax.get_xticklabels())
            ax.set_xlabel(self.variables.get("x", default_x), visible=x_visible)
        # 如果 Y 轴没有标签，则根据是否有刻度标签决定是否显示标签，并添加默认标签
        if not ax.get_ylabel():
            y_visible = any(t.get_visible() for t in ax.get_yticklabels())
            ax.set_ylabel(self.variables.get("y", default_y), visible=y_visible)

    def add_legend_data(
        self, ax, func, common_kws=None, attrs=None, semantic_kws=None,
    ):
        """Add labeled artists to represent the different plot semantics."""
        # 获取图例显示的详细程度设置
        verbosity = self.legend
        # 如果 verbosity 是字符串且不在指定的选项中，则引发 ValueError 异常
        if isinstance(verbosity, str) and verbosity not in ["auto", "brief", "full"]:
            err = "`legend` must be 'auto', 'brief', 'full', or a boolean."
            raise ValueError(err)
        # 如果 verbosity 是 True，则将其设置为 "auto"
        elif verbosity is True:
            verbosity = "auto"

        keys = []  # 用于存储图例键的列表
        legend_kws = {}  # 存储每个图例条目的关键字参数
        # 如果 common_kws 为 None，则初始化为空字典，否则复制 common_kws
        common_kws = {} if common_kws is None else common_kws.copy()
        # 如果 semantic_kws 为 None，则初始化为空字典，否则复制 semantic_kws
        semantic_kws = {} if semantic_kws is None else semantic_kws.copy()

        # 根据 hue、size 和 style 的变量名获取标题，如果只有一个子图例，则将其赋给 title
        titles = {
            title for title in
            (self.variables.get(v, None) for v in ["hue", "size", "style"])
            if title is not None
        }
        title = "" if len(titles) != 1 else titles.pop()
        # 设置标题的默认关键字参数
        title_kws = dict(
            visible=False, color="w", s=0, linewidth=0, marker="", dashes=""
        )

        def update(var_name, val_name, **kws):
            # 更新图例条目的关键字参数
            key = var_name, val_name
            if key in legend_kws:
                legend_kws[key].update(**kws)
            else:
                keys.append(key)
                legend_kws[key] = dict(**kws)

        # 如果 attrs 为 None，则使用默认设置为 hue: "color", size: ["linewidth", "s"], style: None
        if attrs is None:
            attrs = {"hue": "color", "size": ["linewidth", "s"], "style": None}
        # 遍历 attrs 的每个变量和其对应的名称列表，更新图例数据
        for var, names in attrs.items():
            self._update_legend_data(
                update, var, verbosity, title, title_kws, names, semantic_kws.get(var),
            )

        legend_data = {}  # 存储图例数据的字典
        legend_order = []  # 存储图例顺序的列表

        # 不允许 common_kws 中 color=None，以便为 size/style 图例设置中性颜色
        if common_kws.get("color", False) is None:
            common_kws.pop("color")

        # 遍历 keys 列表中的每个图例键
        for key in keys:

            _, label = key  # 解包图例键，获取标签
            kws = legend_kws[key]  # 获取当前图例键对应的关键字参数
            level_kws = {}
            # 创建一个包含所有需要使用的属性的列表
            use_attrs = [
                *self._legend_attributes,
                *common_kws,
                *[attr for var_attrs in semantic_kws.values() for attr in var_attrs],
            ]
            # 遍历 use_attrs 中的每个属性，将其加入到 level_kws 中
            for attr in use_attrs:
                if attr in kws:
                    level_kws[attr] = kws[attr]
            # 调用 func 函数创建艺术家对象，并将其添加到 ax 对象中
            artist = func(label=label, **{"color": ".2", **common_kws, **level_kws})
            # 根据 matplotlib 版本的不同将 artist 添加到 ax 对象中
            if _version_predates(mpl, "3.5.0"):
                if isinstance(artist, mpl.lines.Line2D):
                    ax.add_line(artist)
                elif isinstance(artist, mpl.patches.Patch):
                    ax.add_patch(artist)
                elif isinstance(artist, mpl.collections.Collection):
                    ax.add_collection(artist)
            else:
                ax.add_artist(artist)
            # 将艺术家对象存储到 legend_data 字典中
            legend_data[key] = artist
            # 将图例键添加到 legend_order 列表中
            legend_order.append(key)

        # 设置图例标题、图例数据和图例顺序
        self.legend_title = title
        self.legend_data = legend_data
        self.legend_order = legend_order
    def _update_legend_data(
        self,
        update,
        var,
        verbosity,
        title,
        title_kws,
        attr_names,
        other_props,
    ):
        """Generate legend tick values and formatted labels."""
        # 设定简要显示的刻度数为6
        brief_ticks = 6
        # 获取属性var对应的映射器
        mapper = getattr(self, f"_{var}_map", None)
        # 如果映射器不存在，则返回
        if mapper is None:
            return

        # 确定是否使用简要显示模式
        brief = mapper.map_type == "numeric" and (
            verbosity == "brief"
            or (verbosity == "auto" and len(mapper.levels) > brief_ticks)
        )
        # 如果使用简要模式
        if brief:
            # 根据映射器的类型选择适当的定位器
            if isinstance(mapper.norm, mpl.colors.LogNorm):
                locator = mpl.ticker.LogLocator(numticks=brief_ticks)
            else:
                locator = mpl.ticker.MaxNLocator(nbins=brief_ticks)
            # 确定数据范围
            limits = min(mapper.levels), max(mapper.levels)
            # 使用定位器生成图例条目和格式化标签
            levels, formatted_levels = locator_to_legend_entries(
                locator, limits, self.plot_data[var].infer_objects().dtype
            )
        # 如果映射器的levels为空
        elif mapper.levels is None:
            levels = formatted_levels = []
        else:
            # 否则直接使用映射器的levels
            levels = formatted_levels = mapper.levels

        # 如果没有指定标题且self.variables中包含var
        if not title and self.variables.get(var, None) is not None:
            # 更新标题
            update((self.variables[var], "title"), self.variables[var], **title_kws)

        # 如果other_props为None则设为空字典
        other_props = {} if other_props is None else other_props

        # 遍历levels和formatted_levels，并更新属性
        for level, formatted_level in zip(levels, formatted_levels):
            if level is not None:
                # 获取属性
                attr = mapper(level)
                # 如果attr_names是列表，则为每个名称设置属性
                if isinstance(attr_names, list):
                    attr = {name: attr for name in attr_names}
                elif attr_names is not None:
                    attr = {attr_names: attr}
                # 更新属性，根据other_props中的条件
                attr.update({k: v[level] for k, v in other_props.items() if level in v})
                # 更新图例数据
                update(self.variables[var], formatted_level, **attr)

    # XXX 如果scale_*方法修改plot_data结构，则不能调用两次。这意味着如果它们被调用两次，则应该抛出异常。或者，我们可以存储plot_data的原始版本，并且每次调用它们时都操作存储的版本，而不是当前状态。

    def scale_native(self, axis, *args, **kwargs):

        # 默认情况下，委托给matplotlib处理
        raise NotImplementedError

    def scale_numeric(self, axis, *args, **kwargs):

        # 感觉需要完整性，它应该做什么？
        # 或许处理对数缩放？设置刻度器/格式化器/限制？

        raise NotImplementedError

    def scale_datetime(self, axis, *args, **kwargs):

        # 使用pd.to_datetime将字符串或数字转换为日期时间对象
        # 注意，使用天分辨率将数字转换为日期时间以匹配matplotlib

        raise NotImplementedError
class VariableType(UserString):
    """
    Prevent comparisons elsewhere in the library from using the wrong name.

    Errors are simple assertions because users should not be able to trigger
    them. If that changes, they should be more verbose.

    """

    # TODO we can replace this with typing.Literal on Python 3.8+
    # 定义允许的变量类型，包括"numeric"、"datetime"和"categorical"
    allowed = "numeric", "datetime", "categorical"

    def __init__(self, data):
        # 初始化方法，确保传入的数据在允许的类型列表中
        assert data in self.allowed, data
        super().__init__(data)

    def __eq__(self, other):
        # 等于运算符重载，确保与其他对象比较时，其他对象也在允许的类型列表中
        assert other in self.allowed, other
        return self.data == other


def variable_type(vector, boolean_type="numeric"):
    """
    Determine whether a vector contains numeric, categorical, or datetime data.

    This function differs from the pandas typing API in two ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:`pandas.api.types.CategoricalDtype`.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    boolean_type : 'numeric' or 'categorical'
        Type to use for vectors containing only 0s and 1s (and NAs).

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.
    """
    vector = pd.Series(vector)

    # If a categorical dtype is set, infer categorical
    if isinstance(vector.dtype, pd.CategoricalDtype):
        return VariableType("categorical")

    # Special-case all-na data, which is always "numeric"
    if pd.isna(vector).all():
        return VariableType("numeric")

    # At this point, drop nans to simplify further type inference
    vector = vector.dropna()

    # Special-case binary/boolean data, allow caller to determine
    # This triggers a numpy warning when vector has strings/objects
    # https://github.com/numpy/numpy/issues/6784
    # Because we reduce with .all(), we are agnostic about whether the
    # comparison returns a scalar or vector, so we will ignore the warning.
    # It triggers a separate DeprecationWarning when the vector has datetimes:
    # https://github.com/numpy/numpy/issues/13548
    # This is considered a bug by numpy and will likely go away.
    with warnings.catch_warnings():
        warnings.simplefilter(
            action='ignore', category=(FutureWarning, DeprecationWarning)
        )
        try:
            if np.isin(vector, [0, 1]).all():
                return VariableType(boolean_type)
        except TypeError:
            # .isin comparison is not guaranteed to be possible under NumPy
            # casting rules, depending on the (unknown) dtype of 'vector'
            pass

    # Defer to positive pandas tests
    if pd.api.types.is_numeric_dtype(vector):
        return VariableType("numeric")
    # 如果输入向量的类型是 datetime64 类型，则返回变量类型为 "datetime"
    if pd.api.types.is_datetime64_dtype(vector):
        return VariableType("datetime")

    # --- 如果程序执行到这里，需要检查向量中的元素类型

    # 检查集合中的所有元素是否都是数值类型
    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True

    # 如果向量中所有元素都是数值类型，则返回变量类型为 "numeric"
    if all_numeric(vector):
        return VariableType("numeric")

    # 检查集合中的所有元素是否都是日期时间类型
    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True

    # 如果向量中所有元素都是日期时间类型，则返回变量类型为 "datetime"
    if all_datetime(vector):
        return VariableType("datetime")

    # 否则，最后的回退选项是将变量类型视为 "categorical"
    return VariableType("categorical")
# 确定绘图的方向，基于数据进行判断
def infer_orient(x=None, y=None, orient=None, require_numeric=True):
    """Determine how the plot should be oriented based on the data.

    For historical reasons, the convention is to call a plot "horizontally"
    or "vertically" oriented based on the axis representing its dependent
    variable. Practically, this is used when determining the axis for
    numerical aggregation.

    Parameters
    ----------
    x, y : Vector data or None
        Positional data vectors for the plot.
    orient : string or None
        Specified orientation. If not None, can be "x" or "y", or otherwise
        must start with "v" or "h".
    require_numeric : bool
        If set, raise when the implied dependent variable is not numeric.

    Returns
    -------
    orient : "x" or "y"

    Raises
    ------
    ValueError: When `orient` is an unknown string.
    TypeError: When dependent variable is not numeric, with `require_numeric`

    """

    # 确定变量 x 和 y 的类型
    x_type = None if x is None else variable_type(x)
    y_type = None if y is None else variable_type(y)

    # 非数值依赖变量错误消息模板
    nonnumeric_dv_error = "{} orientation requires numeric `{}` variable."
    # 单变量警告消息模板
    single_var_warning = "{} orientation ignored with only `{}` specified."

    # 如果 x 为 None
    if x is None:
        # 如果 orient 以 "h" 开头，则发出水平方向警告
        if str(orient).startswith("h"):
            warnings.warn(single_var_warning.format("Horizontal", "y"))
        # 如果 require_numeric 为真且 y_type 不是数值型，则引发类型错误
        if require_numeric and y_type != "numeric":
            raise TypeError(nonnumeric_dv_error.format("Vertical", "y"))
        # 返回 "x" 作为方向
        return "x"

    # 如果 y 为 None
    elif y is None:
        # 如果 orient 以 "v" 开头，则发出垂直方向警告
        if str(orient).startswith("v"):
            warnings.warn(single_var_warning.format("Vertical", "x"))
        # 如果 require_numeric 为真且 x_type 不是数值型，则引发类型错误
        if require_numeric and x_type != "numeric":
            raise TypeError(nonnumeric_dv_error.format("Horizontal", "x"))
        # 返回 "y" 作为方向
        return "y"

    # 如果 orient 以 "v" 开头或者 orient 为 "x"
    elif str(orient).startswith("v") or orient == "x":
        # 如果 require_numeric 为真且 y_type 不是数值型，则引发类型错误
        if require_numeric and y_type != "numeric":
            raise TypeError(nonnumeric_dv_error.format("Vertical", "y"))
        # 返回 "x" 作为方向
        return "x"

    # 如果 orient 以 "h" 开头或者 orient 为 "y"
    elif str(orient).startswith("h") or orient == "y":
        # 如果 require_numeric 为真且 x_type 不是数值型，则引发类型错误
        if require_numeric and x_type != "numeric":
            raise TypeError(nonnumeric_dv_error.format("Horizontal", "x"))
        # 返回 "y" 作为方向
        return "y"

    # 如果 orient 不为 None 且不以 'v' 或 'h' 开头
    elif orient is not None:
        err = (
            "`orient` must start with 'v' or 'h' or be None, "
            f"but `{repr(orient)}` was passed."
        )
        # 抛出值错误异常
        raise ValueError(err)

    # 如果 x_type 不是分类型且 y_type 是分类型
    elif x_type != "categorical" and y_type == "categorical":
        # 返回 "y" 作为方向
        return "y"

    # 如果 x_type 不是数值型且 y_type 是数值型
    elif x_type != "numeric" and y_type == "numeric":
        # 返回 "x" 作为方向
        return "x"

    # 如果 x_type 是数值型且 y_type 不是数值型
    elif x_type == "numeric" and y_type != "numeric":
        # 返回 "y" 作为方向
        return "y"

    # 如果 require_numeric 为真且既不是 "x" 变量也不是 "y" 变量为数值型
    elif require_numeric and "numeric" not in (x_type, y_type):
        err = "Neither the `x` nor `y` variable appears to be numeric."
        # 抛出类型错误异常
        raise TypeError(err)

    # 默认返回 "x" 作为方向
    else:
        return "x"
    # dashes 是一个列表，包含了用于 matplotlib.lines.Line2D 类的 `dashes` 参数的有效参数值。第一个规格是实线（""），其余是长短虚线序列。
    
    # 开始用易于区分的虚线规格
    dashes = [
        "",
        (4, 1.5),
        (1, 1),
        (3, 1.25, 1.5, 1.25),
        (5, 1, 1, 1),
    ]
    
    # 现在以编程方式构建所需数量的虚线规格
    p = 3
    while len(dashes) < n:
    
        # 从长短虚线的组合中获取
        a = itertools.combinations_with_replacement([3, 1.25], p)
        b = itertools.combinations_with_replacement([4, 1], p)
    
        # 交错组合，反转其中一个流
        segment_list = itertools.chain(*zip(
            list(a)[1:-1][::-1],
            list(b)[1:-1]
        ))
    
        # 现在插入间隙
        for segments in segment_list:
            gap = min(segments)
            spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
            dashes.append(spec)
    
        p += 1
    
    # 返回最多 n 个虚线规格
    return dashes[:n]
# 构建一个长度可变的唯一标记样式列表，用于表示数据点的标记样式

def unique_markers(n):
    """Build an arbitrarily long list of unique marker styles for points.

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
    # 开始使用一些可以很好区分的标记样式
    markers = [
        "o",                # 圆圈
        "X",                # X形
        (4, 0, 45),         # 自定义的点样式，参数为(marker, rotate, size)
        "P",                # 五角星
        (4, 0, 0),          # 自定义的点样式，参数为(marker, rotate, size)
        (4, 1, 0),          # 自定义的点样式，参数为(marker, rotate, size)
        "^",                # 上三角
        (4, 1, 45),         # 自定义的点样式，参数为(marker, rotate, size)
        "v",                # 下三角
    ]

    # 现在从增加阶数的正多边形中生成更多的标记样式
    s = 5
    while len(markers) < n:
        a = 360 / (s + 1) / 2
        markers.extend([
            (s + 1, 1, a),   # 自定义的点样式，参数为(sides, style, angle)
            (s + 1, 0, a),   # 自定义的点样式，参数为(sides, style, angle)
            (s, 1, 0),       # 自定义的点样式，参数为(sides, style, angle)
            (s, 0, 0),       # 自定义的点样式，参数为(sides, style, angle)
        ])
        s += 1

    # 转换成MarkerStyle对象，仅使用我们需要的部分
    # markers = [mpl.markers.MarkerStyle(m) for m in markers[:n]]

    return markers[:n]


def categorical_order(vector, order=None):
    """Return a list of unique data values.

    Determine an ordered list of levels in ``values``.

    Parameters
    ----------
    vector : list, array, Categorical, or Series
        Vector of "categorical" values
    order : list-like, optional
        Desired order of category levels to override the order determined
        from the ``values`` object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.

    """
    if order is None:
        if hasattr(vector, "categories"):
            order = vector.categories
        else:
            try:
                order = vector.cat.categories
            except (TypeError, AttributeError):
                order = pd.Series(vector).unique()  # 获取唯一的数值

                if variable_type(vector) == "numeric":  # 判断向量的类型是否为数值型
                    order = np.sort(order)  # 对数值型数据进行排序

        order = filter(pd.notnull, order)  # 过滤掉空值

    return list(order)
```