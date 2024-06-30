# `D:\src\scipysrc\seaborn\seaborn\categorical.py`

```
from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial

import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
    desaturate,
    _check_argument,
    _draw_figure,
    _default_color,
    _get_patch_legend_artist,
    _get_transform_functions,
    _scatter_legend_artist,
    _version_predates,
)
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
    EstimateAggregator,
    LetterValues,
    WeightedAggregator,
)
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs


__all__ = [
    "catplot",
    "stripplot", "swarmplot",
    "boxplot", "violinplot", "boxenplot",
    "pointplot", "barplot", "countplot",
]

# 定义一个子类_CategoricalPlotter，继承自VectorPlotter
class _CategoricalPlotter(VectorPlotter):

    # 定义宽格式和扁平格式的数据结构字典
    wide_structure = {"x": "@columns", "y": "@values", "hue": "@columns"}
    flat_structure = {"y": "@values"}

    # 指定图例属性列表
    _legend_attributes = ["color"]

    # 初始化函数，接受多个参数
    def __init__(
        self,
        data=None,
        variables={},
        order=None,
        orient=None,
        require_numeric=False,
        color=None,
        legend="auto",
        
    # 实现向后兼容性，用于处理 hue 参数。

    # force_hue 参数用于在重构期间通过现有测试，并在测试新行为后进行测试。
    # 在工作完成后，可以删除它。

    default_behavior = color is None or palette is not None
    # 检查是否采用了默认行为：颜色为 None 或者调色板不为 None
    # 这是因为旧版本的分类函数默认将调色板应用于分类轴。
    # 现在我们要求显式指定 hue 映射，以更加一致地处理现有情况。
    # 目前没有好的渐进方法来实现这一点。因为它由 hue=None 的默认值触发，
    # 用户总是会收到警告，除非我们引入一些表示此更改的标记“默认”参数。

    if force_hue and "hue" not in self.variables and default_behavior:
        # 如果 force_hue 为 True，并且 self.variables 中没有 "hue"，
        # 且采用了默认行为，则执行以下操作：

        self._redundant_hue = True
        # 设置 _redundant_hue 为 True

        self.plot_data["hue"] = self.plot_data[self.orient]
        # 将 self.orient 列的值赋给 plot_data 中的 "hue" 列

        self.variables["hue"] = self.variables[self.orient]
        # 将 self.orient 列的值赋给 variables 中的 "hue" 列

        self.var_types["hue"] = "categorical"
        # 将 "hue" 列的变量类型标记为 "categorical"

        hue_order = self.var_levels[self.orient]
        # 将 self.orient 列的变量水平赋给 hue_order

        # 由于我们将分类轴变量转换为字符串，
        # 我们需要更新调色板的字典。
        if isinstance(palette, dict):
            palette = {str(k): v for k, v in palette.items()}
            # 将 palette 中的键转换为字符串类型

    else:
        # 否则，如果 self.variables 中包含 "hue" 列，则检查是否多余
        if "hue" in self.variables:
            redundant = (self.plot_data["hue"] == self.plot_data[self.orient]).all()
        else:
            redundant = False
        self._redundant_hue = redundant
        # 将多余的 hue 标记保存到 _redundant_hue 中

    # 以前，分类图使用 color= 参数可以生成调色板的技巧。
    # 因为这是一个显式的参数化，我们将在删除之前的版本中发出警告。
    if "hue" in self.variables and palette is None and color is not None:
        # 如果 self.variables 中包含 "hue" 列，并且 palette 是 None，且 color 不为 None

        if not isinstance(color, str):
            color = mpl.colors.to_hex(color)
            # 将 color 转换为十六进制表示

        palette = f"dark:{color}"
        # 设置 palette 为带有 color 颜色的暗调色板字符串

        msg = (
            "\n\n设置渐变调色板使用 color= 已经弃用，并将在 v0.14.0 版本中移除。"
            f"设置 `palette='{palette}'` 可以获得相同效果。\n"
        )
        warnings.warn(msg, FutureWarning, stacklevel=3)
        # 发出警告，提示用户 color= 参数的设置即将移除

    # 返回更新后的 palette 和 hue_order
    return palette, hue_order
    # 当未指定 'hue' 变量但指定了 'palette' 参数时，提供一种周期，其中 palette= 暗示了当未提供时的 hue=
    def _palette_without_hue_backcompat(self, palette, hue_order):
        if "hue" not in self.variables and palette is not None:
            # 发出警告提示未来版本将移除此用法
            msg = (
                "\n\nPassing `palette` without assigning `hue` is deprecated "
                f"and will be removed in v0.14.0. Assign the `{self.orient}` variable "
                "to `hue` and set `legend=False` for the same effect.\n"
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)

            # 设置 legend=False
            self.legend = False
            # 将 plot_data 中的 self.orient 变量赋值给 'hue'
            self.plot_data["hue"] = self.plot_data[self.orient]
            # 将 self.orient 变量赋值给 variables 中的 'hue'
            self.variables["hue"] = self.variables.get(self.orient)
            # 将 self.orient 变量类型赋值给 var_types 中的 'hue'
            self.var_types["hue"] = self.var_types.get(self.orient)

            # 将 self.orient 变量的级别赋给 hue_order
            hue_order = self.var_levels.get(self.orient)
            # 移除 _var_levels 中的 'hue' 键
            self._var_levels.pop("hue", None)

        return hue_order

    # 当 scale= 或 join= 不再推荐使用时，提供两个循环，其中 scale= 和 join= 仍起作用，但重定向到 kwargs 参数
    def _point_kwargs_backcompat(self, scale, join, kwargs):
        if scale is not deprecated:
            # 根据 scale 参数设置线宽、标记边缘宽度和标记大小
            lw = mpl.rcParams["lines.linewidth"] * 1.8 * scale
            mew = lw * .75
            ms = lw * 2

            # 发出关于 scale 参数将被移除的警告
            msg = (
                "\n\n"
                "The `scale` parameter is deprecated and will be removed in v0.15.0. "
                "You can now control the size of each plot element using matplotlib "
                "`Line2D` parameters (e.g., `linewidth`, `markersize`, etc.)."
                "\n"
            )
            warnings.warn(msg, stacklevel=3)
            # 更新 kwargs 参数中的线宽、标记边缘宽度和标记大小
            kwargs.update(linewidth=lw, markeredgewidth=mew, markersize=ms)

        if join is not deprecated:
            # 发出关于 join 参数将被移除的警告
            msg = (
                "\n\n"
                "The `join` parameter is deprecated and will be removed in v0.15.0."
            )
            if not join:
                # 如果 join=False，则通过 `linestyle='none'` 移除点之间的连线
                msg += (
                    " You can remove the line between points with `linestyle='none'`."
                )
                kwargs.update(linestyle="")
            msg += "\n"
            warnings.warn(msg, stacklevel=3)
    # 提供两轮循环以处理现有签名级别的 err_kws 参数
    def _err_kws_backcompat(self, err_kws, errcolor, errwidth, capsize):
        """Provide two cycles where existing signature-level err_kws are handled."""
        # 处理过时的 err_kws 参数，生成警告信息并建议更新
        def deprecate_err_param(name, key, val):
            if val is deprecated:
                return
            suggest = f"err_kws={{'{key}': {val!r}}}"
            msg = (
                f"\n\nThe `{name}` parameter is deprecated. And will be removed "
                f"in v0.15.0. Pass `{suggest}` instead.\n"
            )
            warnings.warn(msg, FutureWarning, stacklevel=4)
            # 更新 err_kws 字典中的键值对
            err_kws[key] = val

        # 如果 errcolor 参数不为 None，则处理过时的 errcolor 参数
        if errcolor is not None:
            deprecate_err_param("errcolor", "color", errcolor)
        
        # 处理过时的 errwidth 参数
        deprecate_err_param("errwidth", "linewidth", errwidth)

        # 如果 capsize 参数为 None，则更新为 0，并生成相应的警告信息
        if capsize is None:
            capsize = 0
            msg = (
                "\n\nPassing `capsize=None` is deprecated and will be removed "
                "in v0.15.0. Pass `capsize=0` to disable caps.\n"
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)

        # 返回更新后的 err_kws 字典和更新后的 capsize 参数
        return err_kws, capsize

    # 提供两轮循环以处理小提琴图中的尺度参数的后向兼容性
    def _violin_scale_backcompat(self, scale, scale_hue, density_norm, common_norm):
        """Provide two cycles of backcompat for scale kwargs"""
        # 如果 scale 参数不是过时的，则将其赋值给 density_norm，并生成警告信息
        if scale is not deprecated:
            density_norm = scale
            msg = (
                "\n\nThe `scale` parameter has been renamed and will be removed "
                f"in v0.15.0. Pass `density_norm={scale!r}` for the same effect."
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)

        # 如果 scale_hue 参数不是过时的，则将其赋值给 common_norm，并生成警告信息
        if scale_hue is not deprecated:
            common_norm = scale_hue
            msg = (
                "\n\nThe `scale_hue` parameter has been replaced and will be removed "
                f"in v0.15.0. Pass `common_norm={not scale_hue}` for the same effect."
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)

        # 返回更新后的 density_norm 和 common_norm 参数
        return density_norm, common_norm

    # 提供两轮循环以处理小提琴图中的带宽参数的后向兼容性
    def _violin_bw_backcompat(self, bw, bw_method):
        """Provide two cycles of backcompat for violin bandwidth parameterization."""
        # 如果 bw 参数不是过时的，则将其赋值给 bw_method，并生成相应的警告信息
        if bw is not deprecated:
            bw_method = bw
            msg = dedent(f"""\n
                The `bw` parameter is deprecated in favor of `bw_method`/`bw_adjust`.
                Setting `bw_method={bw!r}`, but please see docs for the new parameters
                and update your code. This will become an error in seaborn v0.15.0.
            """)
            warnings.warn(msg, FutureWarning, stacklevel=3)
        
        # 返回更新后的 bw_method 参数
        return bw_method
    def _boxen_scale_backcompat(self, scale, width_method):
        """Provide two cycles of backcompat for scale kwargs"""
        # 如果 scale 参数不是被弃用的状态，则将 width_method 设为 scale
        if scale is not deprecated:
            width_method = scale
            # 构造警告信息，说明 scale 参数已被重命名为 width_method，将在 v0.15 版本移除
            msg = (
                "\n\nThe `scale` parameter has been renamed to `width_method` and "
                f"will be removed in v0.15. Pass `width_method={scale!r}"
            )
            # 如果 scale 参数为 "area"，则需要额外说明结果会有不同
            if scale == "area":
                msg += ", but note that the result for 'area' will appear different."
            else:
                msg += " for the same effect."
            # 发出 FutureWarning 警告，提示即将移除的参数使用
            warnings.warn(msg, FutureWarning, stacklevel=3)

        # 返回更新后的 width_method 参数
        return width_method

    def _complement_color(self, color, base_color, hue_map):
        """Allow a color to be set automatically using a basis of comparison."""
        # 如果 color 参数为 "gray"
        if color == "gray":
            # 发出 FutureWarning 警告，提示从 v0.14.0 开始，"gray" 将默认使用 matplotlib 的定义
            msg = (
                'Use "auto" to set automatic grayscale colors. From v0.14.0, '
                '"gray" will default to matplotlib\'s definition.'
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)
            # 将 color 参数设为 "auto"
            color = "auto"
        # 如果 color 参数为 None 或者是默认值 default
        elif color is None or color is default:
            # 将 color 参数设为 "auto"
            color = "auto"

        # 如果 color 参数不是 "auto"，直接返回 color
        if color != "auto":
            return color

        # 如果 hue_map 的查找表为 None
        if hue_map.lookup_table is None:
            # 如果 base_color 也为 None，则返回 None
            if base_color is None:
                return None
            # 否则，将 base_color 转换为 RGB，并作为 basis 列表的唯一元素
            basis = [mpl.colors.to_rgb(base_color)]
        else:
            # 否则，将 hue_map 的所有值转换为 RGB，并组成 basis 列表
            basis = [mpl.colors.to_rgb(c) for c in hue_map.lookup_table.values()]
        
        # 获取 basis 中所有唯一的 RGB 值
        unique_colors = np.unique(basis, axis=0)
        # 计算这些 RGB 值的亮度，并取最小值的 60% 作为 lum
        light_vals = [rgb_to_hls(*rgb[:3])[1] for rgb in unique_colors]
        lum = min(light_vals) * .6
        # 返回亮度值组成的元组作为颜色
        return (lum, lum, lum)

    def _map_prop_with_hue(self, name, value, fallback, plot_kws):
        """Support pointplot behavior of modifying the marker/linestyle with hue."""
        # 如果 value 参数为默认值 default，则尝试从 plot_kws 中获取 name 对应的值作为 value
        if value is default:
            value = plot_kws.pop(name, fallback)

        # 如果当前对象有 "hue" 变量
        if "hue" in self.variables:
            # 获取 _hue_map 的 levels
            levels = self._hue_map.levels
            # 如果 value 是列表，则将 levels 和 value 一一对应构成 mapping 字典
            if isinstance(value, list):
                mapping = {k: v for k, v in zip(levels, value)}
            else:
                # 否则，将所有 levels 映射到同一个 value 值构成 mapping 字典
                mapping = {k: value for k in levels}
        else:
            # 如果没有 "hue" 变量，则将 value 映射为 None
            mapping = {None: value}

        # 返回构建好的 mapping 字典
        return mapping
    def _adjust_cat_axis(self, ax, axis):
        """Set ticks and limits for a categorical variable."""
        # 如果当前轴不是分类变量，则直接返回，不进行任何调整
        if self.var_types[axis] != "categorical":
            return

        # 如果当前轴上的数据为空，则不进行任何设置，因为无法确定如何设置分类轴
        if self.plot_data[axis].empty:
            return

        # 通过获取当前轴的刻度数量来推断总的分类数目，这依赖于 matplotlib 的实现细节
        n = len(getattr(ax, f"get_{axis}ticks")())

        # 根据轴的类型进行不同的设置
        if axis == "x":
            # 对 x 轴进行设置：关闭 x 轴的网格，设置 x 轴的限制范围
            ax.xaxis.grid(False)
            ax.set_xlim(-.5, n - .5, auto=None)
        else:
            # 对 y 轴进行设置：关闭 y 轴的网格，设置 y 轴的限制范围
            # 注意限制范围对应于之前反转的 y 轴
            ax.yaxis.grid(False)
            ax.set_ylim(n - .5, -.5, auto=None)

    def _dodge_needed(self):
        """Return True when use of `hue` would cause overlaps."""
        # 确定是否需要进行 dodge 变换，即检查是否存在 hue 变量且可能导致重叠
        groupers = list({self.orient, "col", "row"} & set(self.variables))
        if "hue" in self.variables:
            orient = self.plot_data[groupers].value_counts()
            paired = self.plot_data[[*groupers, "hue"]].value_counts()
            return orient.size != paired.size
        return False

    def _dodge(self, keys, data):
        """Apply a dodge transform to coordinates in place."""
        # 如果没有设置 hue 变量，则直接返回，不进行 dodge 变换
        if "hue" not in self.variables:
            return
        
        # 计算 hue 变量的索引和总数
        hue_idx = self._hue_map.levels.index(keys["hue"])
        n = len(self._hue_map.levels)

        # 调整数据的宽度，以适应 dodge 变换
        data["width"] /= n

        # 计算偏移量，以应用 dodge 变换
        full_width = data["width"] * n
        offset = data["width"] * hue_idx + data["width"] / 2 - full_width / 2
        data[self.orient] += offset
    # 从图形对象中反向缩放数据，以确保数据在绘图时显示正确
    def _invert_scale(self, ax, data, vars=("x", "y")):
        """Undo scaling after computation so data are plotted correctly."""
        for var in vars:
            # 获取反向转换函数
            _, inv = _get_transform_functions(ax, var[0])
            # 如果变量是指定的方向并且数据中包含宽度信息
            if var == self.orient and "width" in data:
                # 计算一半宽度
                hw = data["width"] / 2
                # 反向转换计算边缘位置和宽度
                data["edge"] = inv(data[var] - hw)
                data["width"] = inv(data[var] + hw) - data["edge"].to_numpy()
            # 对于每个变量后缀（空字符串，"min"，"max"）
            for suf in ["", "min", "max"]:
                # 如果数据中存在该列
                if (col := f"{var}{suf}") in data:
                    # 对数据进行反向转换
                    data[col] = inv(data[col])

    # 配置图例的显示
    def _configure_legend(self, ax, func, common_kws=None, semantic_kws=None):
        # 根据设定判断是否显示图例
        if self.legend == "auto":
            show_legend = not self._redundant_hue and self.input_format != "wide"
        else:
            show_legend = bool(self.legend)
        # 如果需要显示图例
        if show_legend:
            # 添加图例数据
            self.add_legend_data(ax, func, common_kws, semantic_kws=semantic_kws)
            # 获取图例的句柄和标签
            handles, _ = ax.get_legend_handles_labels()
            # 如果存在图例句柄，则设置图例标题
            if handles:
                ax.legend(title=self.legend_title)

    # 返回原生数值范围上的类别间宽度单位
    @property
    def _native_width(self):
        """Return unit of width separating categories on native numeric scale."""
        # 如果指定方向的变量类型是分类的
        if self.var_types[self.orient] == "categorical":
            return 1

        # 否则，定义宽度为观察间最小间距
        unique_values = np.unique(self.comp_data[self.orient])
        if len(unique_values) > 1:
            native_width = np.nanmin(np.diff(unique_values))
        else:
            native_width = 1
        return native_width

    # 返回为堆叠图生成每个色调级别的偏移量
    def _nested_offsets(self, width, dodge):
        """Return offsets for each hue level for dodged plots."""
        offsets = None
        # 如果变量中包含色调，并且色调映射的级别不为None
        if "hue" in self.variables and self._hue_map.levels is not None:
            n_levels = len(self._hue_map.levels)
            # 如果需要躲避
            if dodge:
                each_width = width / n_levels
                # 生成偏移量数组
                offsets = np.linspace(0, width - each_width, n_levels)
                offsets -= offsets.mean()
            else:
                offsets = np.zeros(n_levels)
        return offsets

    # 注意，这里的绘图方法（在大多数情况下）旨在生成与代码原始版本（0.12之前）完全相同的艺术家，
    # 因此可能存在某些在此上下文中看起来不干净或不合理的奇怪行为，比如为没有观察的变量组合添加空艺术家
    # 计算图形宽度的80%作为条形图的宽度
    width = .8 * self._native_width
    # 使用 _nested_offsets 方法计算条形图的偏移量
    offsets = self._nested_offsets(width, dodge)
    
    # 如果 jitter 参数为 True，则设置 jlim 为 0.1，否则转换 jitter 参数为浮点数赋给 jlim
    if jitter is True:
        jlim = 0.1
    else:
        jlim = float(jitter)
        
    # 如果图形中包含 "hue" 变量，且 dodge 参数为 True，并且 _hue_map.levels 不为空，则调整 jlim 的值
    if "hue" in self.variables and dodge and self._hue_map.levels is not None:
        jlim /= len(self._hue_map.levels)
    jlim *= self._native_width
    
    # 创建一个 partial 函数 jitterer，用于生成均匀分布的随机数，范围在 -jlim 到 +jlim 之间
    jitterer = partial(np.random.uniform, low=-jlim, high=+jlim)
    
    # 初始化 iter_vars 列表，包含 self.orient 变量
    iter_vars = [self.orient]
    # 如果 dodge 为 True，则添加 "hue" 变量到 iter_vars 列表中
    if dodge:
        iter_vars.append("hue")
    
    # 获取当前的绘图区域对象
    ax = self.ax
    # 初始化 dodge_move 和 jitter_move 变量
    dodge_move = jitter_move = 0
    
    # 如果 plot_kws 中包含 "marker" 键且其对应的 MarkerStyle 不是填充样式，则移除 "edgecolor" 键
    if "marker" in plot_kws and not MarkerStyle(plot_kws["marker"]).is_filled():
        plot_kws.pop("edgecolor", None)
    
    # 遍历数据集和变量，使用 iter_data 方法进行迭代
    for sub_vars, sub_data in self.iter_data(iter_vars,
                                             from_comp_data=True,
                                             allow_empty=True):
    
        # 获取当前子图形的绘图区域对象
        ax = self._get_axes(sub_vars)
    
        # 如果 offsets 不为 None 且有非零元素，则根据 sub_data["hue"] 的值映射 dodge_move
        if offsets is not None and (offsets != 0).any():
            dodge_move = offsets[sub_data["hue"].map(self._hue_map.levels.index)]
    
        # 根据 jitterer 函数生成长度与 sub_data 相同的 jitter_move 数组（如果 sub_data 的长度大于 1）
        jitter_move = jitterer(size=len(sub_data)) if len(sub_data) > 1 else 0
    
        # 调整数据，将 sub_data[self.orient] 加上 dodge_move 和 jitter_move
        adjusted_data = sub_data[self.orient] + dodge_move + jitter_move
        sub_data[self.orient] = adjusted_data
        
        # 将调整后的数据反向缩放，并应用到绘图区域中
        self._invert_scale(ax, sub_data)
    
        # 使用 scatter 方法在绘图区域上绘制散点图，颜色为 color，其他参数使用 plot_kws 指定
        points = ax.scatter(sub_data["x"], sub_data["y"], color=color, **plot_kws)
        
        # 如果数据集中包含 "hue" 变量，则设置散点图的填充颜色为 _hue_map 对 sub_data["hue"] 的映射结果
        if "hue" in self.variables:
            points.set_facecolors(self._hue_map(sub_data["hue"]))
    
    # 配置图例样式和属性，使用 _configure_legend 方法
    self._configure_legend(ax, _scatter_legend_artist, common_kws=plot_kws)
        ):

            # 计算图形的宽度，基于原生宽度的百分之八十
            width = .8 * self._native_width
            # 调用函数计算嵌套偏移量
            offsets = self._nested_offsets(width, dodge)

            iter_vars = [self.orient]
            # 如果 dodge 为 True，添加 "hue" 到迭代变量列表中
            if dodge:
                iter_vars.append("hue")

            ax = self.ax
            # 初始化空字典，用于存储点集合
            point_collections = {}
            dodge_move = 0

            # 检查 plot_kws 中是否存在 "marker"，并且它不是填充的 MarkerStyle
            if "marker" in plot_kws and not MarkerStyle(plot_kws["marker"]).is_filled():
                # 如果是空心标记，则移除 plot_kws 中的 "edgecolor"
                plot_kws.pop("edgecolor", None)

            # 遍历数据和变量，从 iter_data 方法中获取
            for sub_vars, sub_data in self.iter_data(iter_vars,
                                                     from_comp_data=True,
                                                     allow_empty=True):

                # 根据子变量获取对应的轴
                ax = self._get_axes(sub_vars)

                # 如果存在偏移量，根据 sub_data["hue"] 的映射调整 dodge_move
                if offsets is not None:
                    dodge_move = offsets[sub_data["hue"].map(self._hue_map.levels.index)]

                # 如果子数据不为空，则根据 orient 调整子数据中的坐标值
                if not sub_data.empty:
                    sub_data[self.orient] = sub_data[self.orient] + dodge_move

                # 根据数据反转比例尺
                self._invert_scale(ax, sub_data)

                # 绘制散点图，使用给定的颜色和 plot_kws
                points = ax.scatter(sub_data["x"], sub_data["y"], color=color, **plot_kws)
                # 如果变量中包含 "hue"，设置点的面颜色为对应的 hue 颜色
                if "hue" in self.variables:
                    points.set_facecolors(self._hue_map(sub_data["hue"]))

                # 如果子数据不为空，将点集合存储到 point_collections 中
                if not sub_data.empty:
                    point_collections[(ax, sub_data[self.orient].iloc[0])] = points

            # 创建 Beeswarm 对象
            beeswarm = Beeswarm(width=width, orient=self.orient, warn_thresh=warn_thresh)
            # 遍历 point_collections 中的项，绘制每个点集合
            for (ax, center), points in point_collections.items():
                # 如果点的偏移量大于 1
                if points.get_offsets().shape[0] > 1:

                    # 定义绘制函数，包含 points 和 center 作为参数
                    def draw(points, renderer, *, center=center):

                        # 使用 beeswarm 对象处理 points 和 center
                        beeswarm(points, center)

                        # 根据 orient 调整 scalex 和 scaley
                        if self.orient == "y":
                            scalex = False
                            scaley = ax.get_autoscaley_on()
                        else:
                            scalex = ax.get_autoscalex_on()
                            scaley = False

                        # 防止撤销在 _adjust_cat_axis 中设置的漂亮的分类轴限制
                        # 因为该方法当前保留了自动缩放标志。可能最好在那里禁用自动缩放以避免这样做。
                        fixed_scale = self.var_types[self.orient] == "categorical"
                        # 更新数据限制
                        ax.update_datalim(points.get_datalim(ax.transData))
                        # 如果未固定缩放且 scalex 或 scaley 为 True，则自动调整视图
                        if not fixed_scale and (scalex or scaley):
                            ax.autoscale_view(scalex=scalex, scaley=scaley)

                        # 调用父类的 draw 方法来绘制点
                        super(points.__class__, points).draw(renderer)

                    # 将 draw 方法绑定到 points 的 draw 属性上
                    points.draw = draw.__get__(points)

            # 调用 _draw_figure 方法来绘制图形
            _draw_figure(ax.figure)
            # 调用 _configure_legend 方法来配置图例
            self._configure_legend(ax, _scatter_legend_artist, plot_kws)

        # 定义 plot_boxes 方法，接受多个参数用于绘制箱线图
        def plot_boxes(
            self,
            width,
            dodge,
            gap,
            fill,
            whis,
            color,
            linecolor,
            linewidth,
            fliersize,
            plot_kws,  # TODO rename user_kws?
    # 绘制带分布盒子图的方法
    def plot_boxens(
        self,
        width,
        dodge,
        gap,
        fill,
        color,
        linecolor,
        linewidth,
        width_method,
        k_depth,
        outlier_prop,
        trust_alpha,
        showfliers,
        box_kws,
        flier_kws,
        line_kws,
        plot_kws,
    ):
        # 进行绘图的各种设置和参数
        ...
    
    
    # 绘制小提琴图的方法
    def plot_violins(
        self,
        width,
        dodge,
        gap,
        split,
        color,
        fill,
        linecolor,
        linewidth,
        inner,
        density_norm,
        common_norm,
        kde_kws,
        inner_kws,
        plot_kws,
    ):
        # 设置小提琴图的参数和样式
        ...
    
    
    # 绘制散点图的方法
    def plot_points(
        self,
        aggregator,
        markers,
        linestyles,
        dodge,
        color,
        capsize,
        err_kws,
        plot_kws,
    ):
        # 定义并设置散点图的特定参数
        ...
    
    
    # 绘制条形图的方法
    def plot_bars(
        self,
        aggregator,
        dodge,
        gap,
        width,
        fill,
        color,
        capsize,
        err_kws,
        plot_kws,
    ):
        # 设置条形图的参数和样式
        ...
        agg_var = {"x": "y", "y": "x"}[self.orient]
        # 根据 self.orient 确定 agg_var 的值，用于后续数据聚合操作

        iter_vars = ["hue"]
        # 指定迭代变量列表，这里仅包含 "hue"

        ax = self.ax
        # 将 self.ax 赋值给 ax，即当前绘图所用的轴对象

        if self._hue_map.levels is None:
            dodge = False
        # 如果 self._hue_map.levels 为 None，则设置 dodge 为 False，否则 dodge 保持不变

        if dodge and capsize is not None:
            capsize = capsize / len(self._hue_map.levels)
        # 如果 dodge 为 True 并且 capsize 不为 None，则更新 capsize 的值为 capsize 除以 self._hue_map.levels 的长度

        if not fill:
            plot_kws.setdefault("linewidth", 1.5 * mpl.rcParams["lines.linewidth"])
        # 如果 fill 为 False，则将 plot_kws 中 "linewidth" 的默认值设置为当前线宽的 1.5 倍

        err_kws.setdefault("linewidth", 1.5 * mpl.rcParams["lines.linewidth"])
        # 设置 err_kws 中 "linewidth" 的默认值为当前线宽的 1.5 倍

        for sub_vars, sub_data in self.iter_data(iter_vars,
                                                 from_comp_data=True,
                                                 allow_empty=True):
            # 迭代 self.iter_data 方法返回的子变量和子数据，允许为空

            ax = self._get_axes(sub_vars)
            # 根据子变量 sub_vars 获取相应的轴对象，赋值给 ax

            agg_data = sub_data if sub_data.empty else (
                sub_data
                .groupby(self.orient)
                .apply(aggregator, agg_var, **groupby_apply_include_groups(False))
                .reset_index()
            )
            # 如果 sub_data 非空，则对其按 self.orient 进行分组并应用 aggregator 函数，将结果重置索引后赋给 agg_data

            agg_data["width"] = width * self._native_width
            # 计算每个条形图的宽度，乘以 self._native_width

            if dodge:
                self._dodge(sub_vars, agg_data)
                # 如果 dodge 为 True，则调用 self._dodge 方法对 agg_data 进行处理

            if gap:
                agg_data["width"] *= 1 - gap
                # 如果 gap 不为 0，则减小 agg_data 的宽度

            agg_data["edge"] = agg_data[self.orient] - agg_data["width"] / 2
            # 计算每个条形图的边缘位置，考虑到条形图的宽度

            self._invert_scale(ax, agg_data)
            # 调用 self._invert_scale 方法，根据数据调整轴的缩放比例

            if self.orient == "x":
                bar_func = ax.bar
                kws = dict(
                    x=agg_data["edge"], height=agg_data["y"], width=agg_data["width"]
                )
            else:
                bar_func = ax.barh
                kws = dict(
                    y=agg_data["edge"], width=agg_data["x"], height=agg_data["width"]
                )
            # 根据 self.orient 选择绘制条形图还是水平条形图，并设置相应的参数字典 kws

            main_color = self._hue_map(sub_vars["hue"]) if "hue" in sub_vars else color
            # 如果 sub_vars 中包含 "hue"，则使用 self._hue_map 方法获取对应颜色，否则使用 color

            # 设置 color 和 facecolor，用于属性循环逻辑
            kws["align"] = "edge"
            if fill:
                kws.update(color=main_color, facecolor=main_color)
            else:
                kws.update(color=main_color, edgecolor=main_color, facecolor="none")
            # 根据 fill 的值更新 kws 中的 color 和 facecolor 或 edgecolor

            bar_func(**{**kws, **plot_kws})
            # 调用 bar_func 方法绘制条形图，传入 kws 和 plot_kws 参数

            if aggregator.error_method is not None:
                self.plot_errorbars(
                    ax, agg_data, capsize,
                    {"color": ".26" if fill else main_color, **err_kws}
                )
            # 如果 aggregator.error_method 不为 None，则调用 self.plot_errorbars 方法绘制误差条

        legend_artist = _get_patch_legend_artist(fill)
        # 调用 _get_patch_legend_artist 方法获取图例艺术家对象

        self._configure_legend(ax, legend_artist, plot_kws)
        # 调用 self._configure_legend 方法配置图例
    # 定义一个方法用于在给定的坐标轴上绘制带有误差条的数据图形
    def plot_errorbars(self, ax, data, capsize, err_kws):
        # 确定变量和对应的变量名，根据方向设置不同的变量
        var = {"x": "y", "y": "x"}[self.orient]
        # 遍历数据中的每一行，将每行转换为字典格式
        for row in data.to_dict("records"):
            row = dict(row)
            # 获取位置和值的数组
            pos = np.array([row[self.orient], row[self.orient]])
            val = np.array([row[f"{var}min"], row[f"{var}max"]])
            
            # 如果设置了误差条大小
            if capsize:
                # 计算误差条的半宽度，考虑图形的本地宽度
                cw = capsize * self._native_width / 2
                # 获取坐标轴的转换函数
                scl, inv = _get_transform_functions(ax, self.orient)
                # 计算误差条的起止位置
                cap = inv(scl(pos[0]) - cw), inv(scl(pos[1]) + cw)
                
                # 扩展位置数组以包含误差条的起止位置
                pos = np.concatenate([
                    [*cap, np.nan], pos, [np.nan, *cap]
                ])
                # 扩展值数组以包含误差条对应的值
                val = np.concatenate([
                    [val[0], val[0], np.nan], val, [np.nan, val[-1], val[-1]],
                ])
            
            # 根据方向不同选择合适的参数顺序，并将数据绘制到坐标轴上
            if self.orient == "x":
                args = pos, val
            else:
                args = val, pos
            ax.plot(*args, **err_kws)
class _CategoricalAggPlotter(_CategoricalPlotter):
    # 定义类变量，指定了绘图时的数据结构，包含 x 和 y 的映射关系
    flat_structure = {"x": "@index", "y": "@values"}

_categorical_docs = dict(
    # 共享的文档字符串
    categorical_narrative=dedent("""\
    See the :ref:`tutorial <categorical_tutorial>` for more information.

    .. note::
        By default, this function treats one of the variables as categorical
        and draws data at ordinal positions (0, 1, ... n) on the relevant axis.
        As of version 0.13.0, this can be disabled by setting `native_scale=True`.
    """),

    # 共享函数参数说明
    input_params=dedent("""\
    x, y, hue : names of variables in `data` or vector data
        Inputs for plotting long-form data. See examples for interpretation.\
    """),
    categorical_data=dedent("""\
    data : DataFrame, Series, dict, array, or list of arrays
        Dataset for plotting. If `x` and `y` are absent, this is
        interpreted as wide-form. Otherwise it is expected to be long-form.\
    """),
    order_vars=dedent("""\
    order, hue_order : lists of strings
        Order to plot the categorical levels in; otherwise the levels are
        inferred from the data objects.\
    """),
    stat_api_params=dedent("""\
    estimator : string or callable that maps vector -> scalar
        Statistical function to estimate within each categorical bin.
    errorbar : string, (string, number) tuple, callable or None
        Name of errorbar method (either "ci", "pi", "se", or "sd"), or a tuple
        with a method name and a level parameter, or a function that maps from a
        vector to a (min, max) interval, or None to hide errorbar. See the
        :doc:`errorbar tutorial </tutorial/error_bars>` for more information.

        .. versionadded:: v0.12.0
    n_boot : int
        Number of bootstrap samples used to compute confidence intervals.
    seed : int, `numpy.random.Generator`, or `numpy.random.RandomState`
        Seed or random number generator for reproducible bootstrapping.
    units : name of variable in `data` or vector data
        Identifier of sampling units; used by the errorbar function to
        perform a multilevel bootstrap and account for repeated measures
    weights : name of variable in `data` or vector data
        Data values or column used to compute weighted statistics.
        Note that the use of weights may limit other statistical options.

        .. versionadded:: v0.13.1\
    """),
    ci=dedent("""\
    ci : float
        Level of the confidence interval to show, in [0, 100].

        .. deprecated:: v0.12.0
            Use `errorbar=("ci", ...)`.\
    """),
    orient=dedent("""\
    orient : "v" | "h" | "x" | "y"
        Orientation of the plot (vertical or horizontal). This is usually
        inferred based on the type of the input variables, but it can be used
        to resolve ambiguity when both `x` and `y` are numeric or when
        plotting wide-form data.

        .. versionchanged:: v0.13.0
            Added 'x'/'y' as options, equivalent to 'v'/'h'.
    """),
    # 图表的方向设置，可以是垂直 ("v") 或水平 ("h")。通常根据输入变量的类型推断，
    # 但在 x 和 y 都是数值或绘制宽格式数据时可能需要手动指定。
    # 
    # .. versionchanged:: v0.13.0
    #     添加 'x'/'y' 选项，相当于 'v'/'h'。
    color=dedent("""\
    color : matplotlib color
        Single color for the elements in the plot.
    """),
    # 绘图元素的单一颜色。
    palette=dedent("""\
    palette : palette name, list, dict, or :class:`matplotlib.colors.Colormap`
        Color palette that maps the hue variable. If the palette is a dictionary,
        keys should be names of levels and values should be matplotlib colors.
        The type/value will sometimes force a qualitative/quantitative mapping.
    """),
    # 映射色调变量的调色板。如果调色板是一个字典，则键应为级别名称，值应为 matplotlib 的颜色。
    # 类型/值有时会强制进行定性/定量映射。
    hue_norm=dedent("""\
    hue_norm : tuple or :class:`matplotlib.colors.Normalize` object
        Normalization in data units for colormap applied to the `hue`
        variable when it is numeric. Not relevant if `hue` is categorical.

        .. versionadded:: v0.12.0
    """),
    # 当色调变量是数值时，对应色彩映射的数据单位标准化。如果色调是分类的，则此选项无关紧要。
    # 
    # .. versionadded:: v0.12.0
    saturation=dedent("""\
    saturation : float
        Proportion of the original saturation to draw fill colors in. Large
        patches often look better with desaturated colors, but set this to
        `1` if you want the colors to perfectly match the input values.
    """),
    # 用于绘制填充颜色的原始饱和度比例。通常，大的区块使用去饱和颜色效果更好，但如果希望颜色与输入值完全匹配，
    # 可将此值设置为 `1`。
    capsize=dedent("""\
    capsize : float
        Width of the "caps" on error bars, relative to bar spacing.
    """),
    # 误差线上的“帽子”宽度，相对于条形间距。
    errcolor=dedent("""\
    errcolor : matplotlib color
        Color used for the error bar lines.

        .. deprecated:: 0.13.0
            Use `err_kws={'color': ...}`.
    """),
    # 用于误差条线的颜色。
    # 
    # .. deprecated:: 0.13.0
    #     使用 `err_kws={'color': ...}`。
    errwidth=dedent("""\
    errwidth : float
        Thickness of error bar lines (and caps), in points.

        .. deprecated:: 0.13.0
            Use `err_kws={'linewidth': ...}`.
    """),
    # 误差条线（及其“帽子”）的线条粗细，以点为单位。
    # 
    # .. deprecated:: 0.13.0
    #     使用 `err_kws={'linewidth': ...}`。
    fill=dedent("""\
    fill : bool
        If True, use a solid patch. Otherwise, draw as line art.

        .. versionadded:: v0.13.0
    """),
    # 如果为 True，则使用实心图块。否则，绘制为线条艺术。
    # 
    # .. versionadded:: v0.13.0
    gap=dedent("""\
    gap : float
        Shrink on the orient axis by this factor to add a gap between dodged elements.

        .. versionadded:: 0.13.0
    """),
    # 在 orient 轴上缩小的因子，以在避让元素之间添加间隙。
    # 
    # .. versionadded:: 0.13.0
    width=dedent("""\
    width : float
        Width allotted to each element on the orient axis. When `native_scale=True`,
        it is relative to the minimum distance between two values in the native scale.
    """),
    # 在 orient 轴上分配给每个元素的宽度。当 `native_scale=True` 时，它相对于本机比例中两个值之间的最小距离。
    dodge=dedent("""\
    dodge : "auto" or bool
        When hue mapping is used, whether elements should be narrowed and shifted along
        the orient axis to eliminate overlap. If `"auto"`, set to `True` when the
        orient variable is crossed with the categorical variable or `False` otherwise.

        .. versionchanged:: 0.13.0
            Added `"auto"` mode as a new default.
    """),
    # 当使用色调映射时，是否应缩窄元素并沿 orient 轴移动以消除重叠。
    # 如果是 `"auto"`，当 orient 变量与分类变量交叉时设置为 `True`，否则设置为 `False`。
    # 
    # .. versionchanged:: 0.13.0
    #     添加 `"auto"` 模式作为新的默认设置。
    linewidth=dedent("""\
    linewidth : float
        Width of the lines that frame the plot elements.
    """),
    # 框架绘图元素的线条宽度。
    linecolor=dedent("""\
    linecolor : color
        Color to use for line elements, when `fill` is True.

        .. versionadded:: v0.13.0\
    """),
    # `linecolor`参数：用于线条元素的颜色，当`fill`为True时生效。

    log_scale=dedent("""\
    log_scale : bool or number, or pair of bools or numbers
        Set axis scale(s) to log. A single value sets the data axis for any numeric
        axes in the plot. A pair of values sets each axis independently.
        Numeric values are interpreted as the desired base (default 10).
        When `None` or `False`, seaborn defers to the existing Axes scale.

        .. versionadded:: v0.13.0\
    """),
    # `log_scale`参数：设置轴的对数尺度。可以是布尔值、数字或布尔值或数字的对。单个值为所有数值轴设置对数尺度，一对值可以分别设置每个轴。数值被解释为所需的基数（默认为10）。当为`None`或`False`时，seaborn会使用现有的轴尺度。

    native_scale=dedent("""\
    native_scale : bool
        When True, numeric or datetime values on the categorical axis will maintain
        their original scaling rather than being converted to fixed indices.

        .. versionadded:: v0.13.0\
    """),
    # `native_scale`参数：当为True时，分类轴上的数值或日期时间值将保持其原始缩放，而不会转换为固定索引。

    formatter=dedent("""\
    formatter : callable
        Function for converting categorical data into strings. Affects both grouping
        and tick labels.

        .. versionadded:: v0.13.0\
    """),
    # `formatter`参数：用于将分类数据转换为字符串的函数。影响分组和刻度标签。

    legend=dedent("""\
    legend : "auto", "brief", "full", or False
        How to draw the legend. If "brief", numeric `hue` and `size`
        variables will be represented with a sample of evenly spaced values.
        If "full", every group will get an entry in the legend. If "auto",
        choose between brief or full representation based on number of levels.
        If `False`, no legend data is added and no legend is drawn.

        .. versionadded:: v0.13.0\
    """),
    # `legend`参数：如何绘制图例。如果为"brief"，数值型的`hue`和`size`变量将用均匀间隔的值样本表示。如果为"full"，每个组都会在图例中显示条目。如果为"auto"，根据级别数量自动选择brief或full表示。如果为`False`，不添加图例数据，也不绘制图例。

    err_kws=dedent("""\
    err_kws : dict
        Parameters of :class:`matplotlib.lines.Line2D`, for the error bar artists.

        .. versionadded:: v0.13.0\
    """),
    # `err_kws`参数：用于误差条线条艺术家的:class:`matplotlib.lines.Line2D`的参数。

    ax_in=dedent("""\
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.\
    """),
    # `ax_in`参数：matplotlib的Axes对象，用于绘制图表，否则使用当前的Axes对象。

    ax_out=dedent("""\
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.\
    """),
    # `ax_out`参数：返回带有绘制图表的Axes对象。

    # Shared see also
    boxplot=dedent("""\
    boxplot : A traditional box-and-whisker plot with a similar API.\
    """),
    # `boxplot`：传统的箱线图，具有类似的API。

    violinplot=dedent("""\
    violinplot : A combination of boxplot and kernel density estimation.\
    """),
    # `violinplot`：箱线图和核密度估计的组合。

    stripplot=dedent("""\
    stripplot : A scatterplot where one variable is categorical. Can be used
                in conjunction with other plots to show each observation.\
    """),
    # `stripplot`：散点图，其中一个变量是分类变量。可以与其他图表结合使用，显示每个观察值。

    swarmplot=dedent("""\
    swarmplot : A categorical scatterplot where the points do not overlap. Can
                be used with other plots to show each observation.\
    """),
    # `swarmplot`：分类散点图，点不重叠。可以与其他图表结合使用，显示每个观察值。

    barplot=dedent("""\
    barplot : Show point estimates and confidence intervals using bars.\
    """),
    # `barplot`：使用条形图显示点估计和置信区间。

    countplot=dedent("""\
    countplot : Show the counts of observations in each categorical bin.\
    """),
    # `countplot`：显示每个分类箱中观察值的计数。

    pointplot=dedent("""\
    pointplot : Show point estimates and confidence intervals using dots.\
    """),
    # `pointplot`：使用点显示点估计和置信区间。

    catplot=dedent("""\
    catplot : Combine a categorical plot with a :class:`FacetGrid`.\
    """)
    # `catplot`：将分类图表与:class:`FacetGrid`组合。
    """),
    # `boxenplot` 参数：增强版箱线图，适用于大数据集。
    boxenplot=dedent("""\
    boxenplot : An enhanced boxplot for larger datasets.\
    """),
# 导入内部文档和示例变量以补充分类绘图方法的文档字符串
boxplot.__doc__ = dedent("""\
    Draw a box plot to show distributions with respect to categories.

    A box plot (or box-and-whisker plot) shows the distribution of quantitative
    data in a way that facilitates comparisons between variables or across
    levels of a categorical variable. The box shows the quartiles of the
    dataset while the whiskers extend to show the rest of the distribution,
    except for points that are determined to be "outliers" using a method
    that is a function of the inter-quartile range.

    {categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    {orient}
    {color}
    {palette}
    {saturation}
    {fill}
    {dodge}
    {width}
    {gap}
    whis : float or pair of floats
        Paramater that controls whisker length. If scalar, whiskers are drawn
        to the farthest datapoint within *whis * IQR* from the nearest hinge.
        If a tuple, it is interpreted as percentiles that whiskers represent.
    {linecolor}
    {linewidth}
""")
    fliersize : float
        # 定义异常值标记的大小，用于指示离群观察值的大小。

    {hue_norm}
        # 此处包含一个占位符，可能用于指定颜色映射的归一化对象。

    {log_scale}
        # 此处包含一个占位符，可能用于指定是否应用对数刻度。

    {native_scale}
        # 此处包含一个占位符，可能用于指定是否使用原生刻度。

    {formatter}
        # 此处包含一个占位符，可能用于指定数据格式化的函数或对象。

    {legend}
        # 此处包含一个占位符，可能用于指定图例对象或配置。

    {ax_in}
        # 此处包含一个占位符，可能用于指定输入的图轴对象。

    kwargs : key, value mappings
        # 其他关键字参数将传递给 matplotlib.axes.Axes.boxplot 方法，
        # 用于绘制箱线图。

    Returns
    -------
    {ax_out}
        # 返回处理后的图轴对象，通常包含已绘制的箱线图。

    See Also
    --------
    {violinplot}
        # 参见小提琴图的文档，可能用于类似功能的比较。

    {stripplot}
        # 参见条形图的文档，可能用于展示单个观察值分布。

    {swarmplot}
        # 参见蜂群图的文档，可能用于显示分布的观察值。

    {catplot}
        # 参见 catplot 的文档，可能是更高级的分类绘图函数。

    Examples
    --------
    .. include:: ../docstrings/boxplot.rst
        # 包含外部链接的例子，展示如何使用此函数的实际示例。

    """).format(**_categorical_docs)
        # 使用 _categorical_docs 字典格式化字符串，填充占位符，返回注释后的字符串。
# 定义一个绘制小提琴图的函数
def violinplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    orient=None, color=None, palette=None, saturation=.75, fill=True,
    inner="box", split=False, width=.8, dodge="auto", gap=0,
    linewidth=None, linecolor="auto", cut=2, gridsize=100,
    bw_method="scott", bw_adjust=1, density_norm="area", common_norm=False,
    hue_norm=None, formatter=None, log_scale=None, native_scale=False,
    legend="auto", scale=deprecated, scale_hue=deprecated, bw=deprecated,
    inner_kws=None, ax=None, **kwargs,
):

    # 创建一个 _CategoricalPlotter 对象，用于处理分类数据的绘图
    p = _CategoricalPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue),
        order=order,
        orient=orient,
        color=color,
        legend=legend,
    )

    # 如果未指定绘图坐标轴，则使用当前的坐标轴
    if ax is None:
        ax = plt.gca()

    # 如果绘图数据为空，则直接返回坐标轴对象
    if p.plot_data.empty:
        return ax

    # 如果 dodge 参数为 "auto"，则检查是否需要进行 dodge 操作
    if dodge == "auto":
        # 在 scale_categorical 修改坐标序列类型之前执行 dodge 操作
        dodge = p._dodge_needed()

    # 如果数据的方向是分类的或者不使用原生缩放，则调整分类数据的缩放和格式化
    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    # 将绘图对象 p 附加到坐标轴 ax 上，并可能设置对数坐标轴
    p._attach(ax, log_scale=log_scale)

    # 在 v0.14.0 版本中要移除的过时警告
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    # 如果 fill 为 True，则 saturation 为指定值，否则 saturation 设为 1
    saturation = saturation if fill else 1

    # 根据调色板 palette、色调顺序 hue_order、色调规范 hue_norm 和饱和度 saturation 映射色调
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)

    # 如果未指定 color 参数，则根据 kwargs 中的相关项为 ax.fill_between 选择默认颜色
    color = _default_color(
        ax.fill_between, hue, color,
        {k: v for k, v in kwargs.items() if k in ["c", "color", "fc", "facecolor"]},
        saturation=saturation,
    )

    # 根据 color 和 hue_map 选择补充色 linecolor
    linecolor = p._complement_color(linecolor, color, p._hue_map)

    # 将 density_norm 和 common_norm 进行过时警告后向兼容处理
    density_norm, common_norm = p._violin_scale_backcompat(
        scale, scale_hue, density_norm, common_norm,
    )

    # 对 bw 方法进行过时警告后向兼容处理
    bw_method = p._violin_bw_backcompat(bw, bw_method)

    # 设置 kde_kws 字典，包含 cut、gridsize、bw_method 和 bw_adjust 参数
    kde_kws = dict(cut=cut, gridsize=gridsize, bw_method=bw_method, bw_adjust=bw_adjust)

    # 如果 inner_kws 为 None，则创建空字典，否则拷贝 inner_kws
    inner_kws = {} if inner_kws is None else inner_kws.copy()

    # 绘制小提琴图，传入 width、dodge、gap、split 等参数和对应的绘图设置
    p.plot_violins(
        width=width,
        dodge=dodge,
        gap=gap,
        split=split,
        color=color,
        fill=fill,
        linecolor=linecolor,
        linewidth=linewidth,
        inner=inner,
        density_norm=density_norm,
        common_norm=common_norm,
        kde_kws=kde_kws,
        inner_kws=inner_kws,
        plot_kws=kwargs,
    )

    # 添加坐标轴标签
    p._add_axis_labels(ax)

    # 调整分类坐标轴的位置和布局
    p._adjust_cat_axis(ax, axis=p.orient)

    # 返回绘制好的坐标轴对象
    return ax


# 将 violinplot 函数的文档字符串设置为详细描述小提琴图绘制功能和参数说明的格式化文本
violinplot.__doc__ = dedent("""\
    Draw a patch representing a KDE and add observations or box plot statistics.

    A violin plot plays a similar role as a box-and-whisker plot. It shows the
    distribution of data points after grouping by one (or more) variables.
    Unlike a box plot, each violin is drawn using a kernel density estimate
    of the underlying distribution.

    {categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
""")
    {orient}
    # 图形的方向设置参数，控制数据在图中的方向或布局方式

    {color}
    # 图形的颜色设置参数，控制数据的颜色样式

    {palette}
    # 调色板参数，用于指定颜色的调色板

    {saturation}
    # 饱和度参数，控制颜色的饱和度

    {fill}
    # 填充参数，控制填充数据区域的样式

    inner : {"box", "quart", "point", "stick", None}
        Representation of the data in the violin interior. One of the following:

        - `"box"`: draw a miniature box-and-whisker plot
        - `"quart"`: show the quartiles of the data
        - `"point"` or `"stick"`: show each observation
        # 内部数据表示形式，指定小提琴图内部数据的展示方式，可以是盒式图、四分位数、点状或棒状，或为空

    split : bool
        Show an un-mirrored distribution, alternating sides when using `hue`.

        .. versionchanged:: v0.13.0
            Previously, this option required a `hue` variable with exactly two levels.
        # 是否显示非镜像分布，在使用 `hue` 参数时，交替显示数据分布在两侧

    {width}
    # 图形宽度参数，控制图形的宽度大小

    {dodge}
    # 躲避参数，用于在图形中避免重叠或冲突

    {gap}
    # 间隙参数，控制图形元素之间的间距

    {linewidth}
    # 线宽参数，控制图形线条的粗细

    {linecolor}
    # 线条颜色参数，控制图形线条的颜色

    cut : float
        Distance, in units of bandwidth, to extend the density past extreme
        datapoints. Set to 0 to limit the violin within the data range.
        # 切除参数，以带宽单位测量，用于超出极端数据点范围时扩展密度，设为0将限制小提琴图在数据范围内

    gridsize : int
        Number of points in the discrete grid used to evaluate the KDE.
        # 网格大小参数，离散网格中用于评估核密度估计的点数

    bw_method : {"scott", "silverman", float}
        Either the name of a reference rule or the scale factor to use when
        computing the kernel bandwidth. The actual kernel size will be
        determined by multiplying the scale factor by the standard deviation of
        the data within each group.

        .. versionadded:: v0.13.0
        # 带宽方法参数，指定计算核带宽时使用的规则名称或比例因子

    bw_adjust : float
        Factor that scales the bandwidth to use more or less smoothing.

        .. versionadded:: v0.13.0
        # 带宽调整因子参数，用于调整带宽以实现更多或更少的平滑效果

    density_norm : {"area", "count", "width"}
        Method that normalizes each density to determine the violin's width.
        If `area`, each violin will have the same area. If `count`, the width
        will be proportional to the number of observations. If `width`, each
        violin will have the same width.

        .. versionadded:: v0.13.0
        # 密度归一化方法参数，用于确定小提琴图宽度的归一化方法

    common_norm : bool
        When `True`, normalize the density across all violins.

        .. versionadded:: v0.13.0
        # 是否在所有小提琴图上归一化密度

    {hue_norm}
    # 色调归一化参数，控制色调的归一化方式

    {formatter}
    # 格式化器参数，用于格式化数据的显示方式

    {log_scale}
    # 对数刻度参数，控制图形是否使用对数刻度

    {native_scale}
    # 原生刻度参数，控制图形是否使用原生刻度

    {legend}
    # 图例参数，控制图例的显示方式

    scale : {"area", "count", "width"}
        .. deprecated:: v0.13.0
            See `density_norm`.
        # 比例参数，已废弃，参见 `density_norm`

    scale_hue : bool
        .. deprecated:: v0.13.0
            See `common_norm`.
        # 色调比例参数，已废弃，参见 `common_norm`

    bw : {'scott', 'silverman', float}
        .. deprecated:: v0.13.0
            See `bw_method` and `bw_adjust`.
        # 带宽参数，已废弃，参见 `bw_method` 和 `bw_adjust`

    inner_kws : dict of key, value mappings
        Keyword arguments for the "inner" plot, passed to one of:

        - :class:`matplotlib.collections.LineCollection` (with `inner="stick"`)
        - :meth:`matplotlib.axes.Axes.scatter` (with `inner="point"`)
        - :meth:`matplotlib.axes.Axes.plot` (with `inner="quart"` or `inner="box"`)

        Additionally, with `inner="box"`, the keywords `box_width`, `whis_width`,
        and `marker` receive special handling for the components of the "box" plot.

        .. versionadded:: v0.13.0
        # 内部关键字参数，传递给内部图的关键字参数

    {ax_in}
    # 输入轴参数，控制输入数据的轴设置

    kwargs : key, value mappings
        Keyword arguments for the violin patches, passsed through to
        :meth:`matplotlib.axes.Axes.fill_between`.
        # 关键字参数，传递给小提琴图补丁的参数

    Returns
    -------
    {ax_out}
    # 返回值，返回结果的轴设置
    See Also
    --------
    {boxplot}
    {stripplot}
    {swarmplot}
    {catplot}

    Examples
    --------
    .. include:: ../docstrings/violinplot.rst

    """).format(**_categorical_docs)


注释：


    # 显示相关的参考链接，指向箱线图、条纹图、蜂群图和类别图的文档
    See Also
    --------
    {boxplot}
    {stripplot}
    {swarmplot}
    {catplot}

    # 显示示例部分，包括引入小提琴图文档的路径
    Examples
    --------
    .. include:: ../docstrings/violinplot.rst

    # 使用_format方法格式化字符串，并传入_categorical_docs字典，生成最终的文档内容
    """).format(**_categorical_docs)
def boxenplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    orient=None, color=None, palette=None, saturation=.75, fill=True,
    dodge="auto", width=.8, gap=0, linewidth=None, linecolor=None,
    width_method="exponential", k_depth="tukey", outlier_prop=0.007, trust_alpha=0.05,
    showfliers=True, hue_norm=None, log_scale=None, native_scale=False, formatter=None,
    legend="auto", scale=deprecated, box_kws=None, flier_kws=None, line_kws=None,
    ax=None, **kwargs,
):

    # 创建一个 _CategoricalPlotter 对象用于处理分类数据的绘图
    p = _CategoricalPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue),
        order=order,
        orient=orient,
        color=color,
        legend=legend,
    )

    # 如果没有指定轴对象，使用当前的 matplotlib 轴对象
    if ax is None:
        ax = plt.gca()

    # 如果数据为空，则直接返回当前轴对象
    if p.plot_data.empty:
        return ax

    # 根据 dodge 参数确定是否需要 dodge 功能，需要在 scale_categorical 更改坐标系列 dtype 之前进行设置
    if dodge == "auto":
        dodge = p._dodge_needed()

    # 如果变量的方向是分类的，或者指定了不使用原生比例尺，则进行分类的比例尺调整
    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    # 将绘图对象 p 与 matplotlib 轴对象 ax 绑定
    p._attach(ax, log_scale=log_scale)

    # Deprecations to remove in v0.14.0.
    # 移除 v0.14.0 版本中的不推荐使用项
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    # Longer-term deprecations
    # 较长期的过时功能
    width_method = p._boxen_scale_backcompat(scale, width_method)

    # 根据 fill 参数决定 saturation 的值
    saturation = saturation if fill else 1

    # 根据 hue 参数映射颜色，设置饱和度为 saturation
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)

    # 根据默认颜色设定线条颜色
    color = _default_color(
        ax.fill_between, hue, color,
        {},  # TODO how to get default color?
        saturation=saturation,
    )

    # 根据 hue 颜色设定线条颜色的补充色
    linecolor = p._complement_color(linecolor, color, p._hue_map)

    # 绘制 boxenplot 主体部分
    p.plot_boxens(
        width=width,
        dodge=dodge,
        gap=gap,
        fill=fill,
        color=color,
        linecolor=linecolor,
        linewidth=linewidth,
        width_method=width_method,
        k_depth=k_depth,
        outlier_prop=outlier_prop,
        trust_alpha=trust_alpha,
        showfliers=showfliers,
        box_kws=box_kws,
        flier_kws=flier_kws,
        line_kws=line_kws,
        plot_kws=kwargs,
    )

    # 添加坐标轴标签
    p._add_axis_labels(ax)

    # 调整分类坐标轴
    p._adjust_cat_axis(ax, axis=p.orient)

    # 返回绘制的轴对象
    return ax

# 设置 boxenplot 函数的文档字符串
boxenplot.__doc__ = dedent("""\
    Draw an enhanced box plot for larger datasets.

    This style of plot was originally named a "letter value" plot because it
    shows a large number of quantiles that are defined as "letter values".  It
    is similar to a box plot in plotting a nonparametric representation of a
    distribution in which all features correspond to actual observations. By
    plotting more quantiles, it provides more information about the shape of
    the distribution, particularly in the tails.

    {categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}

""")
    {orient}
    {color}
    {palette}
    {saturation}
    {fill}
    {dodge}
    {width}
    {gap}
    {linewidth}
    {linecolor}
    # width_method : {"exponential", "linear", "area"}
    #     控制盒图中箱子宽度的计算方法：

    #     - `"exponential"`: 根据对应百分位数来表示
    #     - `"linear"`: 每个箱子宽度减少一个常量值
    #     - `"area"`: 表示该箱子中数据点的密度

    # k_depth : {"tukey", "proportion", "trustworthy", "full"} or int
    #     每个尾部要计算和绘制的级别数量：

    #     - `"tukey"`: 使用 log2(n) - 3 个级别，覆盖与箱线图触须类似的范围
    #     - `"proportion"`: 留下大约 `outlier_prop` 的异常值
    #     - `"trustworthy"`: 扩展到置信度至少为 `trust_alpha` 的级别
    #     - `"full"`: 使用 log2(n) + 1 个级别，并扩展到最极端的点

    # outlier_prop : float
    #     预计为异常值的数据比例；当 `k_depth="proportion"` 时使用。

    # trust_alpha : float
    #     最极端级别的置信度阈值；当 `k_depth="trustworthy"` 时使用。

    # showfliers : bool
    #     如果为 False，则不绘制异常值。

    {hue_norm}
    {log_scale}
    {native_scale}
    {formatter}
    {legend}
    # box_kws : dict
    #     盒子图艺术家的关键字参数；传递给 :class:`matplotlib.patches.Rectangle`。

    #     .. versionadded:: v0.12.0

    # line_kws : dict
    #     表示中位数的线的关键字参数；传递给 :meth:`matplotlib.axes.Axes.plot`。

    #     .. versionadded:: v0.12.0

    # flier_kws : dict
    #     表示异常观测值的散点的关键字参数；传递给 :meth:`matplotlib.axes.Axes.scatter`。

    #     .. versionadded:: v0.12.0

    {ax_in}
    # kwargs : key, value mappings
    #     其他关键字参数传递给 :class:`matplotlib.patches.Rectangle`，会被 `box_kws` 中的参数覆盖。

    # Returns
    # -------
    {ax_out}

    # See Also
    # --------
    {violinplot}
    {boxplot}
    {catplot}

    # Notes
    # -----

    # 对于更详细的解释，您可以阅读介绍此绘图的论文：
    # https://vita.had.co.nz/papers/letter-value-plot.html

    # Examples
    # --------
    # .. include:: ../docstrings/boxenplot.rst

    """).format(**_categorical_docs)
def stripplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    jitter=True, dodge=False, orient=None, color=None, palette=None,
    size=5, edgecolor=default, linewidth=0,
    hue_norm=None, log_scale=None, native_scale=False, formatter=None, legend="auto",
    ax=None, **kwargs
):
    # 创建一个_CategoricalPlotter对象来管理数据和绘图变量
    p = _CategoricalPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue),
        order=order,
        orient=orient,
        color=color,
        legend=legend,
    )

    # 如果没有指定ax，则使用当前的图形坐标轴
    if ax is None:
        ax = plt.gca()

    # 如果没有数据可绘制，直接返回当前坐标轴
    if p.plot_data.empty:
        return ax

    # 如果变量是分类的或者没有使用原生比例尺，则调整为分类的比例尺
    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    # 将数据附加到给定的坐标轴上，支持对数比例尺
    p._attach(ax, log_scale=log_scale)

    # Deprecations to remove in v0.14.0.
    # 移除在 v0.14.0 版本中不再支持的旧版参数
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    # 映射颜色到色板上，并根据hue_order排序和hue_norm归一化
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    # 获取默认颜色用于绘制散点图，考虑hue参数
    color = _default_color(ax.scatter, hue, color, kwargs)
    # 获取边缘颜色，与主要颜色形成对比，基于_hue_map映射
    edgecolor = p._complement_color(edgecolor, color, p._hue_map)

    # 设置默认的绘图参数
    kwargs.setdefault("zorder", 3)
    size = kwargs.get("s", size)

    kwargs.update(
        s=size ** 2,  # 设置标记的大小为size的平方
        edgecolor=edgecolor,  # 设置标记的边缘颜色
        linewidth=linewidth,  # 设置标记的边缘线宽度
    )

    # 绘制strip图，应用jitter和dodge，使用指定的颜色和绘图参数
    p.plot_strips(
        jitter=jitter,
        dodge=dodge,
        color=color,
        plot_kws=kwargs,
    )

    # 添加轴标签，这段注释掉的代码原本位于分布图的绘制方法内，但移出来也许更好？
    # 或者，我们可以考虑在_open函数中添加默认的轴标签，这似乎更明智。
    p._add_axis_labels(ax)
    # 调整分类轴的设置，如方向
    p._adjust_cat_axis(ax, axis=p.orient)

    return ax
    edgecolor : matplotlib color, "gray" is special-cased
        # 每个点周围的线条颜色，使用 matplotlib 的颜色表示，特殊情况下为 "gray"
        # 如果指定为 "gray"，则亮度由用于点主体的调色板确定
        # 注意，stripplot 默认情况下具有 linewidth=0，因此仅当线宽非零时才会显示边缘颜色。
    {linewidth}
        # 线宽参数，用于控制每个点周围边缘的粗细
    {hue_norm}
        # 色调标准化参数，用于映射色调变量到颜色空间
    {log_scale}
        # 对数刻度参数，用于在轴上应用对数刻度
    {native_scale}
        # 原生刻度参数，用于指定轴上的刻度类型
    {formatter}
        # 格式化器参数，用于设置轴上的数值格式
    {legend}
        # 图例参数，用于控制是否显示图例及其位置
    {ax_in}
        # 输入轴参数，指定图形的输入轴对象
    kwargs : key, value mappings
        # 其他关键字参数将传递给 matplotlib.axes.Axes.scatter 方法
    Returns
    -------
    {ax_out}
        # 返回输出轴对象参数，表示返回的轴对象
    See Also
    --------
    {swarmplot}
        # 参见 swarmplot 方法的相关信息
    {boxplot}
        # 参见 boxplot 方法的相关信息
    {violinplot}
        # 参见 violinplot 方法的相关信息
    {catplot}
        # 参见 catplot 方法的相关信息
    Examples
    --------
    .. include:: ../docstrings/stripplot.rst
        # 包含了一个指向 stripplot 文档的引用
    """).format(**_categorical_docs)
        # 使用 _categorical_docs 中的参数格式化字符串，生成最终的文档字符串
def swarmplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    dodge=False, orient=None, color=None, palette=None,
    size=5, edgecolor=None, linewidth=0, hue_norm=None, log_scale=None,
    native_scale=False, formatter=None, legend="auto", warn_thresh=.05,
    ax=None, **kwargs
):
    # 创建一个 _CategoricalPlotter 对象来处理分类绘图
    p = _CategoricalPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue),
        order=order,
        orient=orient,
        color=color,
        legend=legend,
    )

    # 如果未指定绘图区域 ax，则使用当前的 matplotlib Axes
    if ax is None:
        ax = plt.gca()

    # 如果没有数据要绘制，则直接返回当前 Axes 对象
    if p.plot_data.empty:
        return ax

    # 如果数据的方向为分类变量或者不使用原生比例尺，则对其进行分类尺度调整
    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    # 将数据附加到指定的 Axes 对象上
    p._attach(ax, log_scale=log_scale)

    # 如果没有 x 和 y 数据，直接返回当前 Axes 对象
    if not p.has_xy_data:
        return ax

    # 在版本 v0.14.0 中移除的废弃功能
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    # 映射色调到绘图属性
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    # 获取默认的颜色设置
    color = _default_color(ax.scatter, hue, color, kwargs)

    # 根据色调补充边缘颜色
    edgecolor = p._complement_color(edgecolor, color, p._hue_map)

    # 设置默认的绘图属性
    kwargs.setdefault("zorder", 3)
    size = kwargs.get("s", size)

    # 如果未指定 linewidth，则根据 size 设置默认值
    if linewidth is None:
        linewidth = size / 10

    # 更新绘图属性字典
    kwargs.update(dict(
        s=size ** 2,
        edgecolor=edgecolor,
        linewidth=linewidth,
    ))

    # 绘制 swarmplot
    p.plot_swarms(
        dodge=dodge,
        color=color,
        warn_thresh=warn_thresh,
        plot_kws=kwargs,
    )

    # 添加坐标轴标签
    p._add_axis_labels(ax)

    # 调整分类轴
    p._adjust_cat_axis(ax, axis=p.orient)

    # 返回绘图的 Axes 对象
    return ax
    edgecolor : matplotlib color, "gray" is special-cased
        # 每个点周围的线条颜色，使用matplotlib颜色，特别处理了 "gray" 的情况
        Color of the lines around each point. If you pass `"gray"`, the
        brightness is determined by the color palette used for the body
        of the points.
    {linewidth}
        # 线条的宽度，此处应该是一个占位符，会被具体的值替换
    {hue_norm}
        # 色调的标准化方法，此处应该是一个占位符，会被具体的值替换
    {log_scale}
        # 是否使用对数刻度，此处应该是一个占位符，会被具体的值替换
    {native_scale}
        # 是否使用本地刻度，此处应该是一个占位符，会被具体的值替换
    {formatter}
        # 格式化器，此处应该是一个占位符，会被具体的值替换
    {legend}
        # 是否显示图例，此处应该是一个占位符，会被具体的值替换
    warn_thresh : float
        # 触发警告的重叠点比例阈值
        The proportion of points that must overlap to trigger a warning.
    {ax_in}
        # 输入的坐标轴对象，此处应该是一个占位符，会被具体的值替换
    kwargs : key, value mappings
        # 其他关键字参数会传递给 matplotlib.axes.Axes.scatter 方法
        Other keyword arguments are passed through to
        :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    {ax_out}
        # 返回的坐标轴对象，此处应该是一个占位符，会被具体的值替换

    See Also
    --------
    {boxplot}
        # 参见箱线图的相关方法
    {violinplot}
        # 参见小提琴图的相关方法
    {stripplot}
        # 参见条形图的相关方法
    {catplot}
        # 参见分类图的相关方法

    Examples
    --------
    .. include:: ../docstrings/swarmplot.rst
        # 包含一个外部的示例文档，通常是一个链接或者文件路径

    """).format(**_categorical_docs)
        # 使用 _categorical_docs 中的内容格式化文档字符串
def barplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    estimator="mean", errorbar=("ci", 95), n_boot=1000, seed=None, units=None,
    weights=None, orient=None, color=None, palette=None, saturation=.75,
    fill=True, hue_norm=None, width=.8, dodge="auto", gap=0, log_scale=None,
    native_scale=False, formatter=None, legend="auto", capsize=0, err_kws=None,
    ci=deprecated, errcolor=deprecated, errwidth=deprecated, ax=None, **kwargs,
):
    # 根据输入参数设置错误栏的类型，确保向后兼容性
    errorbar = utils._deprecate_ci(errorbar, ci)

    # 如果估算器是长度函数len，则更改为字符串"size"，以避免在Series.agg中出错
    if estimator is len:
        estimator = "size"

    # 创建_CategoricalAggPlotter对象进行分类聚合绘图
    p = _CategoricalAggPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue, units=units, weight=weights),
        order=order,
        orient=orient,
        color=color,
        legend=legend,
    )

    # 如果未提供绘图轴，则默认使用当前的绘图轴
    if ax is None:
        ax = plt.gca()

    # 如果数据为空，则直接返回当前绘图轴对象
    if p.plot_data.empty:
        return ax

    # 自动检测是否需要进行躲避处理，以适应条形图的布局
    if dodge == "auto":
        # 需要在scale_categorical更改坐标系列的数据类型之前进行处理
        dodge = p._dodge_needed()

    # 如果变量类型为分类变量或者不使用原生缩放，则对分类变量进行缩放
    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    # 将绘图对象p附加到指定的绘图轴上，支持对数尺度
    p._attach(ax, log_scale=log_scale)

    # 移除v0.14.0版本中的过时警告
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    # 根据是否填充数据设置饱和度值
    saturation = saturation if fill else 1

    # 映射色调，根据给定的调色板和顺序
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)

    # 设置颜色，根据默认颜色函数和参数进行调整
    color = _default_color(ax.bar, hue, color, kwargs, saturation=saturation)

    # 根据权重数据或者估算器类型选择聚合器类型
    agg_cls = WeightedAggregator if "weight" in p.plot_data else EstimateAggregator
    aggregator = agg_cls(estimator, errorbar, n_boot=n_boot, seed=seed)

    # 将错误关键字参数进行标准化，确保它们符合mpl.lines.Line2D的标准
    err_kws = {} if err_kws is None else normalize_kwargs(err_kws, mpl.lines.Line2D)

    # 移除v0.15.0版本中的过时警告
    err_kws, capsize = p._err_kws_backcompat(err_kws, errcolor, errwidth, capsize)

    # 绘制条形图，根据给定的聚合器、躲避类型、宽度、间隙、颜色和其他绘图关键字参数
    p.plot_bars(
        aggregator=aggregator,
        dodge=dodge,
        width=width,
        gap=gap,
        color=color,
        fill=fill,
        capsize=capsize,
        err_kws=err_kws,
        plot_kws=kwargs,
    )

    # 添加坐标轴标签到绘图对象p上
    p._add_axis_labels(ax)

    # 调整分类轴的显示方式，应用到指定的绘图轴上
    p._adjust_cat_axis(ax, axis=p.orient)

    # 返回绘图轴对象ax
    return ax
    {saturation}
    {fill}
    {hue_norm}
    {width}
    {dodge}
    {gap}
    {log_scale}
    {native_scale}
    {formatter}
    {legend}
    {capsize}
    {err_kws}
    {ci}
    {errcolor}
    {errwidth}
    {ax_in}
    kwargs : key, value mappings
        Other parameters are passed through to :class:`matplotlib.patches.Rectangle`.
    将所有关键字参数作为键值映射传递给 `matplotlib.patches.Rectangle` 类。
    返回值
    -------
    {ax_out}
    返回经过处理的轴对象。
    参见
    --------
    {countplot}
    {pointplot}
    {catplot}
    注解
    -----

    对于数据集中0不是有意义的值的情况， :func:`pointplot` 允许您专注于一个或多个分类变量的级别之间的差异。

    还应该记住，条形图仅显示均值（或其他聚合）值，但通常更信息丰富的是显示每个分类变量级别的值分布。在这些情况下，像 :func:`boxplot` 或 :func:`violinplot` 这样的方法可能更合适。

    示例
    --------
    .. include:: ../docstrings/barplot.rst

    """).format(**_categorical_docs)
def pointplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    estimator="mean", errorbar=("ci", 95), n_boot=1000, seed=None, units=None,
    weights=None, color=None, palette=None, hue_norm=None, markers=default,
    linestyles=default, dodge=False, log_scale=None, native_scale=False,
    orient=None, capsize=0, formatter=None, legend="auto", err_kws=None,
    ci=deprecated, errwidth=deprecated, join=deprecated, scale=deprecated,
    ax=None, **kwargs,
):
    """
    Show point estimates and errors using lines with markers.

    A point plot represents an estimate of central tendency for a numeric
    variable by the position of the dot and provides some indication of the
    uncertainty around that estimate using error bars.

    Point plots can be more useful than bar plots for focusing comparisons
    between different levels of one or more categorical variables. They are
    particularly adept at showing interactions: how the relationship between
    levels of one categorical variable changes across levels of a second
    categorical variable. The lines that join each point from the same `hue`
    """
    
    # 处理 errorbar 参数的向后兼容性，使用 utils._deprecate_ci 函数
    errorbar = utils._deprecate_ci(errorbar, ci)
    
    # 创建 _CategoricalAggPlotter 对象 p，用于处理分类数据的聚合绘图
    p = _CategoricalAggPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue, units=units, weight=weights),
        order=order,
        orient=orient,
        # 处理特殊的向后兼容性，如果未指定 color 和 palette，则使用默认颜色 "C0"
        color="C0" if (color is None and palette is None) else color,
        legend=legend,
    )

    # 如果未提供 ax 参数，则使用当前的绘图区域
    if ax is None:
        ax = plt.gca()

    # 如果没有数据可绘制，则直接返回当前的绘图区域 ax
    if p.plot_data.empty:
        return ax

    # 如果数据的变量类型为分类变量或者未启用原生比例尺，则调用 scale_categorical 函数
    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    # 将 p 对象附加到绘图区域 ax 上，并应用对数比例尺
    p._attach(ax, log_scale=log_scale)

    # 移除 v0.14.0 版本中的一些弃用项
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    # 映射 hue 变量到颜色空间
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    
    # 获取默认的颜色
    color = _default_color(ax.plot, hue, color, kwargs)

    # 根据数据是否有权重选择聚合器类
    agg_cls = WeightedAggregator if "weight" in p.plot_data else EstimateAggregator
    # 创建聚合器对象 aggregator
    aggregator = agg_cls(estimator, errorbar, n_boot=n_boot, seed=seed)
    # 标准化 err_kws 参数
    err_kws = {} if err_kws is None else normalize_kwargs(err_kws, mpl.lines.Line2D)

    # 移除 v0.15.0 版本中的一些弃用项
    p._point_kwargs_backcompat(scale, join, kwargs)
    err_kws, capsize = p._err_kws_backcompat(err_kws, None, errwidth, capsize)

    # 绘制点图
    p.plot_points(
        aggregator=aggregator,
        markers=markers,
        linestyles=linestyles,
        dodge=dodge,
        color=color,
        capsize=capsize,
        err_kws=err_kws,
        plot_kws=kwargs,
    )

    # 添加轴标签
    p._add_axis_labels(ax)
    # 调整分类轴
    p._adjust_cat_axis(ax, axis=p.orient)

    # 返回绘图区域 ax
    return ax
    level allow interactions to be judged by differences in slope, which is
    easier for the eyes than comparing the heights of several groups of points
    or bars.


    {categorical_narrative}


    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    {stat_api_params}
    {color}
    {palette}
    {hue_norm}


    markers : string or list of strings
        Markers to use for each of the `hue` levels.


    linestyles : string or list of strings
        Line styles to use for each of the `hue` levels.


    dodge : bool or float
        Amount to separate the points for each level of the `hue` variable along
        the categorical axis. Setting to `True` will apply a small default.


    {log_scale}
    {native_scale}
    {orient}
    {capsize}
    {formatter}
    {legend}
    {err_kws}
    {ci}
    {errwidth}


    join : bool
        If `True`, connect point estimates with a line.

        .. deprecated:: v0.13.0
            Set `linestyle="none"` to remove the lines between the points.


    scale : float
        Scale factor for the plot elements.

        .. deprecated:: v0.13.0
            Control element sizes with :class:`matplotlib.lines.Line2D` parameters.


    {ax_in}
    kwargs : key, value mappings
        Other parameters are passed through to :class:`matplotlib.lines.Line2D`.

        .. versionadded:: v0.13.0


    Returns
    -------
    {ax_out}


    See Also
    --------
    {barplot}
    {catplot}


    Notes
    -----
    It is important to keep in mind that a point plot shows only the mean (or
    other estimator) value, but in many cases it may be more informative to
    show the distribution of values at each level of the categorical variables.
    In that case, other approaches such as a box or violin plot may be more
    appropriate.


    Examples
    --------
    .. include:: ../docstrings/pointplot.rst


    """).format(**_categorical_docs)
# 定义一个函数 countplot，用于绘制分类变量的计数条形图
def countplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    orient=None, color=None, palette=None, saturation=.75, fill=True, hue_norm=None,
    stat="count", width=.8, dodge="auto", gap=0, log_scale=None, native_scale=False,
    formatter=None, legend="auto", ax=None, **kwargs
):

    # 如果 x 为 None 而 y 不为 None，则设定 orient 为 "y"，并尝试推断 x
    if x is None and y is not None:
        orient = "y"
        x = 1 if list(y) else None
    # 如果 x 不为 None 而 y 为 None，则设定 orient 为 "x"，并尝试推断 y
    elif x is not None and y is None:
        orient = "x"
        y = 1 if list(x) else None
    # 如果 x 和 y 同时不为 None，则抛出 TypeError 异常
    elif x is not None and y is not None:
        raise TypeError("Cannot pass values for both `x` and `y`.")

    # 创建 _CategoricalAggPlotter 对象 p，用于处理分类聚合绘图
    p = _CategoricalAggPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue),
        order=order,
        orient=orient,
        color=color,
        legend=legend,
    )

    # 如果没有指定绘图坐标系 ax，则默认使用当前的 Axes 对象
    if ax is None:
        ax = plt.gca()

    # 如果没有数据可绘制，则直接返回当前的 Axes 对象
    if p.plot_data.empty:
        return ax

    # 如果 dodge 为 "auto"，则根据条件确定是否需要进行分组绘制
    dodge = p._dodge_needed() if dodge == "auto" else dodge

    # 如果绘图方向为分类变量或者不使用原生缩放，则调整分类变量的尺度
    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    # 将绘图对象 p 与指定的 Axes 对象 ax 绑定，并根据需要设定对数刻度
    p._attach(ax, log_scale=log_scale)

    # Deprecations to remove in v0.14.0.
    # 移除 v0.14.0 版本中的过时功能
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    # 根据 fill 参数设定饱和度 saturation
    saturation = saturation if fill else 1

    # 根据调色板 palette、色调顺序 hue_order、色调规范化 hue_norm 和饱和度 saturation 进行映射
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)

    # 根据 ax.bar 的默认颜色处理 hue、color，返回最终的 color
    color = _default_color(ax.bar, hue, color, kwargs, saturation)

    # 在宽格式数据下，将计数轴数据统一设定为 1
    count_axis = {"x": "y", "y": "x"}[p.orient]
    if p.input_format == "wide":
        p.plot_data[count_axis] = 1

    # 检查统计类型参数 stat 是否有效，若无效则抛出错误
    _check_argument("stat", ["count", "percent", "probability", "proportion"], stat)

    # 将变量轴的统计类型设定为 stat
    p.variables[count_axis] = stat

    # 若统计类型不为 "count"，则根据总数对计数数据进行归一化处理
    if stat != "count":
        denom = 100 if stat == "percent" else 1
        p.plot_data[count_axis] /= len(p.plot_data) / denom

    # 创建 EstimateAggregator 对象 aggregator，用于计算汇总统计数据
    aggregator = EstimateAggregator("sum", errorbar=None)

    # 绘制条形图，设定聚合器 aggregator、分组方式 dodge、条形宽度 width、间隙 gap、颜色 color 等参数
    p.plot_bars(
        aggregator=aggregator,
        dodge=dodge,
        width=width,
        gap=gap,
        color=color,
        fill=fill,
        capsize=0,
        err_kws={},
        plot_kws=kwargs,
    )

    # 添加坐标轴标签到 Axes 对象 ax
    p._add_axis_labels(ax)

    # 调整分类轴的显示方式，对 Axes 对象 ax 进行调整
    p._adjust_cat_axis(ax, axis=p.orient)

    # 返回最终绘制的 Axes 对象 ax
    return ax
    {saturation}
    {fill}
    {hue_norm}

这里展示了一些可选参数，它们用于调整色彩饱和度、填充、以及色调的归一化。


    stat : {'count', 'percent', 'proportion', 'probability'}

这是一个关键参数，用于指定要计算的统计量类型。当设置为 `'count'` 时，条形图的高度表示数据的计数；当设置为其他值时，条形图的高度将被归一化，以便它们在绘图中总和为100%（对于 `'percent'`）或为1（对于其他情况）。


        Statistic to compute; when not `'count'`, bar heights will be normalized so that
        they sum to 100 (for `'percent'`) or 1 (otherwise) across the plot.

这是 `stat` 参数的具体说明，解释了它的作用和行为。


        .. versionadded:: v0.13.0

这行注释指出了在版本 v0.13.0 中添加了 `stat` 参数。


    {width}
    {dodge}
    {gap}
    {log_scale}
    {native_scale}
    {formatter}
    {legend}
    {ax_in}

这些是可选的参数，用于设置条形图的宽度、间距、是否使用对数刻度、是否使用原生刻度、格式化器、是否显示图例以及输入轴。


    kwargs : key, value mappings

这个参数允许用户传递任意其他参数，这些参数会传递给 `matplotlib.patches.Rectangle` 类。


    Returns
    -------
    {ax_out}

这说明了函数的返回结果，返回一个轴对象 `{ax_out}`。


    See Also
    --------
    histplot : Bin and count observations with additional options.
    {barplot}
    {catplot}

这些是与当前函数相关联的其它函数，提供了进一步了解和使用的可能性。


    Examples
    --------
    .. include:: ../docstrings/countplot.rst

    """).format(**_categorical_docs)

这段代码展示了一个例子，引用了一个外部的文档文件 `countplot.rst`，用于展示函数的使用示例。


    """).format(**_categorical_docs)

这里使用了一个格式化字符串，通过 `_categorical_docs` 字典提供的值替换了文本中的占位符 `{}`。
def catplot(
    data=None, *, x=None, y=None, hue=None, row=None, col=None, kind="strip",
    estimator="mean", errorbar=("ci", 95), n_boot=1000, seed=None, units=None,
    weights=None, order=None, hue_order=None, row_order=None, col_order=None,
    col_wrap=None, height=5, aspect=1, log_scale=None, native_scale=False,
    formatter=None, orient=None, color=None, palette=None, hue_norm=None,
    legend="auto", legend_out=True, sharex=True, sharey=True,
    margin_titles=False, facet_kws=None, ci=deprecated, **kwargs
):
    # 检查是否尝试在特定轴上绘图，并发出警告
    if "ax" in kwargs:
        msg = ("catplot 是一个图级别的函数，不接受目标轴参数。您可以尝试使用 {kind}plot")
        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    # 不饱和的绘图类型列表
    desaturated_kinds = ["bar", "count", "box", "violin", "boxen"]
    # 未分组的绘图类型列表
    undodged_kinds = ["strip", "swarm", "point"]

    # 根据绘图类型选择相应的绘图类
    if kind in ["bar", "point", "count"]:
        Plotter = _CategoricalAggPlotter
    else:
        Plotter = _CategoricalPlotter

    # 如果绘图类型是 "count"，处理特殊情况
    if kind == "count":
        if x is None and y is not None:
            orient = "y"
            x = 1
        elif x is not None and y is None:
            orient = "x"
            y = 1
        elif x is not None and y is not None:
            raise ValueError("Cannot pass values for both `x` and `y`.")

    # 创建绘图对象
    p = Plotter(
        data=data,
        variables=dict(
            x=x, y=y, hue=hue, row=row, col=col, units=units, weight=weights
        ),
        order=order,
        orient=orient,
        # 处理特殊的向后兼容性，其中 pointplot 最初不会默认多色绘制，除非指定了调色板。
        color="C0" if kind == "point" and palette is None and color is None else color,
        legend=legend,
    )

    # 处理缺少名称信息的面板变量
    for var in ["row", "col"]:
        if var in p.variables and p.variables[var] is None:
            p.variables[var] = f"_{var}_"

    # 调整用于 FacetGrid 的绘图数据数据框
    facet_data = p.plot_data.rename(columns=p.variables)
    facet_data = facet_data.loc[:, ~facet_data.columns.duplicated()]

    col_name = p.variables.get("col", None)
    row_name = p.variables.get("row", None)

    if facet_kws is None:
        facet_kws = {}

    # 创建 FacetGrid 对象
    g = FacetGrid(
        data=facet_data, row=row_name, col=col_name, col_wrap=col_wrap,
        row_order=row_order, col_order=col_order, sharex=sharex, sharey=sharey,
        legend_out=legend_out, margin_titles=margin_titles,
        height=height, aspect=aspect,
        **facet_kws,
    )

    # 在此处进行捕获，因为 scale_categorical 将会插入一个（空）的 x 变量，即使它是空的。
    # 目前不清楚是否需要这样做，或者是否禁用它是更干净的解决方案。
    has_xy_data = p.has_xy_data
    # 如果不是本地比例或者变量类型为“categorical”，则调用缩放分类变量的方法
    if not native_scale or p.var_types[p.orient] == "categorical":
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    # 将图形对象 g 附加到绘图对象 p 上，支持对数比例尺
    p._attach(g, log_scale=log_scale)

    # 如果没有 xy 数据，则直接返回图形对象 g
    if not has_xy_data:
        return g

    # 进行 v0.14.0 版本中即将移除的功能的兼容处理
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    # 其他即将移除的功能的兼容处理
    errorbar = utils._deprecate_ci(errorbar, ci)

    # 设定饱和度，默认为0.75，如果是 desaturated_kinds 中的图形并且 fill 为真，则为 0.75，否则为 1
    saturation = kwargs.pop(
        "saturation",
        0.75 if kind in desaturated_kinds and kwargs.get("fill", True) else 1
    )
    # 应用色调映射到图形对象 p 上
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)

    # 如果没有色调数据，则设定默认颜色，避免每个艺术家使用不同颜色
    if hue is None:
        color = "C0" if color is None else color
        # 如果饱和度小于 1，则对颜色进行去饱和处理
        if saturation < 1:
            color = desaturate(color, saturation)

    # 如果图形类型为 "strip" 或 "swarm"
    if kind in ["strip", "swarm"]:
        # 标准化关键字参数
        kwargs = normalize_kwargs(kwargs, mpl.collections.PathCollection)
        # 设定边缘颜色，用于区分图形对象
        kwargs["edgecolor"] = p._complement_color(
            kwargs.pop("edgecolor", default), color, p._hue_map
        )

    # 设定宽度，默认为 0.8
    width = kwargs.pop("width", 0.8)
    # 设定是否需要躲避，默认为 False 或 "auto"
    dodge = kwargs.pop("dodge", False if kind in undodged_kinds else "auto")
    if dodge == "auto":
        # 检查是否需要自动躲避
        dodge = p._dodge_needed()

    # 如果数据中包含 "weight" 参数
    if "weight" in p.plot_data:
        # 如果图形类型不是 "bar" 或 "point"，则发出警告
        if kind not in ["bar", "point"]:
            msg = f"The `weights` parameter has no effect with kind={kind!r}."
            warnings.warn(msg, stacklevel=2)
        # 使用 WeightedAggregator 类
        agg_cls = WeightedAggregator
    else:
        # 使用 EstimateAggregator 类
        agg_cls = EstimateAggregator

    # 如果图形类型为 "strip"
    if kind == "strip":
        # 设定抖动参数，默认为 True
        jitter = kwargs.pop("jitter", True)
        plot_kws = kwargs.copy()
        plot_kws.setdefault("zorder", 3)  # 设置图形层次顺序
        plot_kws.setdefault("linewidth", 0)  # 设置线宽为 0
        if "s" not in plot_kws:
            plot_kws["s"] = plot_kws.pop("size", 5) ** 2  # 设置散点大小

        # 绘制条带图
        p.plot_strips(
            jitter=jitter,
            dodge=dodge,
            color=color,
            plot_kws=plot_kws,
        )

    # 如果图形类型为 "swarm"
    elif kind == "swarm":
        # 设定警告阈值，默认为 0.05
        warn_thresh = kwargs.pop("warn_thresh", .05)
        plot_kws = kwargs.copy()
        plot_kws.setdefault("zorder", 3)  # 设置图形层次顺序
        if "s" not in plot_kws:
            plot_kws["s"] = plot_kws.pop("size", 5) ** 2  # 设置散点大小

        if plot_kws.setdefault("linewidth", 0) is None:
            plot_kws["linewidth"] = np.sqrt(plot_kws["s"]) / 10  # 计算线宽

        # 绘制蜂群图
        p.plot_swarms(
            dodge=dodge,
            color=color,
            warn_thresh=warn_thresh,
            plot_kws=plot_kws,
        )
    elif kind == "box":
        # 将 kwargs 复制到 plot_kws 中，以备后续使用
        plot_kws = kwargs.copy()
        # 从 plot_kws 中弹出 "gap" 参数，默认为 0
        gap = plot_kws.pop("gap", 0)
        # 从 plot_kws 中弹出 "fill" 参数，默认为 True
        fill = plot_kws.pop("fill", True)
        # 从 plot_kws 中弹出 "whis" 参数，默认为 1.5
        whis = plot_kws.pop("whis", 1.5)
        # 从 plot_kws 中弹出 "linewidth" 参数，没有默认值
        linewidth = plot_kws.pop("linewidth", None)
        # 从 plot_kws 中弹出 "fliersize" 参数，默认为 5
        fliersize = plot_kws.pop("fliersize", 5)
        # 从 plot_kws 中弹出 "linecolor" 参数，默认为 "auto"，通过 _complement_color 方法获取颜色
        linecolor = p._complement_color(
            plot_kws.pop("linecolor", "auto"), color, p._hue_map
        )

        # 调用 p 对象的 plot_boxes 方法，传入相应的参数进行箱线图绘制
        p.plot_boxes(
            width=width,
            dodge=dodge,
            gap=gap,
            fill=fill,
            whis=whis,
            color=color,
            linecolor=linecolor,
            linewidth=linewidth,
            fliersize=fliersize,
            plot_kws=plot_kws,
        )

    elif kind == "violin":
        # 将 kwargs 复制到 plot_kws 中，以备后续使用
        plot_kws = kwargs.copy()
        # 从 plot_kws 中弹出 "gap" 参数，默认为 0
        gap = plot_kws.pop("gap", 0)
        # 从 plot_kws 中弹出 "fill" 参数，默认为 True
        fill = plot_kws.pop("fill", True)
        # 从 plot_kws 中弹出 "split" 参数，默认为 False
        split = plot_kws.pop("split", False)
        # 从 plot_kws 中弹出 "inner" 参数，默认为 "box"
        inner = plot_kws.pop("inner", "box")
        # 从 plot_kws 中弹出 "density_norm" 参数，默认为 "area"
        density_norm = plot_kws.pop("density_norm", "area")
        # 从 plot_kws 中弹出 "common_norm" 参数，默认为 False
        common_norm = plot_kws.pop("common_norm", False)

        # 从 plot_kws 中弹出 "scale" 和 "scale_hue" 参数，进行后向兼容处理，获取对应的 density_norm 和 common_norm
        scale = plot_kws.pop("scale", deprecated)
        scale_hue = plot_kws.pop("scale_hue", deprecated)
        density_norm, common_norm = p._violin_scale_backcompat(
            scale, scale_hue, density_norm, common_norm,
        )

        # 对 bw 参数进行后向兼容处理，获取合适的 bw_method
        bw_method = p._violin_bw_backcompat(
            plot_kws.pop("bw", deprecated), plot_kws.pop("bw_method", "scott")
        )
        # 构建 kde_kws 字典，包含 cut、gridsize、bw_adjust 和 bw_method 参数
        kde_kws = dict(
            cut=plot_kws.pop("cut", 2),
            gridsize=plot_kws.pop("gridsize", 100),
            bw_adjust=plot_kws.pop("bw_adjust", 1),
            bw_method=bw_method,
        )

        # 从 plot_kws 中弹出 "inner_kws" 参数，并进行复制
        inner_kws = plot_kws.pop("inner_kws", {}).copy()
        # 从 plot_kws 中弹出 "linewidth" 参数，没有默认值
        linewidth = plot_kws.pop("linewidth", None)
        # 从 plot_kws 中弹出 "linecolor" 参数，默认为 "auto"，通过 _complement_color 方法获取颜色
        linecolor = plot_kws.pop("linecolor", "auto")
        # 通过 _complement_color 方法获取颜色
        linecolor = p._complement_color(linecolor, color, p._hue_map)

        # 调用 p 对象的 plot_violins 方法，传入相应的参数进行小提琴图绘制
        p.plot_violins(
            width=width,
            dodge=dodge,
            gap=gap,
            split=split,
            color=color,
            fill=fill,
            linecolor=linecolor,
            linewidth=linewidth,
            inner=inner,
            density_norm=density_norm,
            common_norm=common_norm,
            kde_kws=kde_kws,
            inner_kws=inner_kws,
            plot_kws=plot_kws,
        )
    # 如果图形类型为 "boxen"，则处理箱线图相关的参数和设置
    elif kind == "boxen":
        # 复制关键字参数，准备修改和使用
        plot_kws = kwargs.copy()
        # 提取并移除关键字参数中的 "gap"，默认为 0
        gap = plot_kws.pop("gap", 0)
        # 提取并移除关键字参数中的 "fill"，默认为 True
        fill = plot_kws.pop("fill", True)
        # 提取并移除关键字参数中的 "linecolor"，默认为 "auto"
        linecolor = plot_kws.pop("linecolor", "auto")
        # 提取并移除关键字参数中的 "linewidth"
        linewidth = plot_kws.pop("linewidth", None)
        # 提取并移除关键字参数中的 "k_depth"，默认为 "tukey"
        k_depth = plot_kws.pop("k_depth", "tukey")
        # 提取并移除关键字参数中的 "width_method"，默认为 "exponential"
        width_method = plot_kws.pop("width_method", "exponential")
        # 提取并移除关键字参数中的 "outlier_prop"，默认为 0.007
        outlier_prop = plot_kws.pop("outlier_prop", 0.007)
        # 提取并移除关键字参数中的 "trust_alpha"，默认为 0.05
        trust_alpha = plot_kws.pop("trust_alpha", 0.05)
        # 提取并移除关键字参数中的 "showfliers"，默认为 True
        showfliers = plot_kws.pop("showfliers", True)
        # 提取并移除关键字参数中的 "box_kws"，默认为空字典
        box_kws = plot_kws.pop("box_kws", {})
        # 提取并移除关键字参数中的 "flier_kws"，默认为空字典
        flier_kws = plot_kws.pop("flier_kws", {})
        # 提取并移除关键字参数中的 "line_kws"，默认为空字典
        line_kws = plot_kws.pop("line_kws", {})
        
        # 如果关键字参数中有 "scale"，则调用 p._boxen_scale_backcompat 方法处理
        if "scale" in plot_kws:
            width_method = p._boxen_scale_backcompat(
                plot_kws["scale"], width_method
            )
        
        # 根据线条颜色的补充颜色计算方式，更新 linecolor 变量
        linecolor = p._complement_color(linecolor, color, p._hue_map)
        
        # 调用 p.plot_boxens 方法绘制箱线图
        p.plot_boxens(
            width=width,
            dodge=dodge,
            gap=gap,
            fill=fill,
            color=color,
            linecolor=linecolor,
            linewidth=linewidth,
            width_method=width_method,
            k_depth=k_depth,
            outlier_prop=outlier_prop,
            trust_alpha=trust_alpha,
            showfliers=showfliers,
            box_kws=box_kws,
            flier_kws=flier_kws,
            line_kws=line_kws,
            plot_kws=plot_kws,
        )

    # 如果图形类型为 "point"，则处理散点图相关的参数和设置
    elif kind == "point":
        # 使用指定的估算器和参数创建聚合器对象
        aggregator = agg_cls(estimator, errorbar, n_boot=n_boot, seed=seed)
        
        # 提取并移除关键字参数中的 "markers"，默认为默认值
        markers = kwargs.pop("markers", default)
        # 提取并移除关键字参数中的 "linestyles"，默认为默认值
        linestyles = kwargs.pop("linestyles", default)
        
        # 检查并处理废弃项，暂未解除废弃状态
        # capsize = kwargs.pop("capsize", 0)
        # err_kws = normalize_kwargs(kwargs.pop("err_kws", {}), mpl.lines.Line2D)
        
        # 使用 p._point_kwargs_backcompat 方法处理部分废弃的关键字参数
        p._point_kwargs_backcompat(
            kwargs.pop("scale", deprecated),
            kwargs.pop("join", deprecated),
            kwargs
        )
        
        # 使用 p._err_kws_backcompat 方法处理废弃的错误条关键字参数
        err_kws, capsize = p._err_kws_backcompat(
            normalize_kwargs(kwargs.pop("err_kws", {}), mpl.lines.Line2D),
            None,
            errwidth=kwargs.pop("errwidth", deprecated),
            capsize=kwargs.pop("capsize", 0),
        )
        
        # 调用 p.plot_points 方法绘制散点图
        p.plot_points(
            aggregator=aggregator,
            markers=markers,
            linestyles=linestyles,
            dodge=dodge,
            color=color,
            capsize=capsize,
            err_kws=err_kws,
            plot_kws=kwargs,
        )
    # 如果图的类型是条形图
    elif kind == "bar":

        # 使用指定的聚合器创建聚合对象，包括估算器、误差条、重抽样次数等参数
        aggregator = agg_cls(estimator, errorbar, n_boot=n_boot, seed=seed)

        # 处理并兼容旧版参数，并设置误差条的长度
        err_kws, capsize = p._err_kws_backcompat(
            normalize_kwargs(kwargs.pop("err_kws", {}), mpl.lines.Line2D),
            errcolor=kwargs.pop("errcolor", deprecated),
            errwidth=kwargs.pop("errwidth", deprecated),
            capsize=kwargs.pop("capsize", 0),
        )
        # 获取并移除参数中的 'gap' 和 'fill'，设置为默认值
        gap = kwargs.pop("gap", 0)
        fill = kwargs.pop("fill", True)

        # 调用条形图绘制函数，传入各种参数
        p.plot_bars(
            aggregator=aggregator,
            dodge=dodge,
            width=width,
            gap=gap,
            color=color,
            fill=fill,
            capsize=capsize,
            err_kws=err_kws,
            plot_kws=kwargs,
        )

    # 如果图的类型是计数图
    elif kind == "count":

        # 创建计数图的聚合对象，不包括误差条
        aggregator = EstimateAggregator("sum", errorbar=None)

        # 根据绘图方向设置计数轴
        count_axis = {"x": "y", "y": "x"}[p.orient]
        # 设置计数数据为1，表示每个条的高度
        p.plot_data[count_axis] = 1

        # 统计选项包括'count'、'percent'、'probability'、'proportion'，根据传入参数设置统计类型
        stat_options = ["count", "percent", "probability", "proportion"]
        stat = _check_argument("stat", stat_options, kwargs.pop("stat", "count"))
        # 根据统计类型调整计数数据
        p.variables[count_axis] = stat
        if stat != "count":
            denom = 100 if stat == "percent" else 1
            p.plot_data[count_axis] /= len(p.plot_data) / denom

        # 获取并移除参数中的 'gap' 和 'fill'，设置为默认值
        gap = kwargs.pop("gap", 0)
        fill = kwargs.pop("fill", True)

        # 调用条形图绘制函数，传入各种参数
        p.plot_bars(
            aggregator=aggregator,
            dodge=dodge,
            width=width,
            gap=gap,
            color=color,
            fill=fill,
            capsize=0,  # 计数图没有误差条
            err_kws={},  # 空字典作为误差条参数
            plot_kws=kwargs,
        )

    # 如果图的类型不在预期之列，抛出错误
    else:
        msg = (
            f"Invalid `kind`: {kind!r}. Options are 'strip', 'swarm', "
            "'box', 'boxen', 'violin', 'bar', 'count', and 'point'."
        )
        raise ValueError(msg)

    # 调整每个轴的分类轴标签
    for ax in g.axes.flat:
        p._adjust_cat_axis(ax, axis=p.orient)

    # 设置坐标轴标签
    g.set_axis_labels(p.variables.get("x"), p.variables.get("y"))
    # 设置图标题
    g.set_titles()
    # 调整布局以适应所有元素
    g.tight_layout()

    # 更新图例数据，删除现有的图例
    for ax in g.axes.flat:
        g._update_legend_data(ax)
        ax.legend_ = None

    # 根据需求自动添加图例
    if legend == "auto":
        show_legend = not p._redundant_hue and p.input_format != "wide"
    else:
        show_legend = bool(legend)
    if show_legend:
        g.add_legend(title=p.variables.get("hue"), label_order=hue_order)

    # 如果提供了数据，则用新的数据框替换 FacetGrid 中的数据
    if data is not None:
        g.data = data

    # 返回生成的 FacetGrid 对象
    return g
# 将 `catplot` 对象的文档字符串设为格式化的字符串，用于绘制分类图到 `FacetGrid` 上
catplot.__doc__ = dedent("""\
    Figure-level interface for drawing categorical plots onto a FacetGrid.

    This function provides access to several axes-level functions that
    show the relationship between a numerical and one or more categorical
    variables using one of several visual representations. The `kind`
    parameter selects the underlying axes-level function to use.

    Categorical scatterplots:

    - :func:`stripplot` (with `kind="strip"`; the default)
    - :func:`swarmplot` (with `kind="swarm"`)

    Categorical distribution plots:

    - :func:`boxplot` (with `kind="box"`)
    - :func:`violinplot` (with `kind="violin"`)
    - :func:`boxenplot` (with `kind="boxen"`)

    Categorical estimate plots:

    - :func:`pointplot` (with `kind="point"`)
    - :func:`barplot` (with `kind="bar"`)
    - :func:`countplot` (with `kind="count"`)

    Extra keyword arguments are passed to the underlying function, so you
    should refer to the documentation for each to see kind-specific options.

    {categorical_narrative}

    After plotting, the :class:`FacetGrid` with the plot is returned and can
    be used directly to tweak supporting plot details or add other layers.

    Parameters
    ----------
    {categorical_data}
    {input_params}
    row, col : names of variables in `data` or vector data
        Categorical variables that will determine the faceting of the grid.
    kind : str
        The kind of plot to draw, corresponds to the name of a categorical
        axes-level plotting function. Options are: "strip", "swarm", "box", "violin",
        "boxen", "point", "bar", or "count".
    {stat_api_params}
    {order_vars}
    row_order, col_order : lists of strings
        Order to organize the rows and/or columns of the grid in; otherwise the
        orders are inferred from the data objects.
    {col_wrap}
    {height}
    {aspect}
    {log_scale}
    {native_scale}
    {formatter}
    {orient}
    {color}
    {palette}
    {hue_norm}
    {legend}
    {legend_out}
    {share_xy}
    {margin_titles}
    facet_kws : dict
        Dictionary of other keyword arguments to pass to :class:`FacetGrid`.
    kwargs : key, value pairings
        Other keyword arguments are passed through to the underlying plotting
        function.

    Returns
    -------
    :class:`FacetGrid`
        Returns the :class:`FacetGrid` object with the plot on it for further
        tweaking.

    Examples
    --------
    .. include:: ../docstrings/catplot.rst

    """).format(**_categorical_docs)


class Beeswarm:
    """Modifies a scatterplot artist to show a beeswarm plot."""
    def __init__(self, orient="x", width=0.8, warn_thresh=.05):
        # 初始化 Beeswarm 对象，设置方向（默认水平方向）、点群的宽度（默认为0.8）、警告阈值（默认为0.05）
        self.orient = orient
        self.width = width
        self.warn_thresh = warn_thresh
    def __call__(self, points, center):
        """
        Swarm `points`, a PathCollection, around the `center` position.
        """
        # Convert from point size (area) to diameter

        # 获取图表的坐标轴对象
        ax = points.axes
        # 获取图表的 DPI（每英寸点数）
        dpi = ax.figure.dpi

        # 获取点集合的原始位置数据
        orig_xy_data = points.get_offsets()

        # 将分类轴的位置重置为中心线
        cat_idx = 1 if self.orient == "y" else 0
        orig_xy_data[:, cat_idx] = center

        # 将数据坐标转换为点集坐标
        orig_x_data, orig_y_data = orig_xy_data.T
        orig_xy = ax.transData.transform(orig_xy_data)

        # 如果 orient 是 "y"，重新排序变量，使 x 成为分类轴
        if self.orient == "y":
            orig_xy = orig_xy[:, [1, 0]]

        # 添加一个列，表示每个点的半径
        sizes = points.get_sizes()
        if sizes.size == 1:
            sizes = np.repeat(sizes, orig_xy.shape[0])
        edge = points.get_linewidth().item()
        radii = (np.sqrt(sizes) + edge) / 2 * (dpi / 72)
        orig_xy = np.c_[orig_xy, radii]

        # 沿着值轴排序，以便进行 beeswarm 调整
        sorter = np.argsort(orig_xy[:, 1])
        orig_xyr = orig_xy[sorter]

        # 调整沿着分类轴的点，以防止重叠
        new_xyr = np.empty_like(orig_xyr)
        new_xyr[sorter] = self.beeswarm(orig_xyr)

        # 将点坐标转换回数据坐标
        if self.orient == "y":
            new_xy = new_xyr[:, [1, 0]]
        else:
            new_xy = new_xyr[:, :2]
        new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T

        # 添加 gutter（边距）
        t_fwd, t_inv = _get_transform_functions(ax, self.orient)
        if self.orient == "y":
            self.add_gutters(new_y_data, center, t_fwd, t_inv)
        else:
            self.add_gutters(new_x_data, center, t_fwd, t_inv)

        # 重新定位点，以避免重叠
        if self.orient == "y":
            points.set_offsets(np.c_[orig_x_data, new_y_data])
        else:
            points.set_offsets(np.c_[new_x_data, orig_y_data])
    def beeswarm(self, orig_xyr):
        """Adjust x position of points to avoid overlaps."""
        # 在这个方法中，`x` 始终是分类轴

        # Swarm 的中心，以点的坐标表示
        midline = orig_xyr[0, 0]

        # 用第一个点开始 Swarm
        swarm = np.atleast_2d(orig_xyr[0])

        # 遍历剩余的点
        for xyr_i in orig_xyr[1:]:
            # 找出可能与当前放置点重叠的 Swarm 中的点
            neighbors = self.could_overlap(xyr_i, swarm)

            # 找到可能在每个 Swarm 邻居之间单独有效的位置
            candidates = self.position_candidates(xyr_i, neighbors)

            # 根据它们与中心的距离排序候选项
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]

            # 找到第一个不与任何邻居重叠的候选项
            new_xyr_i = self.first_non_overlapping_candidate(candidates, neighbors)

            # 将其放入 Swarm 中
            swarm = np.vstack([swarm, new_xyr_i])

        return swarm

    def could_overlap(self, xyr_i, swarm):
        """Return a list of all swarm points that could overlap with target."""
        # 因为我们通过 Swarm 向后工作并可以短路，for 循环比向量化更快速
        _, y_i, r_i = xyr_i
        neighbors = []
        for xyr_j in reversed(swarm):
            _, y_j, r_j = xyr_j
            if (y_i - y_j) < (r_i + r_j):
                neighbors.append(xyr_j)
            else:
                break
        return np.array(neighbors)[::-1]

    def position_candidates(self, xyr_i, neighbors):
        """Return a list of coordinates that might be valid by adjusting x."""
        candidates = [xyr_i]
        x_i, y_i, r_i = xyr_i
        left_first = True
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy ** 2, 0)) * 1.05
            cl, cr = (x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i)
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)
    def first_non_overlapping_candidate(self, candidates, neighbors):
        """
        Find the first candidate that does not overlap with the swarm.
        """

        # If we have no neighbors, all candidates are good.
        if len(neighbors) == 0:
            return candidates[0]

        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]
        neighbors_r = neighbors[:, 2]

        for xyr_i in candidates:
            # Extract candidate's coordinates and radius
            x_i, y_i, r_i = xyr_i

            # Calculate differences in x and y coordinates with neighbors
            dx = neighbors_x - x_i
            dy = neighbors_y - y_i

            # Calculate squared distances between candidate and neighbors
            sq_distances = np.square(dx) + np.square(dy)

            # Calculate required separation distance squared
            sep_needed = np.square(neighbors_r + r_i)

            # Check if candidate overlaps with any neighbor
            # Good candidate does not overlap any neighbor if all squared distances are greater than or equal to sep_needed
            good_candidate = np.all(sq_distances >= sep_needed)

            if good_candidate:
                return xyr_i

        # If no non-overlapping candidate is found, raise an error
        raise RuntimeError(
            "No non-overlapping candidates found. This should not happen."
        )

    def add_gutters(self, points, center, trans_fwd, trans_inv):
        """
        Stop points from extending beyond their territory by adjusting them within the gutters.
        """
        # Calculate half of the width
        half_width = self.width / 2

        # Calculate the low and high gutter boundaries
        low_gutter = trans_inv(trans_fwd(center) - half_width)
        off_low = points < low_gutter
        if off_low.any():
            points[off_low] = low_gutter

        high_gutter = trans_inv(trans_fwd(center) + half_width)
        off_high = points > high_gutter
        if off_high.any():
            points[off_high] = high_gutter

        # Calculate the proportion of points that required adjustment
        gutter_prop = (off_high + off_low).sum() / len(points)

        # Issue a warning if the proportion of adjusted points exceeds a threshold
        if gutter_prop > self.warn_thresh:
            msg = (
                "{:.1%} of the points cannot be placed; you may want "
                "to decrease the size of the markers or use stripplot."
            ).format(gutter_prop)
            warnings.warn(msg, UserWarning)

        # Return the adjusted points
        return points
# 使用命名元组创建 BoxPlotArtists 类型，包含 box, median, whiskers, caps, fliers, mean 六个字段
BoxPlotArtists = namedtuple("BoxPlotArtists", "box median whiskers caps fliers mean")

# 定义 BoxPlotContainer 类
class BoxPlotContainer:

    # 初始化方法，接收一个包含各种绘图元素的字典 artist_dict
    def __init__(self, artist_dict):
        # 从 artist_dict 中获取各种绘图元素并分别赋值给实例变量
        self.boxes = artist_dict["boxes"]
        self.medians = artist_dict["medians"]
        self.whiskers = artist_dict["whiskers"]
        self.caps = artist_dict["caps"]
        self.fliers = artist_dict["fliers"]
        self.means = artist_dict["means"]

        # 初始化私有变量
        self._label = None
        # 将所有绘图元素合并到一个列表中，并赋值给 _children
        self._children = [
            *self.boxes,
            *self.medians,
            *self.whiskers,
            *self.caps,
            *self.fliers,
            *self.means,
        ]

    # 返回对象的字符串表示，包含 boxes 数量的信息
    def __repr__(self):
        return f"<BoxPlotContainer object with {len(self.boxes)} boxes>"

    # 实现索引访问，返回对应索引的 BoxPlotArtists 对象
    def __getitem__(self, idx):
        # 计算对应索引在各列表中的切片
        pair_slice = slice(2 * idx, 2 * idx + 2)
        # 返回一个 BoxPlotArtists 对象，包含对应索引的绘图元素
        return BoxPlotArtists(
            self.boxes[idx] if self.boxes else [],
            self.medians[idx] if self.medians else [],
            self.whiskers[pair_slice] if self.whiskers else [],
            self.caps[pair_slice] if self.caps else [],
            self.fliers[idx] if self.fliers else [],
            self.means[idx] if self.means else [],
        )

    # 实现迭代器，依次返回各个索引对应的 BoxPlotArtists 对象
    def __iter__(self):
        yield from (self[i] for i in range(len(self.boxes)))

    # 获取对象的标签值
    def get_label(self):
        return self._label

    # 设置对象的标签值
    def set_label(self, value):
        self._label = value

    # 获取所有子元素（绘图元素）的列表
    def get_children(self):
        return self._children

    # 移除所有子元素
    def remove(self):
        for child in self._children:
            child.remove()
```