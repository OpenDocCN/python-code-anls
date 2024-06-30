# `D:\src\scipysrc\seaborn\seaborn\distributions.py`

```
# 导入必要的库和模块，用于绘制数据分布的可视化图表
"""Plotting functions for visualizing distributions."""
from numbers import Number  # 导入 Number 类型，用于数值判断
from functools import partial  # 导入 partial 函数，用于创建部分应用函数
import math  # 导入 math 库，用于数学运算
import textwrap  # 导入 textwrap 模块，用于文本包装
import warnings  # 导入 warnings 模块，用于警告处理

import numpy as np  # 导入 numpy 库，用于数值计算
import pandas as pd  # 导入 pandas 库，用于数据处理
import matplotlib as mpl  # 导入 matplotlib 库的基础模块
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import matplotlib.transforms as tx  # 导入 matplotlib.transforms 模块，用于转换操作
from matplotlib.cbook import normalize_kwargs  # 从 matplotlib.cbook 导入 normalize_kwargs 函数，用于处理关键字参数
from matplotlib.colors import to_rgba  # 导入 to_rgba 函数，用于颜色转换
from matplotlib.collections import LineCollection  # 导入 LineCollection 类，用于绘制线集合

from ._base import VectorPlotter  # 从本地模块导入 VectorPlotter 类，用于基础绘图操作

# We have moved univariate histogram computation over to the new Hist class,
# but still use the older Histogram for bivariate computation.
# 导入统计相关的类和函数
from ._statistics import ECDF, Histogram, KDE  # 导入 ECDF、Histogram、KDE 类
from ._stats.counting import Hist  # 从统计计数模块导入 Hist 类

from .axisgrid import (  # 导入 axisgrid 模块的相关函数和类
    FacetGrid,  # 导入 FacetGrid 类，用于创建多面板图
    _facet_docs,  # 导入 _facet_docs，用于多面板图的文档
)
from .utils import (  # 导入工具函数
    remove_na,  # 导入 remove_na 函数，用于去除缺失值
    _get_transform_functions,  # 导入 _get_transform_functions 函数，用于获取转换函数
    _kde_support,  # 导入 _kde_support 函数，用于 KDE 支持计算
    _check_argument,  # 导入 _check_argument 函数，用于参数检查
    _assign_default_kwargs,  # 导入 _assign_default_kwargs 函数，用于分配默认关键字参数
    _default_color,  # 导入 _default_color 函数，用于设置默认颜色
)
from .palettes import color_palette  # 从调色板模块导入 color_palette 函数，用于获取颜色方案
from .external import husl  # 导入 husl 库，用于颜色空间转换
from .external.kde import gaussian_kde  # 从外部导入高斯核密度估计函数 gaussian_kde
from ._docstrings import (  # 导入文档字符串相关模块
    DocstringComponents,  # 导入 DocstringComponents 类，用于文档字符串组件
    _core_docs,  # 导入核心文档字符串
)


__all__ = ["displot", "histplot", "kdeplot", "ecdfplot", "rugplot", "distplot"]

# ==================================================================================== #
# Module documentation
# ==================================================================================== #

# 定义分布参数字典 _dist_params，包含多个关键字参数的说明
_dist_params = dict(
    multiple="""
multiple : {{"layer", "stack", "fill"}}
    Method for drawing multiple elements when semantic mapping creates subsets.
    Only relevant with univariate data.
    """,
    log_scale="""
log_scale : bool or number, or pair of bools or numbers
    Set axis scale(s) to log. A single value sets the data axis for any numeric
    axes in the plot. A pair of values sets each axis independently.
    Numeric values are interpreted as the desired base (default 10).
    When `None` or `False`, seaborn defers to the existing Axes scale.
    """,
    legend="""
legend : bool
    If False, suppress the legend for semantic variables.
    """,
    cbar="""
cbar : bool
    If True, add a colorbar to annotate the color mapping in a bivariate plot.
    Note: Does not currently support plots with a ``hue`` variable well.
    """,
    cbar_ax="""
cbar_ax : :class:`matplotlib.axes.Axes`
    Pre-existing axes for the colorbar.
    """,
    cbar_kws="""
cbar_kws : dict
    Additional parameters passed to :meth:`matplotlib.figure.Figure.colorbar`.
    """,
)

# 定义参数文档字典 _param_docs，结合了核心参数和分布参数的文档说明
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    facets=DocstringComponents(_facet_docs),
    dist=DocstringComponents(_dist_params),
    kde=DocstringComponents.from_function_params(KDE.__init__),
    hist=DocstringComponents.from_function_params(Histogram.__init__),
    ecdf=DocstringComponents.from_function_params(ECDF.__init__),
)

# ==================================================================================== #
# Internal API
# ==================================================================================== #

# 定义一个继承自VectorPlotter的_DistributionPlotter类
class _DistributionPlotter(VectorPlotter):

    # 定义两种结构的默认值
    wide_structure = {"x": "@values", "hue": "@columns"}  # 宽格式结构
    flat_structure = {"x": "@values"}  # 平坦格式结构

    def __init__(
        self,
        data=None,
        variables={},
    ):
        # 调用父类VectorPlotter的构造函数初始化
        super().__init__(data=data, variables=variables)

    @property
    def univariate(self):
        """Return True if only x or y are used."""
        # 返回布尔值，指示是否仅使用了"x"或"y"
        # TODO 可能可以移到核心部分，但现在放在这里。
        # 我们希望在概念上清楚，univariate仅适用于"x/y"，而不适用于其他语义，其他语义可能存在。
        # 我们尚未就"x/y"的概念名称达成一致。
        return bool({"x", "y"} - set(self.variables))

    @property
    def data_variable(self):
        """Return the variable with data for univariate plots."""
        # 返回用于单变量绘图的数据变量
        # TODO 这也可以放在核心部分，但它应该有一个更好的名称。
        if not self.univariate:
            raise AttributeError("This is not a univariate plot")
        return {"x", "y"}.intersection(self.variables).pop()

    @property
    def has_xy_data(self):
        """Return True at least one of x or y is defined."""
        # 返回布尔值，指示至少定义了"x"或"y"中的一个
        # TODO 参考上面关于这个应该放在哪里的观点
        return bool({"x", "y"} & set(self.variables))

    def _add_legend(
        self,
        ax_obj, artist, fill, element, multiple, alpha, artist_kws, legend_kws,
    ):
        """Add artists that reflect semantic mappings and put them in a legend."""
        # 添加反映语义映射的艺术家，并将它们放入图例中
        handles = []
        labels = []
        for level in self._hue_map.levels:
            color = self._hue_map(level)

            kws = self._artist_kws(
                artist_kws, fill, element, multiple, color, alpha
            )

            # 由于柱状图的颜色循环集成的问题，将颜色添加到kws中，但在这里直接设置艺术家属性时会导致问题，因此在此处删除它
            if "facecolor" in kws:
                kws.pop("color", None)

            handles.append(artist(**kws))  # 创建艺术家对象并添加到handles列表
            labels.append(level)  # 添加标签到labels列表

        if isinstance(ax_obj, mpl.axes.Axes):
            # 如果ax_obj是一个Axes对象，则在其上添加图例
            ax_obj.legend(handles, labels, title=self.variables["hue"], **legend_kws)
        else:  # 否则，假设是一个FacetGrid。 TODO 改进此处的逻辑
            # 构造一个图例数据字典，将标签和handles对应起来
            legend_data = dict(zip(labels, handles))
            ax_obj.add_legend(
                legend_data,
                title=self.variables["hue"],
                label_order=self.var_levels["hue"],
                **legend_kws
            )
    def _artist_kws(self, kws, fill, element, multiple, color, alpha):
        """Handle differences between artists in filled/unfilled plots."""
        # 复制传入的关键字参数字典
        kws = kws.copy()
        # 如果需要填充图形
        if fill:
            # 标准化关键字参数，用于多边形集合的Matplotlib对象
            kws = normalize_kwargs(kws, mpl.collections.PolyCollection)
            # 设置默认的面颜色为给定颜色的RGBA值
            kws.setdefault("facecolor", to_rgba(color, alpha))

            # 如果元素类型为 "bars"
            if element == "bars":
                # 修复 bar() 方法在属性循环中的问题
                # 参考：https://github.com/matplotlib/matplotlib/issues/19385
                kws["color"] = "none"

            # 如果多个元素是 "stack" 或 "fill" 或者元素类型为 "bars"
            if multiple in ["stack", "fill"] or element == "bars":
                # 设置默认的边缘颜色为Matplotlib配置的补丁边缘颜色
                kws.setdefault("edgecolor", mpl.rcParams["patch.edgecolor"])
            else:
                # 否则设置默认的边缘颜色为给定颜色的完全不透明的RGBA值
                kws.setdefault("edgecolor", to_rgba(color, 1))
        # 如果不需要填充且元素类型为 "bars"
        elif element == "bars":
            # 设置面颜色为 "none"
            kws["facecolor"] = "none"
            # 设置边缘颜色为给定颜色的RGBA值
            kws["edgecolor"] = to_rgba(color, alpha)
        else:
            # 否则设置颜色为给定颜色的RGBA值
            kws["color"] = to_rgba(color, alpha)
        # 返回处理后的关键字参数字典
        return kws

    def _quantile_to_level(self, data, quantile):
        """Return data levels corresponding to quantile cuts of mass."""
        # 将 quantile 切割的质量对应的数据级别返回
        isoprop = np.asarray(quantile)
        # 展平数据并按降序排序
        values = np.ravel(data)
        sorted_values = np.sort(values)[::-1]
        # 归一化值，计算累积和，得到归一化后的累积值
        normalized_values = np.cumsum(sorted_values) / values.sum()
        # 在归一化后的累积值中搜索对应于 (1 - isoprop) 的位置索引
        idx = np.searchsorted(normalized_values, 1 - isoprop)
        # 根据索引取出对应的排序值，使用 "clip" 模式处理超出边界的情况
        levels = np.take(sorted_values, idx, mode="clip")
        # 返回结果数据级别
        return levels

    def _cmap_from_color(self, color):
        """Return a sequential colormap given a color seed."""
        # 根据颜色种子返回一个顺序色彩映射
        # 类似这里的许多其他内容，这个函数在广泛应用中很有用，
        # 但保持在这个类中以表明我并没有过多考虑它的具体实现细节……
        r, g, b, _ = to_rgba(color)
        # 将 RGB 转换为 HUSL 色彩空间的 H、S 值
        h, s, _ = husl.rgb_to_husl(r, g, b)
        # 在指定范围内生成用于构建颜色映射的均匀分布点
        xx = np.linspace(-1, 1, int(1.15 * 256))[:256]
        ramp = np.zeros((256, 3))
        ramp[:, 0] = h
        ramp[:, 1] = s * np.cos(xx)
        ramp[:, 2] = np.linspace(35, 80, 256)
        # 将 HUSL 转换为 RGB 并进行裁剪，得到颜色数组
        colors = np.clip([husl.husl_to_rgb(*hsl) for hsl in ramp], 0, 1)
        # 返回Matplotlib的 ListedColormap 对象，反转颜色数组
        return mpl.colors.ListedColormap(colors[::-1])

    def _default_discrete(self):
        """Find default values for discrete hist estimation based on variable type."""
        # 如果是单变量
        if self.univariate:
            # 判断数据变量的类型是否为 "categorical"
            discrete = self.var_types[self.data_variable] == "categorical"
        else:
            # 否则，判断 x 和 y 变量是否为 "categorical" 类型
            discrete_x = self.var_types["x"] == "categorical"
            discrete_y = self.var_types["y"] == "categorical"
            discrete = discrete_x, discrete_y
        # 返回判断结果
        return discrete
    # 修改密度数据结构以处理多个密度
    def _resolve_multiple(self, curves, multiple):
        """Modify the density data structure to handle multiple densities."""

        # 默认基线，所有密度从0开始
        baselines = {k: np.zeros_like(v) for k, v in curves.items()}

        # TODO 应该有一个中央清理站来检查是否已经分配了任何“分组”（术语？）语义
        if "hue" not in self.variables:
            # 如果变量中没有'hue'，直接返回原始曲线数据和基线
            return curves, baselines

        # 如果 multiple 是 'stack' 或 'fill'
        if multiple in ("stack", "fill"):

            # 将曲线数据转换为 DataFrame，并反转列的顺序以从顶部到底部绘制
            curves = pd.DataFrame(curves).iloc[:, ::-1]

            # 查找嵌套在列/行变量中的列分组
            column_groups = {}
            for i, keyd in enumerate(map(dict, curves.columns)):
                facet_key = keyd.get("col", None), keyd.get("row", None)
                column_groups.setdefault(facet_key, [])
                column_groups[facet_key].append(i)

            # 将基线初始化为曲线的复制
            baselines = curves.copy()

            # 对于每个列分组，进行堆叠或填充操作
            for col_idxs in column_groups.values():
                cols = curves.columns[col_idxs]

                # 计算每行的归一化常数
                norm_constant = curves[cols].sum(axis="columns")

                # 对曲线进行累积和以进行堆叠
                curves[cols] = curves[cols].cumsum(axis="columns")

                # 如果 multiple 是 'fill'，则按行总和进行归一化
                if multiple == "fill":
                    curves[cols] = curves[cols].div(norm_constant, axis="index")

                # 定义每个段的起始位置作为基线
                baselines[cols] = curves[cols].shift(1, axis=1).fillna(0)

        # 如果 multiple 是 'dodge'
        if multiple == "dodge":

            # 考虑唯一语义（非分面）级别的情况
            hue_levels = self.var_levels["hue"]
            n = len(hue_levels)
            f_fwd, f_inv = self._get_scale_transforms(self.data_variable)
            for key in curves:

                # 获取'hue'键的值
                level = dict(key)["hue"]
                hist = curves[key].reset_index(name="heights")
                level_idx = hue_levels.index(level)

                # 对边缘和宽度进行重新映射
                a = f_fwd(hist["edges"])
                b = f_fwd(hist["edges"] + hist["widths"])
                w = (b - a) / n
                new_min = f_inv(a + level_idx * w)
                new_max = f_inv(a + (level_idx + 1) * w)
                hist["widths"] = new_max - new_min
                hist["edges"] = new_min

                # 设置新的边缘和宽度作为曲线的新索引
                curves[key] = hist.set_index(["edges", "widths"])["heights"]

        # 返回修改后的曲线数据和基线数据
        return curves, baselines
    # 定义计算单变量密度的方法
    def _compute_univariate_density(
        self,
        data_variable,
        common_norm,
        common_grid,
        estimate_kws,
        warn_singular=True,
    ):
        # 初始化核密度估计器对象
        estimator = KDE(**estimate_kws)

        # 如果数据集除了"x"和"y"之外还有其他变量
        if set(self.variables) - {"x", "y"}:
            # 如果使用公共网格
            if common_grid:
                # 获取所有非空观测数据
                all_observations = self.comp_data.dropna()
                # 定义估计支持范围
                estimator.define_support(all_observations[data_variable])
        else:
            # 如果没有其他变量，则不使用常规网格
            common_norm = False

        # 获取所有数据并删除缺失值
        all_data = self.plot_data.dropna()

        # 计算整体权重
        if common_norm and "weights" in all_data:
            whole_weight = all_data["weights"].sum()
        else:
            whole_weight = len(all_data)

        # 存储密度估计结果的字典
        densities = {}

        # 遍历每个子集数据
        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

            # 提取该子集中的数据点并移除空值
            observations = sub_data[data_variable]

            # 提取该子集观测的权重
            if "weights" in self.variables:
                weights = sub_data["weights"]
                part_weight = weights.sum()
            else:
                weights = None
                part_weight = len(sub_data)

            # 估计该级别的观测密度
            variance = np.nan_to_num(observations.var())
            singular = len(observations) < 2 or math.isclose(variance, 0)
            try:
                if not singular:
                    # 因为数值失败可能表现出不同的方式，所以需要复杂的方法
                    density, support = estimator(observations, weights=weights)
            except np.linalg.LinAlgError:
                singular = True

            # 如果发现数据集是奇异的（例如方差为0），则跳过密度估计
            if singular:
                msg = (
                    "Dataset has 0 variance; skipping density estimate. "
                    "Pass `warn_singular=False` to disable this warning."
                )
                # 如果允许警告奇异数据集，则发出警告
                if warn_singular:
                    warnings.warn(msg, UserWarning, stacklevel=4)
                continue

            # 反转支持点的缩放
            _, f_inv = self._get_scale_transforms(self.data_variable)
            support = f_inv(support)

            # 如果使用公共标准化，则调整密度以保证所有子集的积分为1
            if common_norm:
                density *= part_weight / whole_weight

            # 存储该级别的密度结果
            key = tuple(sub_vars.items())
            densities[key] = pd.Series(density, index=support)

        # 返回所有级别的密度字典
        return densities

    # -------------------------------------------------------------------------------- #
    # 绘图
    # -------------------------------------------------------------------------------- #
    # 绘制单变量直方图的方法
    def plot_univariate_histogram(
        self,
        multiple,
        element,
        fill,
        common_norm,
        common_bins,
        shrink,
        kde,
        kde_kws,
        color,
        legend,
        line_kws,
        estimate_kws,
        **plot_kws,
    ):
        # 绘制双变量直方图的方法
        def plot_bivariate_histogram(
            self,
            common_bins,  # 公共的直方图箱数
            common_norm,  # 是否使用公共的归一化
            thresh,  # 阈值
            pthresh,  # p 值的阈值
            pmax,  # p 值的最大值
            color,  # 绘图的颜色
            legend,  # 是否显示图例
            cbar,  # 是否显示颜色条
            cbar_ax,  # 颜色条的轴对象
            cbar_kws,  # 颜色条的其他参数
            estimate_kws,  # 估计的其他参数
            **plot_kws,  # 绘图的其他参数
        ):
            # 绘制单变量核密度估计的方法
            def plot_univariate_density(
                self,
                multiple,  # 是否多个变量
                common_norm,  # 是否使用公共的归一化
                common_grid,  # 公共的网格
                warn_singular,  # 是否警告奇异
                fill,  # 是否填充
                color,  # 绘图的颜色
                legend,  # 是否显示图例
                estimate_kws,  # 估计的其他参数
                **plot_kws,  # 绘图的其他参数
            ):
                # 绘制双变量核密度估计的方法
                def plot_bivariate_density(
                    self,
                    common_norm,  # 是否使用公共的归一化
                    fill,  # 是否填充
                    levels,  # 等级
                    thresh,  # 阈值
                    color,  # 绘图的颜色
                    legend,  # 是否显示图例
                    cbar,  # 是否显示颜色条
                    warn_singular,  # 是否警告奇异
                    cbar_ax,  # 颜色条的轴对象
                    cbar_kws,  # 颜色条的其他参数
                    estimate_kws,  # 估计的其他参数
                    **contour_kws,  # 等高线的其他参数
                ):
                    pass  # 这里是方法的占位符，实际方法未定义或未列出详细注释
                # 返回空方法的占位符
                return plot_bivariate_density
            # 返回空方法的占位符
            return plot_univariate_density
        # 返回空方法的占位符
        return plot_bivariate_histogram
    # 返回空方法的占位符
    return plot_univariate_histogram
    def plot_univariate_ecdf(self, estimate_kws, legend, **plot_kws):
        # 创建 ECDF 估计器对象，使用传入的参数
        estimator = ECDF(**estimate_kws)

        # 根据数据变量设置绘图风格为步阶图
        drawstyles = dict(x="steps-post", y="steps-pre")
        plot_kws["drawstyle"] = drawstyles[self.data_variable]

        # 遍历数据子集，对数据进行转换和绘图
        for sub_vars, sub_data in self.iter_data(
            "hue", reverse=True, from_comp_data=True,
        ):

            # 计算 ECDF
            if sub_data.empty:
                continue

            # 获取观测值和权重（如果有）
            observations = sub_data[self.data_variable]
            weights = sub_data.get("weights", None)
            stat, vals = estimator(observations, weights=weights)

            # 根据语义映射设置艺术属性
            artist_kws = plot_kws.copy()
            if "hue" in self.variables:
                artist_kws["color"] = self._hue_map(sub_vars["hue"])

            # 将数据变量返回到线性域
            ax = self._get_axes(sub_vars)
            _, inv = _get_transform_functions(ax, self.data_variable)
            vals = inv(vals)

            # 在“对数”比例上手动设置最小值
            if isinstance(inv.__self__, mpl.scale.LogTransform):
                vals[0] = -np.inf

            # 确定绘图方向
            if self.data_variable == "x":
                plot_args = vals, stat
                stat_variable = "y"
            else:
                plot_args = stat, vals
                stat_variable = "x"

            # 如果估计器的统计量是“count”，则设置顶部边缘
            if estimator.stat == "count":
                top_edge = len(observations)
            else:
                top_edge = 1

            # 绘制当前子集的线条
            artist, = ax.plot(*plot_args, **artist_kws)
            sticky_edges = getattr(artist.sticky_edges, stat_variable)
            sticky_edges[:] = 0, top_edge

        # --- 完成图表的最后处理 ----
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        stat = estimator.stat.capitalize()
        default_x = default_y = ""
        if self.data_variable == "x":
            default_y = stat
        if self.data_variable == "y":
            default_x = stat
        # 添加坐标轴标签
        self._add_axis_labels(ax, default_x, default_y)

        # 如果数据中包含“hue”并且需要图例，则添加图例
        if "hue" in self.variables and legend:
            # 创建偏函数用于创建图例的艺术对象
            artist = partial(mpl.lines.Line2D, [], [])
            alpha = plot_kws.get("alpha", 1)
            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(
                ax_obj, artist, False, False, None, alpha, plot_kws, {},
            )
    def plot_rug(self, height, expand_margins, legend, **kws):
        # 对每个子数据和子变量进行迭代，从组件数据开始迭代
        for sub_vars, sub_data, in self.iter_data(from_comp_data=True):
            # 获取当前子变量对应的轴对象
            ax = self._get_axes(sub_vars)
            
            # 设置默认的线宽
            kws.setdefault("linewidth", 1)
            
            # 如果需要扩展边距
            if expand_margins:
                # 获取当前轴对象的边距
                xmarg, ymarg = ax.margins()
                # 根据变量类型调整边距
                if "x" in self.variables:
                    ymarg += height * 2
                if "y" in self.variables:
                    xmarg += height * 2
                # 更新轴对象的边距
                ax.margins(x=xmarg, y=ymarg)
            
            # 如果变量中包含 "hue"
            if "hue" in self.variables:
                # 移除可能存在的颜色参数
                kws.pop("c", None)
                kws.pop("color", None)
            
            # 如果变量中包含 "x"
            if "x" in self.variables:
                # 绘制 x 轴方向的 rugplot
                self._plot_single_rug(sub_data, "x", height, ax, kws)
            # 如果变量中包含 "y"
            if "y" in self.variables:
                # 绘制 y 轴方向的 rugplot
                self._plot_single_rug(sub_data, "y", height, ax, kws)
            
            # --- 完成图形的最终设置
            # 添加轴标签
            self._add_axis_labels(ax)
            # 如果变量中包含 "hue" 并且需要图例
            if "hue" in self.variables and legend:
                # TODO 最好让图例的图标看起来像地毯
                # 创建一个类似于 rug 的图例对象
                legend_artist = partial(mpl.lines.Line2D, [], [])
                # 添加图例
                self._add_legend(
                    ax, legend_artist, False, False, None, 1, {}, {},
                )

    def _plot_single_rug(self, sub_data, var, height, ax, kws):
        """沿着图中某一轴绘制 rugplot."""
        # 获取对应轴上的数据向量
        vector = sub_data[var]
        n = len(vector)

        # 将数据返回到线性域
        _, inv = _get_transform_functions(ax, var)
        vector = inv(vector)

        # 总是添加一个具有不同颜色的单一集合
        if "hue" in self.variables:
            # 根据 "hue" 变量映射颜色
            colors = self._hue_map(sub_data["hue"])
        else:
            colors = None

        # 构建 LineCollection 所需的数值数组
        if var == "x":
            # 使用混合变换工厂创建转换对象
            trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
            # 构建 xy 对数组
            xy_pairs = np.column_stack([
                np.repeat(vector, 2), np.tile([0, height], n)
            ])

        if var == "y":
            # 使用混合变换工厂创建转换对象
            trans = tx.blended_transform_factory(ax.transAxes, ax.transData)
            # 构建 xy 对数组
            xy_pairs = np.column_stack([
                np.tile([0, height], n), np.repeat(vector, 2)
            ])

        # 在图中绘制线条
        line_segs = xy_pairs.reshape([n, 2, 2])
        ax.add_collection(LineCollection(
            line_segs, transform=trans, colors=colors, **kws
        ))

        # 自动调整视图
        ax.autoscale_view(scalex=var == "x", scaley=var == "y")
# ==================================================================================== #
# External API
# ==================================================================================== #

# 定义绘制直方图的函数，支持单变量或双变量直方图
def histplot(
    data=None, *,
    # 向量变量
    x=None, y=None, hue=None, weights=None,
    # 直方图计算参数
    stat="count", bins="auto", binwidth=None, binrange=None,
    discrete=None, cumulative=False, common_bins=True, common_norm=True,
    # 直方图外观参数
    multiple="layer", element="bars", fill=True, shrink=1,
    # 使用核密度估计进行直方图平滑
    kde=False, kde_kws=None, line_kws=None,
    # 双变量直方图参数
    thresh=0, pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None,
    # 色调映射参数
    palette=None, hue_order=None, hue_norm=None, color=None,
    # 坐标轴信息
    log_scale=None, legend=True, ax=None,
    # 其他外观关键字参数
    **kwargs,
):

    # 创建 _DistributionPlotter 对象来处理数据和变量
    p = _DistributionPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue, weights=weights),
    )

    # 根据色调映射参数对色调进行映射
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    # 如果未提供绘图的坐标轴，则使用当前的坐标轴
    if ax is None:
        ax = plt.gca()

    # 将 _DistributionPlotter 对象连接到指定的坐标轴上
    p._attach(ax, log_scale=log_scale)

    # 如果是单变量直方图，则执行以下操作
    if p.univariate:  # 注意，双变量图不会循环处理
        if fill:
            # 如果指定填充，则选择绘制条形图或填充区域
            method = ax.bar if element == "bars" else ax.fill_between
        else:
            # 否则选择绘制线条
            method = ax.plot
        # 获取默认颜色
        color = _default_color(method, hue, color, kwargs)

    # 如果没有 x 和 y 数据，则直接返回坐标轴
    if not p.has_xy_data:
        return ax

    # 默认对分类变量使用离散的分组
    if discrete is None:
        discrete = p._default_discrete()

    # 设置估计参数
    estimate_kws = dict(
        stat=stat,
        bins=bins,
        binwidth=binwidth,
        binrange=binrange,
        discrete=discrete,
        cumulative=cumulative,
    )

    # 如果是单变量直方图，则执行单变量直方图绘制函数
    if p.univariate:
        p.plot_univariate_histogram(
            multiple=multiple,
            element=element,
            fill=fill,
            shrink=shrink,
            common_norm=common_norm,
            common_bins=common_bins,
            kde=kde,
            kde_kws=kde_kws,
            color=color,
            legend=legend,
            estimate_kws=estimate_kws,
            line_kws=line_kws,
            **kwargs,
        )

    # 如果是双变量直方图，则执行双变量直方图绘制函数
    else:
        p.plot_bivariate_histogram(
            common_bins=common_bins,
            common_norm=common_norm,
            thresh=thresh,
            pthresh=pthresh,
            pmax=pmax,
            color=color,
            legend=legend,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            estimate_kws=estimate_kws,
            **kwargs,
        )

    # 返回绘制后的坐标轴对象
    return ax


# 设置 histplot 函数的文档字符串，描述其功能和使用方法
histplot.__doc__ = """\
Plot univariate or bivariate histograms to show distributions of datasets.

A histogram is a classic visualization tool that represents the distribution
of one or more variables by counting the number of observations that fall within
discrete bins.
"""
# 此函数用于规范化每个箱内计算的统计量，以估计频率、密度或概率质量，并且可以添加通过核密度估计获取的平滑曲线，类似于 :func:`kdeplot`。

更多信息请参考 :ref:`用户指南 <tutorial_hist>`。

参数
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
weights : vector 或 ``data`` 中的键
    如果提供，按照这些因子加权对应数据点在每个箱中的贡献计数。
{params.hist.stat}
{params.hist.bins}
{params.hist.binwidth}
{params.hist.binrange}
discrete : bool
    如果为 True，则默认 ``binwidth=1``，并且绘制条形图时它们会居中于相应的数据点。这样可以避免使用离散（整数）数据时可能出现的“间隙”。
cumulative : bool
    如果为 True，则绘制累积计数随着箱增加的图表。
common_bins : bool
    如果为 True，当语义变量产生多个图时使用相同的箱。如果使用参考规则确定箱，则将使用整个数据集计算。
common_norm : bool
    如果为 True 并且使用标准化统计量，则标准化将应用于整个数据集。否则，将独立地对每个直方图进行标准化。
multiple : {{"layer", "dodge", "stack", "fill"}}
    解决语义映射创建子集时多个元素的方法。仅适用于单变量数据。
element : {{"bars", "step", "poly"}}
    直方图统计量的视觉表示。仅适用于单变量数据。
fill : bool
    如果为 True，则填充直方图下方的空间。仅适用于单变量数据。
shrink : number
    将每个条的宽度相对于 binwidth 缩放此因子。仅适用于单变量数据。
kde : bool
    如果为 True，则计算核密度估计以平滑分布，并显示在图中作为一条或多条线。
    仅适用于单变量数据。
kde_kws : dict
    控制 KDE 计算的参数，如 :func:`kdeplot` 中所述。
line_kws : dict
    控制 KDE 可视化的参数，传递给 :meth:`matplotlib.axes.Axes.plot`。
thresh : number 或 None
    统计量小于或等于此值的单元格将是透明的。仅适用于双变量数据。
pthresh : number 或 None
    类似于 ``thresh``，但是一个 [0, 1] 范围内的值，使得累积计数（或其他统计量，当使用时）达到总数的这一比例将是透明的。
pmax : number 或 None
    [0, 1] 范围内的值，设置颜色映射的饱和点，以便低于此比例的单元格（或其他统计量，当使用时）将构成此比例。
{params.dist.cbar}
{params.dist.cbar_ax}
{params.dist.cbar_kws}
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}
{params.core.color}
{params.dist.log_scale}
{params.dist.legend}
{params.core.ax}
kwargs
    Other keyword arguments are passed to one of the following matplotlib
    functions:

    - :meth:`matplotlib.axes.Axes.bar` (univariate, element="bars")
    - :meth:`matplotlib.axes.Axes.fill_between` (univariate, other element, fill=True)
    - :meth:`matplotlib.axes.Axes.plot` (univariate, other element, fill=False)
    - :meth:`matplotlib.axes.Axes.pcolormesh` (bivariate)
    # --- Start with backwards compatability for versions < 0.11.0 ----------------

    # 处理 `data2` 的过时警告
    if "data2" in kwargs:
        msg = "`data2` 已移除（被 `y` 替代）；请更新您的代码。"
        raise TypeError(msg)

    # 处理 `vertical` 参数的过时警告
    vertical = kwargs.pop("vertical", None)
    if vertical is not None:
        if vertical:
            action_taken = "将数据分配给 `y`。"
            if x is None:
                data, y = y, data
            else:
                x, y = y, x
        else:
            action_taken = "将数据分配给 `x`。"
        msg = textwrap.dedent(f"""\n
        `vertical` 参数已弃用；{action_taken}
        这将在 seaborn v0.14.0 中成为错误；请更新您的代码。
        """)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # 处理 `bw` 参数的过时警告
    bw = kwargs.pop("bw", None)
    if bw is not None:
        msg = textwrap.dedent(f"""\n
        `bw` 参数已弃用，请使用 `bw_method` 和 `bw_adjust`。
        设置 `bw_method={bw}`，但请查看文档获取新参数信息并更新您的代码。这将在 seaborn v0.14.0 中成为错误。
        """)
        warnings.warn(msg, UserWarning, stacklevel=2)
        bw_method = bw

    # 处理其他未指定的关键字参数
    # 如果 kwargs 中有 "kernel" 参数，则弹出并检查其值
    if kwargs.pop("kernel", None) is not None:
        # 构造警告信息，指出对于替代核的支持已被移除，并将使用高斯核
        msg = textwrap.dedent("""\n
        Support for alternate kernels has been removed; using Gaussian kernel.
        This will become an error in seaborn v0.14.0; please update your code.
        """)
        # 发出警告
        warnings.warn(msg, UserWarning, stacklevel=2)

    # 处理 "shade_lowest" 参数的弃用
    shade_lowest = kwargs.pop("shade_lowest", None)
    if shade_lowest is not None:
        if shade_lowest:
            # 如果 shade_lowest 为 True，则设置 thresh 为 0
            thresh = 0
        # 构造警告信息，指出 "shade_lowest" 已被 "thresh" 取代
        msg = textwrap.dedent(f"""\n
        `shade_lowest` has been replaced by `thresh`; setting `thresh={thresh}.
        This will become an error in seaborn v0.14.0; please update your code.
        """)
        # 发出警告
        warnings.warn(msg, UserWarning, stacklevel=2)

    # 处理 "shade" 参数的弃用
    shade = kwargs.pop("shade", None)
    if shade is not None:
        # 将 "shade" 的值赋给 "fill"，并构造警告信息
        fill = shade
        msg = textwrap.dedent(f"""\n
        `shade` is now deprecated in favor of `fill`; setting `fill={shade}`.
        This will become an error in seaborn v0.14.0; please update your code.
        """)
        # 发出 FutureWarning
        warnings.warn(msg, FutureWarning, stacklevel=2)

    # 处理 "n_levels" 参数，将其作为别名处理，并准备将来弃用
    levels = kwargs.pop("n_levels", levels)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # 创建 _DistributionPlotter 对象 p，传入相关数据和变量
    p = _DistributionPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue, weights=weights),
    )

    # 设置调色板、hue_order 和 hue_norm
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    # 如果未指定 ax，则使用当前图形的轴
    if ax is None:
        ax = plt.gca()

    # 将数据附加到轴上，仅允许 "numeric" 和 "datetime" 类型的数据，支持对数刻度
    p._attach(ax, allowed_types=["numeric", "datetime"], log_scale=log_scale)

    # 根据 fill 参数选择方法（填充或绘制）
    method = ax.fill_between if fill else ax.plot

    # 确定默认颜色
    color = _default_color(method, hue, color, kwargs)

    # 如果没有 x、y 数据，则直接返回轴
    if not p.has_xy_data:
        return ax

    # 准备传递给 statistics.KDE 的参数字典
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )
    # 如果是单变量情况
    if p.univariate:
        # 复制关键字参数，以避免直接修改原始参数
        plot_kws = kwargs.copy()
        # 调用对象 p 的单变量密度绘制方法
        p.plot_univariate_density(
            multiple=multiple,        # 是否多重绘制
            common_norm=common_norm,  # 是否使用公共的标准化
            common_grid=common_grid,  # 是否使用公共的网格
            fill=fill,                # 是否填充区域
            color=color,              # 绘图颜色
            legend=legend,            # 是否显示图例
            warn_singular=warn_singular,  # 是否警告奇异情况
            estimate_kws=estimate_kws,    # 估计关键字参数
            **plot_kws,               # 其他传递给绘图方法的关键字参数
        )
    
    else:
        # 调用对象 p 的双变量密度绘制方法
        p.plot_bivariate_density(
            common_norm=common_norm,   # 是否使用公共的标准化
            fill=fill,                 # 是否填充区域
            levels=levels,             # 绘制的密度等级
            thresh=thresh,             # 绘制的阈值
            legend=legend,             # 是否显示图例
            color=color,               # 绘图颜色
            warn_singular=warn_singular,  # 是否警告奇异情况
            cbar=cbar,                 # 是否显示颜色条
            cbar_ax=cbar_ax,           # 颜色条的轴
            cbar_kws=cbar_kws,         # 颜色条的关键字参数
            estimate_kws=estimate_kws,  # 估计关键字参数
            **kwargs,                  # 其他传递给绘图方法的关键字参数
        )
    
    # 返回绘图对象 ax
    return ax
# 设置 kdeplot 的文档字符串，用于描述核密度估计绘图函数的功能和参数用法
kdeplot.__doc__ = """\
Plot univariate or bivariate distributions using kernel density estimation.

A kernel density estimate (KDE) plot is a method for visualizing the
distribution of observations in a dataset, analogous to a histogram. KDE
represents the data using a continuous probability density curve in one or
more dimensions.

The approach is explained further in the :ref:`user guide <tutorial_kde>`.

Relative to a histogram, KDE can produce a plot that is less cluttered and
more interpretable, especially when drawing multiple distributions. But it
has the potential to introduce distortions if the underlying distribution is
bounded or not smooth. Like a histogram, the quality of the representation
also depends on the selection of good smoothing parameters.

Parameters
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
weights : vector or key in ``data``
    If provided, weight the kernel density estimation using these values.
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}
{params.core.color}
fill : bool or None
    If True, fill in the area under univariate density curves or between
    bivariate contours. If None, the default depends on ``multiple``.
{params.dist.multiple}
common_norm : bool
    If True, scale each conditional density by the number of observations
    such that the total area under all densities sums to 1. Otherwise,
    normalize each density independently.
common_grid : bool
    If True, use the same evaluation grid for each kernel density estimate.
    Only relevant with univariate data.
{params.kde.cumulative}
{params.kde.bw_method}
{params.kde.bw_adjust}
warn_singular : bool
    If True, issue a warning when trying to estimate the density of data
    with zero variance.
{params.dist.log_scale}
levels : int or vector
    Number of contour levels or values to draw contours at. A vector argument
    must have increasing values in [0, 1]. Levels correspond to iso-proportions
    of the density: e.g., 20% of the probability mass will lie below the
    contour drawn for 0.2. Only relevant with bivariate data.
thresh : number in [0, 1]
    Lowest iso-proportion level at which to draw a contour line. Ignored when
    ``levels`` is a vector. Only relevant with bivariate data.
gridsize : int
    Number of points on each dimension of the evaluation grid.
{params.kde.cut}
{params.kde.clip}
{params.dist.legend}
{params.dist.cbar}
{params.dist.cbar_ax}
{params.dist.cbar_kws}
{params.core.ax}
kwargs
    Other keyword arguments are passed to one of the following matplotlib
    functions:

    - :meth:`matplotlib.axes.Axes.plot` (univariate, ``fill=False``),
    - :meth:`matplotlib.axes.Axes.fill_between` (univariate, ``fill=True``),
    - :meth:`matplotlib.axes.Axes.contour` (bivariate, ``fill=False``),
    - :meth:`matplotlib.axes.contourf` (bivariate, ``fill=True``).

Returns
-------
{returns.ax}

See Also
--------
{seealso.displot}
{seealso.histplot}
{seealso.ecdfplot}
{seealso.jointplot}
"""
# 定义了一个函数 ecdfplot，用于绘制经验累积分布函数图(ECDF plot)
def ecdfplot(
    data=None, *,
    # 向量变量
    x=None, y=None, hue=None, weights=None,
    # 计算参数
    stat="proportion", complementary=False,
    # 色调映射参数
    palette=None, hue_order=None, hue_norm=None,
    # 坐标轴信息
    log_scale=None, legend=True, ax=None,
    # 其他外观关键词
    **kwargs,
):
    # 创建一个 DistributionPlotter 对象 p，用于数据分布的绘制
    p = _DistributionPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue, weights=weights),
    )

    # 根据参数设置色调映射
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    # 如果未指定绘图的坐标轴，则使用当前坐标轴
    if ax is None:
        ax = plt.gca()

    # 将 DistributionPlotter 对象 p 附加到指定的坐标轴上，设置是否对数化
    p._attach(ax, log_scale=log_scale)
    # 从关键字参数中获取颜色信息，支持"color"和"c"两种键名，如果不存在则默认为None
    color = kwargs.pop("color", kwargs.pop("c", None))
    # 将处理后的颜色信息重新放入kwargs字典中，调用_default_color函数处理默认颜色逻辑
    kwargs["color"] = _default_color(ax.plot, hue, color, kwargs)

    # 如果图形对象p没有XY数据，则直接返回当前的坐标轴对象ax
    if not p.has_xy_data:
        return ax

    # 对于双变量情况，暂未实现双变量ECDF图形，抛出未实现错误
    # 目前仅支持单变量ECDF图形
    if not p.univariate:
        raise NotImplementedError("Bivariate ECDF plots are not implemented")

    # 构建估计参数字典estimate_kws
    estimate_kws = dict(
        stat=stat,  # 统计量名称
        complementary=complementary,  # 补充参数
    )

    # 调用p对象的plot_univariate_ecdf方法进行单变量ECDF图形绘制
    # 将估计参数字典estimate_kws和其他关键字参数传递给plot_univariate_ecdf方法
    p.plot_univariate_ecdf(
        estimate_kws=estimate_kws,  # 传递估计参数字典
        legend=legend,  # 是否显示图例
        **kwargs,  # 其他所有关键字参数
    )

    # 返回更新后的坐标轴对象ax
    return ax
# 设置 ecdfplot 的文档字符串，用于描述 ECDF 绘图函数的功能、参数和返回值
ecdfplot.__doc__ = """\
Plot empirical cumulative distribution functions.

An ECDF represents the proportion or count of observations falling below each
unique value in a dataset. Compared to a histogram or density plot, it has the
advantage that each observation is visualized directly, meaning that there are
no binning or smoothing parameters that need to be adjusted. It also aids direct
comparisons between multiple distributions. A downside is that the relationship
between the appearance of the plot and the basic properties of the distribution
(such as its central tendency, variance, and the presence of any bimodality)
may not be as intuitive.

More information is provided in the :ref:`user guide <tutorial_ecdf>`.

Parameters
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
weights : vector or key in ``data``
    If provided, weight the contribution of the corresponding data points
    towards the cumulative distribution using these values.
{params.ecdf.stat}
{params.ecdf.complementary}
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}
{params.dist.log_scale}
{params.dist.legend}
{params.core.ax}
kwargs
    Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.plot`.

Returns
-------
{returns.ax}

See Also
--------
{seealso.displot}
{seealso.histplot}
{seealso.kdeplot}
{seealso.rugplot}

Examples
--------

.. include:: ../docstrings/ecdfplot.rst

""".format(
    params=_param_docs,   # 描述参数的文档字符串
    returns=_core_docs["returns"],   # 描述返回值的文档字符串
    seealso=_core_docs["seealso"],   # 描述相关函数的文档字符串
)


def rugplot(
    data=None, *, x=None, y=None, hue=None, height=.025, expand_margins=True,
    palette=None, hue_order=None, hue_norm=None, legend=True, ax=None, **kwargs
):
    """
    Plot rugplot on a matplotlib axis.

    Parameters
    ----------
    data : DataFrame, array, or list
        Input data structure.
    x, y, hue : names of variables in ``data`` or vector data, optional
        Variables that specify positions on the x and y axes.
    height : float, optional
        Height of ticks as proportion of the axis.
    expand_margins : bool, optional
        Whether to expand the axis limits to include the full range of the rug ticks.
    palette : palette name, list, or dict, optional
        Colors to use for the different levels of the ``hue`` variable.
    hue_order : list, optional
        Order for the levels of the hue variable, otherwise determined from the data.
    hue_norm : tuple or Normalize object, optional
        Normalization method for the hue variable.
    legend : bool, optional
        Whether to draw a legend when plotting multiple hues on the same axis.
    ax : matplotlib axis, optional
        Axis to draw onto; otherwise, use the current axis.
    **kwargs
        Other keyword arguments passed to :meth:`matplotlib.axes.Axes.scatter`.

    Notes
    -----
    - This function provides rug plots, which are a type of univariate plot showing
      distribution of data along a single axis.
    - Rug plots are typically used in combination with other plots to show individual
      data points in relation to a distribution.

    Examples
    --------
    .. include:: ../docstrings/rugplot.rst

    """
    
    # 兼容 Seaborn 版本小于 0.11.0 的代码
    a = kwargs.pop("a", None)   # 移除参数 "a" 并赋值给变量 a
    axis = kwargs.pop("axis", None)   # 移除参数 "axis" 并赋值给变量 axis
    
    if a is not None:
        data = a   # 将参数 "a" 的值赋给 data
        # 提示用户更新代码，因为参数 "a" 将在未来版本中不再支持
        msg = textwrap.dedent("""\n
        The `a` parameter has been replaced; use `x`, `y`, and/or `data` instead.
        Please update your code; This will become an error in seaborn v0.14.0.
        """)
        warnings.warn(msg, UserWarning, stacklevel=2)   # 发出警告信息，提醒用户更新代码
    # 如果 axis 参数不为 None，则根据 axis 的值进行处理
    if axis is not None:
        # 如果 axis 为 "x"，则将 data 赋值给 x
        if axis == "x":
            x = data
        # 如果 axis 为 "y"，则将 data 赋值给 y
        elif axis == "y":
            y = data
        # 将 data 置为 None，表示数据已经分配给了 x 或 y
        data = None
        # 创建警告信息，提醒用户 axis 参数已被弃用，建议使用新的参数
        msg = textwrap.dedent(f"""\n
        The `axis` parameter has been deprecated; use the `{axis}` parameter instead.
        Please update your code; this will become an error in seaborn v0.14.0.
        """)
        # 引发 UserWarning，并指明警告位置为调用函数的上一层
        warnings.warn(msg, UserWarning, stacklevel=2)

    # 检查并处理 vertical 参数
    vertical = kwargs.pop("vertical", None)
    if vertical is not None:
        # 如果 vertical 为 True，则根据情况调整 x 和 y 的赋值关系
        if vertical:
            action_taken = "assigning data to `y`."
            # 如果 x 为空，则将 y 赋值给 data，将 data 赋值给 y
            if x is None:
                data, y = y, data
            else:
                # 否则，交换 x 和 y 的赋值关系
                x, y = y, x
        else:
            # 如果 vertical 为 False，则根据情况调整 x 和 y 的赋值关系
            action_taken = "assigning data to `x`."
        # 创建警告信息，提醒用户 vertical 参数已被弃用，建议更新代码
        msg = textwrap.dedent(f"""\n
        The `vertical` parameter is deprecated; {action_taken}
        This will become an error in seaborn v0.14.0; please update your code.
        """)
        # 引发 UserWarning，并指明警告位置为调用函数的上一层
        warnings.warn(msg, UserWarning, stacklevel=2)

    # 创建 _DistributionPlotter 对象 p，传入数据和变量信息
    p = _DistributionPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue),
    )
    # 对 p 进行色调映射
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    # 如果未提供 ax 参数，则使用当前图形的坐标轴
    if ax is None:
        ax = plt.gca()

    # 将 p 附加到指定的坐标轴 ax 上
    p._attach(ax)

    # 获取并设置图形的颜色参数
    color = kwargs.pop("color", kwargs.pop("c", None))
    kwargs["color"] = _default_color(ax.plot, hue, color, kwargs)

    # 如果 p 没有有效的 x、y 数据，则直接返回 ax
    if not p.has_xy_data:
        return ax

    # 绘制 rug plot
    p.plot_rug(height, expand_margins, legend, **kwargs)

    # 返回最终的坐标轴对象 ax
    return ax
# 设置 rugplot 的文档字符串，用于显示边缘分布并沿着 x 和 y 轴绘制刻度线
rugplot.__doc__ = """\
Plot marginal distributions by drawing ticks along the x and y axes.

This function is intended to complement other plots by showing the location
of individual observations in an unobtrusive way.

Parameters
----------
{params.core.data}  # 数据集
{params.core.xy}    # x 和 y 变量
{params.core.hue}   # 色调变量
height : float      # 每个 rug 元素覆盖的轴范围比例。可以是负数。
expand_margins : bool   # 如果为 True，则增加轴的边距以避免与其他元素重叠。
{params.core.palette}   # 调色板
{params.core.hue_order}    # 色调顺序
{params.core.hue_norm}     # 色调归一化
legend : bool       # 如果为 False，则不为语义变量添加图例。
{params.core.ax}    # 绘图坐标轴
kwargs              # 其他关键字参数传递给 matplotlib.collections.LineCollection 的方法。

Returns
-------
{returns.ax}    # 返回坐标轴对象

Examples
--------

.. include:: ../docstrings/rugplot.rst

""".format(
    params=_param_docs,   # 参数文档
    returns=_core_docs["returns"],   # 返回值文档
)


def displot(
    data=None, *,
    # 向量变量
    x=None, y=None, hue=None, row=None, col=None, weights=None,
    # 其他绘图参数
    kind="hist", rug=False, rug_kws=None, log_scale=None, legend=True,
    # 色调映射参数
    palette=None, hue_order=None, hue_norm=None, color=None,
    # 分面参数
    col_wrap=None, row_order=None, col_order=None,
    height=5, aspect=1, facet_kws=None,
    **kwargs,
):

    p = _DistributionPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue, weights=weights, row=row, col=col),
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)  # 映射色调

    _check_argument("kind", ["hist", "kde", "ecdf"], kind)   # 检查绘图类型参数

    # --- 初始化 FacetGrid 对象

    # 检查是否尝试在特定轴上绘图并发出警告
    if "ax" in kwargs:
        msg = (
            "`displot` is a figure-level function and does not accept "
            "the ax= parameter. You may wish to try {}plot.".format(kind)
        )
        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    for var in ["row", "col"]:
        # 处理缺乏名称信息的分面变量
        if var in p.variables and p.variables[var] is None:
            p.variables[var] = f"_{var}_"

    # 适应用于 FacetGrid 的 plot_data 数据框架
    grid_data = p.plot_data.rename(columns=p.variables)
    grid_data = grid_data.loc[:, ~grid_data.columns.duplicated()]

    col_name = p.variables.get("col")
    row_name = p.variables.get("row")

    if facet_kws is None:
        facet_kws = {}

    g = FacetGrid(
        data=grid_data, row=row_name, col=col_name,
        col_wrap=col_wrap, row_order=row_order,
        col_order=col_order, height=height,
        aspect=aspect,
        **facet_kws,
    )

    # 现在将轴对象附加到绘图器对象
    if kind == "kde":
        allowed_types = ["numeric", "datetime"]
    else:
        allowed_types = None
    p._attach(g, allowed_types=allowed_types, log_scale=log_scale)   # 将轴对象附加到绘图器对象
    # 如果数据对象 p 没有 x/y 数据，则直接返回 g
    if not p.has_xy_data:
        return g

    # 如果 color 和 hue 都未指定，则将 color 设置为默认值 "C0"
    if color is None and hue is None:
        color = "C0"
    # XXX else warn if hue is not None?  # 如果 hue 不为 None，可能需要发出警告（未实现）

    # 将图例设置到 kwargs 中
    kwargs["legend"] = legend

    # --- 绘制图表

    # 如果绘图类型是 "hist"（直方图）
    if kind == "hist":

        # 复制 kwargs 到 hist_kws
        hist_kws = kwargs.copy()

        # 提取将直接传递给 Histogram 的参数
        estimate_defaults = {}
        _assign_default_kwargs(estimate_defaults, Histogram.__init__, histplot)

        estimate_kws = {}
        # 将默认值分配给对应的参数
        for key, default_val in estimate_defaults.items():
            estimate_kws[key] = hist_kws.pop(key, default_val)

        # 处理推断默认值
        if estimate_kws["discrete"] is None:
            estimate_kws["discrete"] = p._default_discrete()

        hist_kws["estimate_kws"] = estimate_kws

        # 设置颜色为 color（如果未指定的话）
        hist_kws.setdefault("color", color)

        # 如果是单变量情况
        if p.univariate:

            # 分配默认参数到 hist_kws，调用单变量直方图绘制函数
            _assign_default_kwargs(hist_kws, p.plot_univariate_histogram, histplot)
            p.plot_univariate_histogram(**hist_kws)

        else:

            # 分配默认参数到 hist_kws，调用双变量直方图绘制函数
            _assign_default_kwargs(hist_kws, p.plot_bivariate_histogram, histplot)
            p.plot_bivariate_histogram(**hist_kws)

    # 如果绘图类型是 "kde"（核密度估计）
    elif kind == "kde":

        # 复制 kwargs 到 kde_kws
        kde_kws = kwargs.copy()

        # 提取将直接传递给 KDE 的参数
        estimate_defaults = {}
        _assign_default_kwargs(estimate_defaults, KDE.__init__, kdeplot)

        estimate_kws = {}
        # 将默认值分配给对应的参数
        for key, default_val in estimate_defaults.items():
            estimate_kws[key] = kde_kws.pop(key, default_val)

        kde_kws["estimate_kws"] = estimate_kws
        kde_kws["color"] = color

        # 如果是单变量情况
        if p.univariate:

            # 分配默认参数到 kde_kws，调用单变量核密度估计绘制函数
            _assign_default_kwargs(kde_kws, p.plot_univariate_density, kdeplot)
            p.plot_univariate_density(**kde_kws)

        else:

            # 分配默认参数到 kde_kws，调用双变量核密度估计绘制函数
            _assign_default_kwargs(kde_kws, p.plot_bivariate_density, kdeplot)
            p.plot_bivariate_density(**kde_kws)

    # 如果绘图类型是 "ecdf"（经验累积分布函数）
    elif kind == "ecdf":

        # 复制 kwargs 到 ecdf_kws
        ecdf_kws = kwargs.copy()

        # 提取将直接传递给 ECDF 估计器的参数
        estimate_kws = {}
        estimate_defaults = {}
        _assign_default_kwargs(estimate_defaults, ECDF.__init__, ecdfplot)
        for key, default_val in estimate_defaults.items():
            estimate_kws[key] = ecdf_kws.pop(key, default_val)

        ecdf_kws["estimate_kws"] = estimate_kws
        ecdf_kws["color"] = color

        # 如果是单变量情况
        if p.univariate:

            # 分配默认参数到 ecdf_kws，调用单变量 ECDF 绘制函数
            _assign_default_kwargs(ecdf_kws, p.plot_univariate_ecdf, ecdfplot)
            p.plot_univariate_ecdf(**ecdf_kws)

        else:

            # 双变量 ECDF 绘图尚未实现
            raise NotImplementedError("Bivariate ECDF plots are not implemented")

    # 所有绘图类型都可以包含一个 rug
    # 如果 rug 参数为真，则执行以下操作
    if rug:
        # 如果 rug_kws 为 None，则设为空字典
        if rug_kws is None:
            rug_kws = {}
        # 将 rugplot 方法的默认关键字参数赋给 rug_kws
        _assign_default_kwargs(rug_kws, p.plot_rug, rugplot)
        # 设置 rug 图例为 False
        rug_kws["legend"] = False
        # 如果 color 不为空，则设置 rug 颜色
        if color is not None:
            rug_kws["color"] = color
        # 使用 rug_kws 中的参数调用 plot_rug 方法
        p.plot_rug(**rug_kws)

    # 调用 FacetGrid 的设置坐标轴标签方法
    # 注意图例目前在绘图方法内设置
    g.set_axis_labels(
        x_var=p.variables.get("x", g.axes.flat[0].get_xlabel()),
        y_var=p.variables.get("y", g.axes.flat[0].get_ylabel()),
    )
    # 设置 FacetGrid 的标题
    g.set_titles()
    # 调整布局以确保图形紧凑
    g.tight_layout()

    # 如果 data 不为空且 x 或 y 不为空，则执行以下操作
    if data is not None and (x is not None or y is not None):
        # 如果 data 不是 DataFrame，则转换为 DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        # 将 data 与 g.data 合并，保留 g.data 中不在 data 中的列
        g.data = pd.merge(
            data,
            g.data[g.data.columns.difference(data.columns)],
            left_index=True,
            right_index=True,
        )
    else:
        # 将 p.variables 中为 None 的项添加下划线前后缀，作为列名
        wide_cols = {
            k: f"_{k}_" if v is None else v for k, v in p.variables.items()
        }
        # 使用 wide_cols 重命名 plot_data 的列，并赋给 g.data
        g.data = p.plot_data.rename(columns=wide_cols)

    # 返回 FacetGrid 对象 g
    return g
# 将 displot 的文档字符串赋值给 __doc__ 属性，用于描述该函数的功能和用法
displot.__doc__ = """\
Figure-level interface for drawing distribution plots onto a FacetGrid.

This function provides access to several approaches for visualizing the
univariate or bivariate distribution of data, including subsets of data
defined by semantic mapping and faceting across multiple subplots. The
``kind`` parameter selects the approach to use:

- :func:`histplot` (with ``kind="hist"``; the default)
- :func:`kdeplot` (with ``kind="kde"``)
- :func:`ecdfplot` (with ``kind="ecdf"``; univariate-only)

Additionally, a :func:`rugplot` can be added to any kind of plot to show
individual observations.

Extra keyword arguments are passed to the underlying function, so you should
refer to the documentation for each to understand the complete set of options
for making plots with this interface.

See the :doc:`distribution plots tutorial <../tutorial/distributions>` for a more
in-depth discussion of the relative strengths and weaknesses of each approach.
The distinction between figure-level and axes-level functions is explained
further in the :doc:`user guide <../tutorial/function_overview>`.

Parameters
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
{params.facets.rowcol}
weights : vector or key in ``data``
    Observation weights used for computing the distribution function.
kind : {{"hist", "kde", "ecdf"}}
    Approach for visualizing the data. Selects the underlying plotting function
    and determines the additional set of valid parameters.
rug : bool
    If True, show each observation with marginal ticks (as in :func:`rugplot`).
rug_kws : dict
    Parameters to control the appearance of the rug plot.
{params.dist.log_scale}
{params.dist.legend}
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}
{params.core.color}
{params.facets.col_wrap}
{params.facets.rowcol_order}
{params.facets.height}
{params.facets.aspect}
{params.facets.facet_kws}
kwargs
    Other keyword arguments are documented with the relevant axes-level function:

    - :func:`histplot` (with ``kind="hist"``)
    - :func:`kdeplot` (with ``kind="kde"``)
    - :func:`ecdfplot` (with ``kind="ecdf"``)

Returns
-------
{returns.facetgrid}

See Also
--------
{seealso.histplot}
{seealso.kdeplot}
{seealso.rugplot}
{seealso.ecdfplot}
{seealso.jointplot}

Examples
--------

See the API documentation for the axes-level functions for more details
about the breadth of options available for each plot kind.

.. include:: ../docstrings/displot.rst

""".format(
    # 使用 _param_docs 字典中的参数文档字符串来替换 {params} 占位符
    params=_param_docs,
    # 使用 _core_docs 字典中的返回值文档字符串来替换 {returns} 占位符
    returns=_core_docs["returns"],
    # 使用 _core_docs 字典中的相关函数文档字符串来替换 {seealso} 占位符
    seealso=_core_docs["seealso"],
)


# =========================================================================== #
# DEPRECATED FUNCTIONS LIVE BELOW HERE
# =========================================================================== #


def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From https://stats.stackexchange.com/questions/798/
    # 将数组 a 转换为 NumPy 数组
    a = np.asarray(a)
    # 如果数组 a 的长度小于 2，则返回 1
    if len(a) < 2:
        return 1
    
    # 使用 numpy 库计算数组 a 的四分位距（IQR）
    iqr = np.subtract.reduce(np.nanpercentile(a, [75, 25]))
    
    # 计算 Freedman-Diaconis 规则中的箱宽（bin width）
    h = 2 * iqr / (len(a) ** (1 / 3))
    
    # 如果箱宽 h 等于 0，则回退到使用 sqrt(a) 来确定箱子数量
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        # 否则，根据最大值和最小值之差来确定箱子数量，并向上取整
        return int(np.ceil((a.max() - a.min()) / h))
def distplot(a=None, bins=None, hist=True, kde=True, rug=False, fit=None,
             hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None,
             color=None, vertical=False, norm_hist=False, axlabel=None,
             label=None, ax=None, x=None):
    """
    DEPRECATED

    This function has been deprecated and will be removed in seaborn v0.14.0.
    It has been replaced by :func:`histplot` and :func:`displot`, two functions
    with a modern API and many more capabilities.

    For a guide to updating, please see this notebook:

    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751

    """

    # 如果使用核密度估计（kde）而不是直方图（hist），则建议使用`kdeplot`（核密度图的轴级函数）
    # 否则建议使用`histplot`（直方图的轴级函数）
    if kde and not hist:
        axes_level_suggestion = (
            "`kdeplot` (an axes-level function for kernel density plots)"
        )
    else:
        axes_level_suggestion = (
            "`histplot` (an axes-level function for histograms)"
        )

    # 构建警告信息的多行字符串
    msg = textwrap.dedent(f"""

    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.

    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or {axes_level_suggestion}.

    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    """)
    # 发出警告信息
    warnings.warn(msg, UserWarning, stacklevel=2)

    # 如果没有提供轴（ax），则默认为当前图形的主轴
    if ax is None:
        ax = plt.gca()

    # 智能地标记支持轴（axlabel）
    label_ax = bool(axlabel)
    if axlabel is None and hasattr(a, "name"):
        axlabel = a.name
        if axlabel is not None:
            label_ax = True

    # 支持新式API，如果提供了x参数，则将其赋值给a
    if x is not None:
        a = x

    # 将a转换为1维浮点数数组
    a = np.asarray(a, float)
    if a.ndim > 1:
        a = a.squeeze()

    # 从数组中删除空值
    a = remove_na(a)

    # 决定是否对直方图进行归一化
    norm_hist = norm_hist or kde or (fit is not None)

    # 处理默认的参数字典
    hist_kws = {} if hist_kws is None else hist_kws.copy()
    kde_kws = {} if kde_kws is None else kde_kws.copy()
    rug_kws = {} if rug_kws is None else rug_kws.copy()
    fit_kws = {} if fit_kws is None else fit_kws.copy()

    # 从当前颜色循环中获取颜色
    if color is None:
        if vertical:
            line, = ax.plot(0, a.mean())
        else:
            line, = ax.plot(a.mean(), 0)
        color = line.get_color()
        line.remove()

    # 将标签插入合适的关键字参数字典中
    if label is not None:
        if hist:
            hist_kws["label"] = label
        elif kde:
            kde_kws["label"] = label
        elif rug:
            rug_kws["label"] = label
        elif fit:
            fit_kws["label"] = label
    # 如果指定绘制直方图
    if hist:
        # 如果未指定 bins 参数，则根据 Freedman-Diaconis 准则自动选择 bins 的数量，最多为 50
        if bins is None:
            bins = min(_freedman_diaconis_bins(a), 50)
        # 设置直方图的默认透明度为 0.4
        hist_kws.setdefault("alpha", 0.4)
        # 设置直方图的默认是否进行归一化
        hist_kws.setdefault("density", norm_hist)

        # 根据 vertical 参数确定直方图的方向
        orientation = "horizontal" if vertical else "vertical"
        # 从 hist_kws 字典中弹出 "color" 键对应的值，如果不存在，则使用 color 参数的值
        hist_color = hist_kws.pop("color", color)
        # 绘制直方图到当前的轴对象 ax 上，使用指定的 bins、orientation、颜色等参数
        ax.hist(a, bins, orientation=orientation,
                color=hist_color, **hist_kws)
        # 如果 hist_color 和 color 不同，则重新设置 hist_kws 中的 "color" 键为 hist_color

    # 根据 vertical 参数确定轴的方向
    axis = "y" if vertical else "x"

    # 如果指定绘制核密度估计图
    if kde:
        # 从 kde_kws 字典中弹出 "color" 键对应的值，如果不存在，则使用 color 参数的值
        kde_color = kde_kws.pop("color", color)
        # 调用 kdeplot 函数绘制核密度估计图到当前的轴对象 ax 上，使用指定的数据轴和颜色等参数
        kdeplot(**{axis: a}, ax=ax, color=kde_color, **kde_kws)
        # 如果 kde_color 和 color 不同，则重新设置 kde_kws 中的 "color" 键为 kde_color

    # 如果指定绘制地毯图
    if rug:
        # 从 rug_kws 字典中弹出 "color" 键对应的值，如果不存在，则使用 color 参数的值
        rug_color = rug_kws.pop("color", color)
        # 调用 rugplot 函数绘制地毯图到当前的轴对象 ax 上，使用指定的数据轴和颜色等参数
        rugplot(**{axis: a}, ax=ax, color=rug_color, **rug_kws)
        # 如果 rug_color 和 color 不同，则重新设置 rug_kws 中的 "color" 键为 rug_color

    # 如果指定绘制拟合曲线
    if fit is not None:
        # 定义拟合分布的概率密度函数 pdf(x)
        def pdf(x):
            return fit.pdf(x, *params)

        # 从 fit_kws 字典中弹出 "color" 键对应的值，如果不存在，则使用默认颜色 "#282828"
        fit_color = fit_kws.pop("color", "#282828")
        # 设置核密度估计的网格大小，默认为 200
        gridsize = fit_kws.pop("gridsize", 200)
        # 设置拟合曲线的切除值，默认为 3
        cut = fit_kws.pop("cut", 3)
        # 设置拟合曲线的剪裁范围，默认为 (-inf, inf)
        clip = fit_kws.pop("clip", (-np.inf, np.inf))
        # 使用 Scott's method 计算核密度估计的带宽
        bw = gaussian_kde(a).scotts_factor() * a.std(ddof=1)
        # 计算拟合曲线的支持点 x
        x = _kde_support(a, bw, gridsize, cut, clip)
        # 对数据 a 进行拟合，返回拟合参数 params
        params = fit.fit(a)
        # 计算拟合曲线在 x 上的概率密度值 y
        y = pdf(x)
        # 如果 vertical 为 True，则交换 x 和 y 的值，以适应水平绘制
        if vertical:
            x, y = y, x
        # 绘制拟合曲线到当前的轴对象 ax 上，使用指定的 x、y 数据和颜色等参数
        ax.plot(x, y, color=fit_color, **fit_kws)
        # 如果 fit_color 不等于默认颜色 "#282828"，则重新设置 fit_kws 中的 "color" 键为 fit_color

    # 如果指定对轴对象进行标签设置
    if label_ax:
        # 如果 vertical 为 True，则设置轴对象 ax 的 y 轴标签为 axlabel
        if vertical:
            ax.set_ylabel(axlabel)
        # 否则设置轴对象 ax 的 x 轴标签为 axlabel
        else:
            ax.set_xlabel(axlabel)

    # 返回绘制完成的轴对象 ax
    return ax
```