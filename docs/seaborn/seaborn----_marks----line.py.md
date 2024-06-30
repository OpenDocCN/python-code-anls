# `D:\src\scipysrc\seaborn\seaborn\_marks\line.py`

```
# 引入未来的注解特性，以支持类型注解中的自引用类型
from __future__ import annotations
# 引入用于数据类的装饰器
from dataclasses import dataclass
# 引入类变量类型的支持
from typing import ClassVar

# 引入 numpy 库，并使用别名 np
import numpy as np
# 引入 matplotlib 库，并使用别名 mpl
import matplotlib as mpl

# 从 seaborn 库中引入相关的模块和类
from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableFloat,
    MappableString,
    MappableColor,
    resolve_properties,
    resolve_color,
    document_properties,
)

# 应用文档属性装饰器，用于类的文档化
@document_properties
# 使用数据类装饰器，用于创建带有自动属性生成的数据类
@dataclass
# 定义 Path 类，继承自 Mark 类
class Path(Mark):
    """
    A mark connecting data points in the order they appear.

    See also
    --------
    Line : A mark connecting data points with sorting along the orientation axis.
    Paths : A faster but less-flexible mark for drawing many paths.

    Examples
    --------
    .. include:: ../docstrings/objects.Path.rst

    """
    # 定义颜色属性，并指定默认值为可映射颜色 'C0'
    color: MappableColor = Mappable("C0")
    # 定义透明度属性，并指定默认值为可映射的浮点数 1
    alpha: MappableFloat = Mappable(1)
    # 定义线宽属性，并指定默认值为可映射的线宽设置
    linewidth: MappableFloat = Mappable(rc="lines.linewidth")
    # 定义线型属性，并指定默认值为可映射的线型设置
    linestyle: MappableString = Mappable(rc="lines.linestyle")
    # 定义标记类型属性，并指定默认值为可映射的标记设置
    marker: MappableString = Mappable(rc="lines.marker")
    # 定义标记大小属性，并指定默认值为可映射的标记大小设置
    pointsize: MappableFloat = Mappable(rc="lines.markersize")
    # 定义填充颜色属性，并指定依赖于 color 属性
    fillcolor: MappableColor = Mappable(depend="color")
    # 定义边缘颜色属性，并指定依赖于 color 属性
    edgecolor: MappableColor = Mappable(depend="color")
    # 定义标记边缘宽度属性，并指定默认值为可映射的标记边缘宽度设置
    edgewidth: MappableFloat = Mappable(rc="lines.markeredgewidth")

    # 定义类变量 _sort，用于标识是否对数据进行排序
    _sort: ClassVar[bool] = False

    # 定义绘图方法 _plot，接受分割生成器、比例尺和方向参数
    def _plot(self, split_gen, scales, orient):
        # 使用分割生成器迭代处理数据分块及其对应的轴
        for keys, data, ax in split_gen(keep_na=not self._sort):
            # 解析属性值，如颜色、填充颜色、边缘颜色等，并应用比例尺
            vals = resolve_properties(self, keys, scales)
            # 解析并设置绘图所需的颜色
            vals["color"] = resolve_color(self, keys, scales=scales)
            # 解析并设置填充颜色
            vals["fillcolor"] = resolve_color(self, keys, prefix="fill", scales=scales)
            # 解析并设置边缘颜色
            vals["edgecolor"] = resolve_color(self, keys, prefix="edge", scales=scales)

            # 如果标记需要排序，则按指定方向对数据进行排序
            if self._sort:
                data = data.sort_values(orient, kind="mergesort")

            # 复制艺术家关键字参数，并处理端点风格
            artist_kws = self.artist_kws.copy()
            self._handle_capstyle(artist_kws, vals)

            # 创建 Line2D 对象，并添加到指定轴上
            line = mpl.lines.Line2D(
                data["x"].to_numpy(),
                data["y"].to_numpy(),
                color=vals["color"],
                linewidth=vals["linewidth"],
                linestyle=vals["linestyle"],
                marker=vals["marker"],
                markersize=vals["pointsize"],
                markerfacecolor=vals["fillcolor"],
                markeredgecolor=vals["edgecolor"],
                markeredgewidth=vals["edgewidth"],
                **artist_kws,
            )
            ax.add_line(line)
    # 定义一个方法 `_legend_artist`，用于生成一个图例艺术家对象
    def _legend_artist(self, variables, value, scales):

        # 为变量列表中的每个变量创建一个键值对，值为给定的值
        keys = {v: value for v in variables}
        
        # 使用解析函数 `resolve_properties` 解析属性，并根据比例尺 `scales` 获得属性值
        vals = resolve_properties(self, keys, scales)
        
        # 解析颜色属性 `color`，并根据比例尺 `scales` 获得颜色值
        vals["color"] = resolve_color(self, keys, scales=scales)
        
        # 解析填充颜色属性 `fillcolor`，并根据比例尺 `scales` 获得填充颜色值
        vals["fillcolor"] = resolve_color(self, keys, prefix="fill", scales=scales)
        
        # 解析边缘颜色属性 `edgecolor`，并根据比例尺 `scales` 获得边缘颜色值
        vals["edgecolor"] = resolve_color(self, keys, prefix="edge", scales=scales)

        # 复制艺术家关键字参数到 `artist_kws` 字典
        artist_kws = self.artist_kws.copy()
        
        # 处理 `capstyle`，处理 matplotlib 的问题，设置线段端点样式
        self._handle_capstyle(artist_kws, vals)

        # 创建并返回一个 matplotlib 的 `Line2D` 艺术家对象，参数包括颜色、线宽、线型、标记等
        return mpl.lines.Line2D(
            [], [],
            color=vals["color"],
            linewidth=vals["linewidth"],
            linestyle=vals["linestyle"],
            marker=vals["marker"],
            markersize=vals["pointsize"],
            markerfacecolor=vals["fillcolor"],
            markeredgecolor=vals["edgecolor"],
            markeredgewidth=vals["edgewidth"],
            **artist_kws,
        )

    # 定义一个方法 `_handle_capstyle`，用于处理线段端点样式
    def _handle_capstyle(self, kws, vals):

        # 解决 matplotlib 的问题 https://github.com/matplotlib/matplotlib/issues/23437
        # 如果线型为空，则使用默认的实线端点样式作为替代
        if vals["linestyle"][1] is None:
            capstyle = kws.get("solid_capstyle", mpl.rcParams["lines.solid_capstyle"])
            kws["dash_capstyle"] = capstyle
@document_properties
@dataclass
class Line(Path):
    """
    A mark connecting data points with sorting along the orientation axis.

    See also
    --------
    Path : A mark connecting data points in the order they appear.
    Lines : A faster but less-flexible mark for drawing many lines.

    Examples
    --------
    .. include:: ../docstrings/objects.Line.rst

    """
    _sort: ClassVar[bool] = True  # Class variable indicating whether to sort data points


@document_properties
@dataclass
class Paths(Mark):
    """
    A faster but less-flexible mark for drawing many paths.

    See also
    --------
    Path : A mark connecting data points in the order they appear.

    Examples
    --------
    .. include:: ../docstrings/objects.Paths.rst

    """
    color: MappableColor = Mappable("C0")  # Color of the paths
    alpha: MappableFloat = Mappable(1)  # Transparency level
    linewidth: MappableFloat = Mappable(rc="lines.linewidth")  # Width of the lines
    linestyle: MappableString = Mappable(rc="lines.linestyle")  # Style of the lines

    _sort: ClassVar[bool] = False  # Class variable indicating whether to sort paths

    def __post_init__(self):
        """
        Post-initialization method to set artist properties.

        Sets the capstyle property for LineCollection artists based on rc parameters.
        """
        # LineCollection artists have a capstyle property but don't source its value
        # from the rc, so we do that manually here. Unfortunately, because we add
        # only one LineCollection, we have the use the same capstyle for all lines
        # even when they are dashed. It's a slight inconsistency, but looks fine IMO.
        self.artist_kws.setdefault("capstyle", mpl.rcParams["lines.solid_capstyle"])

    def _plot(self, split_gen, scales, orient):
        """
        Plotting method for Paths.

        Parameters
        ----------
        split_gen : function
            Generator function for splitting data.
        scales : dict
            Scales for resolving properties.
        orient : str
            Orientation axis ('h' for horizontal, 'v' for vertical).

        """
        line_data = {}
        for keys, data, ax in split_gen(keep_na=not self._sort):
            """
            Iterate over split data and populate line segments, colors, linewidths,
            and linestyles for each axis.
            """

            if ax not in line_data:
                line_data[ax] = {
                    "segments": [],
                    "colors": [],
                    "linewidths": [],
                    "linestyles": [],
                }

            segments = self._setup_segments(data, orient)  # Setup line segments
            line_data[ax]["segments"].extend(segments)  # Extend segments for the axis
            n = len(segments)

            vals = resolve_properties(self, keys, scales)  # Resolve properties
            vals["color"] = resolve_color(self, keys, scales=scales)  # Resolve color

            line_data[ax]["colors"].extend([vals["color"]] * n)  # Extend colors for the axis
            line_data[ax]["linewidths"].extend([vals["linewidth"]] * n)  # Extend linewidths
            line_data[ax]["linestyles"].extend([vals["linestyle"]] * n)  # Extend linestyles

        for ax, ax_data in line_data.items():
            lines = mpl.collections.LineCollection(**ax_data, **self.artist_kws)
            # Handle datalim update manually
            # https://github.com/matplotlib/matplotlib/issues/23129
            ax.add_collection(lines, autolim=False)  # Add LineCollection to the axis
            if ax_data["segments"]:
                xy = np.concatenate(ax_data["segments"])  # Concatenate all segments
                ax.update_datalim(xy)  # Manually update datalim for the axis
    # 定义私有方法 `_legend_artist`，用于创建图例图形对象
    def _legend_artist(self, variables, value, scales):
        # 根据给定的变量和数值解析属性并生成关键字参数字典
        key = resolve_properties(self, {v: value for v in variables}, scales)

        # 复制艺术家关键字参数，设置线段末端风格
        artist_kws = self.artist_kws.copy()
        capstyle = artist_kws.pop("capstyle")
        artist_kws["solid_capstyle"] = capstyle
        artist_kws["dash_capstyle"] = capstyle

        # 创建并返回线条对象，使用解析后的关键字参数
        return mpl.lines.Line2D(
            [], [],
            color=key["color"],
            linewidth=key["linewidth"],
            linestyle=key["linestyle"],
            **artist_kws,
        )

    # 定义私有方法 `_setup_segments`，用于设置数据段
    def _setup_segments(self, data, orient):
        # 如果需要排序，则按指定的方向对数据进行稳定排序
        if self._sort:
            data = data.sort_values(orient, kind="mergesort")

        # 将数据的 x 和 y 值堆叠成二维数组，以避免块合并
        xy = np.column_stack([data["x"], data["y"]])

        # 返回二维数组作为数据段的列表
        return [xy]
@document_properties
@dataclass
# 创建一个名为 Lines 的类，它继承自 Paths 类，并且带有特定的文档属性
class Lines(Paths):
    """
    A faster but less-flexible mark for drawing many lines.
    
    See also
    --------
    Line : A mark connecting data points with sorting along the orientation axis.
    
    Examples
    --------
    .. include:: ../docstrings/objects.Lines.rst
    
    """
    _sort: ClassVar[bool] = True  # 类变量 _sort 被设置为 True


@document_properties
@dataclass
# 创建一个名为 Range 的类，它继承自 Paths 类，并且带有特定的文档属性
class Range(Paths):
    """
    An oriented line mark drawn between min/max values.
    
    Examples
    --------
    .. include:: ../docstrings/objects.Range.rst
    
    """
    def _setup_segments(self, data, orient):
        # 设置段落的准备工作，根据给定的数据和方向
        
        # TODO better checks on what variables we have
        # TODO what if only one exist?
        
        # 确定与 orient 相反的方向值
        val = {"x": "y", "y": "x"}[orient]
        
        # 如果数据列中不存在 {val}min 或 {val}max 列
        if not set(data.columns) & {f"{val}min", f"{val}max"}:
            # 根据 orient 对数据进行分组，并使用最小值和最大值进行聚合
            agg = {f"{val}min": (val, "min"), f"{val}max": (val, "max")}
            data = data.groupby(orient).agg(**agg).reset_index()

        # 提取 orient、{val}min 和 {val}max 列，并将数据转换为长格式
        cols = [orient, f"{val}min", f"{val}max"]
        data = data[cols].melt(orient, value_name=val)[["x", "y"]]
        
        # 将数据按 orient 分组并转换为 numpy 数组，存储在 segments 中
        segments = [d.to_numpy() for _, d in data.groupby(orient)]
        return segments


@document_properties
@dataclass
# 创建一个名为 Dash 的类，它继承自 Paths 类，并且带有特定的文档属性
class Dash(Paths):
    """
    A line mark drawn as an oriented segment for each datapoint.
    
    Examples
    --------
    .. include:: ../docstrings/objects.Dash.rst
    
    """
    width: MappableFloat = Mappable(.8, grouping=False)

    def _setup_segments(self, data, orient):
        # 设置段落的准备工作，根据给定的数据和方向
        
        # 确定 orient 在 x 或 y 中的索引
        ori = ["x", "y"].index(orient)
        
        # 将数据中的 "x" 和 "y" 列转换为 numpy 数组
        xys = data[["x", "y"]].to_numpy().astype(float)
        
        # 创建由两个 xys 组成的段落数组
        segments = np.stack([xys, xys], axis=1)
        
        # 调整每个段落的起始和结束位置，根据数据的宽度调整
        segments[:, 0, ori] -= data["width"] / 2
        segments[:, 1, ori] += data["width"] / 2
        
        return segments
```