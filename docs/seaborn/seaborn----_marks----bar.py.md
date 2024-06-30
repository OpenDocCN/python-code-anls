# `D:\src\scipysrc\seaborn\seaborn\_marks\bar.py`

```
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableBool,
    MappableColor,
    MappableFloat,
    MappableStyle,
    resolve_properties,
    resolve_color,
    document_properties
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from matplotlib.artist import Artist
    from seaborn._core.scales import Scale


class BarBase(Mark):

    def _make_patches(self, data, scales, orient):
        """
        根据传入的数据和尺度信息生成条形图所需的矩形条列表和相应值的列表。

        Parameters:
        - data: 包含绘图数据的数据结构
        - scales: 包含尺度信息的字典
        - orient: 条形图的方向，可以是"x"或"y"

        Returns:
        - bars: 生成的矩形条对象列表
        - vals: 矩形条对应的数值列表
        """

        # 获取数据变换的函数
        transform = scales[orient]._matplotlib_scale.get_transform()
        forward = transform.transform
        reverse = transform.inverted().transform

        # 确定另一个轴的标识符
        other = {"x": "y", "y": "x"}[orient]

        # 计算矩形条的位置和宽度
        pos = reverse(forward(data[orient]) - data["width"] / 2)
        width = reverse(forward(data[orient]) + data["width"] / 2) - pos

        # 计算矩形条的高度或宽度以及基线位置
        val = (data[other] - data["baseline"]).to_numpy()
        base = data["baseline"].to_numpy()

        # 解析属性，获取绘图所需的关键字参数
        kws = self._resolve_properties(data, scales)
        if orient == "x":
            kws.update(x=pos, y=base, w=width, h=val)
        else:
            kws.update(x=base, y=pos, w=val, h=width)

        # 移除不必要的关键字参数
        kws.pop("width", None)
        kws.pop("baseline", None)

        # 确定数值维度（高度或宽度），生成矩形条和对应数值的列表
        val_dim = {"x": "h", "y": "w"}[orient]
        bars, vals = [], []

        for i in range(len(data)):
            # 为当前行数据创建字典
            row = {k: v[i] for k, v in kws.items()}

            # 跳过没有数值的矩形条
            if not np.nan_to_num(row[val_dim]):
                continue

            # 创建矩形条对象并添加到列表中
            bar = mpl.patches.Rectangle(
                xy=(row["x"], row["y"]),
                width=row["w"],
                height=row["h"],
                facecolor=row["facecolor"],
                edgecolor=row["edgecolor"],
                linestyle=row["edgestyle"],
                linewidth=row["edgewidth"],
                **self.artist_kws,
            )
            bars.append(bar)
            vals.append(row[val_dim])

        return bars, vals

    def _resolve_properties(self, data, scales):
        """
        根据数据和尺度信息解析绘图属性。

        Parameters:
        - data: 包含绘图数据的数据结构
        - scales: 包含尺度信息的字典

        Returns:
        - resolved: 解析后的绘图属性字典
        """

        # 调用外部函数解析绘图属性
        resolved = resolve_properties(self, data, scales)

        # 解析颜色属性
        resolved["facecolor"] = resolve_color(self, data, "", scales)
        resolved["edgecolor"] = resolve_color(self, data, "edge", scales)

        # 处理填充透明度
        fc = resolved["facecolor"]
        if isinstance(fc, tuple):
            resolved["facecolor"] = fc[0], fc[1], fc[2], fc[3] * resolved["fill"]
        else:
            fc[:, 3] = fc[:, 3] * resolved["fill"]  # TODO Is inplace mod a problem?
            resolved["facecolor"] = fc

        return resolved

    def _legend_artist(
        self, variables: list[str], value: Any, scales: dict[str, Scale],
    ) -> Artist:
        # 定义函数签名，指定返回类型为Artist对象
        # TODO return some sensible default? （待办：返回一些合理的默认值？）
        
        # 创建字典key，通过变量列表variables的值为键，value为值
        key = {v: value for v in variables}
        
        # 调用对象的_resolve_properties方法，解析key中的属性值，使用给定的比例scales
        key = self._resolve_properties(key, scales)
        
        # 使用matplotlib的patches模块创建一个Patch对象artist，根据key中的属性设置其样式
        artist = mpl.patches.Patch(
            facecolor=key["facecolor"],
            edgecolor=key["edgecolor"],
            linewidth=key["edgewidth"],
            linestyle=key["edgestyle"],
        )
        
        # 返回创建的Patch对象artist
        return artist
# 使用 @document_properties 装饰器设置文档属性
# 使用 @dataclass 装饰器声明 Bar 类为数据类
class Bar(BarBase):
    """
    A bar mark drawn between baseline and data values.

    See also
    --------
    Bars : A faster bar mark with defaults more suitable for histograms.

    Examples
    --------
    .. include:: ../docstrings/objects.Bar.rst

    """

    # 设置颜色属性，默认为青色 ('C0')，不进行分组
    color: MappableColor = Mappable("C0", grouping=False)
    # 设置透明度属性，默认为 0.7，不进行分组
    alpha: MappableFloat = Mappable(.7, grouping=False)
    # 设置填充属性，默认为 True，不进行分组
    fill: MappableBool = Mappable(True, grouping=False)
    # 设置边缘颜色属性，依赖于 color 属性，默认不进行分组
    edgecolor: MappableColor = Mappable(depend="color", grouping=False)
    # 设置边缘透明度属性，默认为 1，不进行分组
    edgealpha: MappableFloat = Mappable(1, grouping=False)
    # 设置边缘宽度属性，依赖于 rc 参数 'patch.linewidth'，默认不进行分组
    edgewidth: MappableFloat = Mappable(rc="patch.linewidth", grouping=False)
    # 设置边缘样式属性，默认为实线 '-'，不进行分组
    edgestyle: MappableStyle = Mappable("-", grouping=False)
    # 设置宽度属性，默认为 0.8，不进行分组
    width: MappableFloat = Mappable(.8, grouping=False)
    # 设置基线位置属性，默认为 0，不进行分组
    baseline: MappableFloat = Mappable(0, grouping=False)  # TODO *is* this mappable?

    # 定义内部方法 _plot，用于绘制图形
    def _plot(self, split_gen, scales, orient):

        # 根据方向确定值的索引位置
        val_idx = ["y", "x"].index(orient)

        # 遍历分割生成器的结果
        for _, data, ax in split_gen():

            # 调用 _make_patches 方法生成 bars 和 vals
            bars, vals = self._make_patches(data, scales, orient)

            # 遍历每个 bar 对象
            for bar in bars:

                # 由于我们裁剪了图形（见下文），边缘看起来实际宽度的一半
                # 这是一个笨拙的解决方法，如果直接操作 artist 可能会有意外
                bar.set_linewidth(bar.get_linewidth() * 2)
                linestyle = bar.get_linestyle()
                if linestyle[1]:
                    linestyle = (linestyle[0], tuple(x / 2 for x in linestyle[1]))
                bar.set_linestyle(linestyle)

                # 处理边缘线在条形图堆叠或并列时的重叠问题的一种方法
                # 这可能导致问题，需要在后续重新审视
                # 用 bbox 裁剪应该比路径更快，但是无法获取与坐标轴 bbox 的交集
                bar.set_clip_path(bar.get_path(), bar.get_transform() + ax.transData)
                if self.artist_kws.get("clip_on", True):
                    # 似乎上述 hack 撤销了默认的坐标轴裁剪
                    bar.set_clip_box(ax.bbox)
                bar.sticky_edges[val_idx][:] = (0, np.inf)
                ax.add_patch(bar)

            # 添加一个容器，对于例如 Axes.bar_label 非常有用
            orientation = {"x": "vertical", "y": "horizontal"}[orient]
            container_kws = dict(datavalues=vals, orientation=orientation)
            container = mpl.container.BarContainer(bars, **container_kws)
            ax.add_container(container)


@document_properties
@dataclass
    """
    A faster bar mark with defaults more suitable for histograms.

    See also
    --------
    Bar : A bar mark drawn between baseline and data values.

    Examples
    --------
    .. include:: ../docstrings/objects.Bars.rst

    """
    # 设置颜色属性，默认为'C0'，不进行分组
    color: MappableColor = Mappable("C0", grouping=False)
    # 设置透明度属性，默认为0.7，不进行分组
    alpha: MappableFloat = Mappable(.7, grouping=False)
    # 设置填充属性，默认为True，不进行分组
    fill: MappableBool = Mappable(True, grouping=False)
    # 设置边框颜色属性，默认从'patch.edgecolor'中获取，不进行分组
    edgecolor: MappableColor = Mappable(rc="patch.edgecolor", grouping=False)
    # 设置边框透明度属性，默认为1，不进行分组
    edgealpha: MappableFloat = Mappable(1, grouping=False)
    # 设置边框宽度属性，默认自动，不进行分组
    edgewidth: MappableFloat = Mappable(auto=True, grouping=False)
    # 设置边框样式属性，默认为'-'，不进行分组
    edgestyle: MappableStyle = Mappable("-", grouping=False)
    # 设置宽度属性，默认为1，不进行分组
    width: MappableFloat = Mappable(1, grouping=False)
    # 设置基线属性，默认为0，不进行分组
    baseline: MappableFloat = Mappable(0, grouping=False)  # TODO *is* this mappable?

    def _plot(self, split_gen, scales, orient):
        # 根据方向确定索引值
        ori_idx = ["x", "y"].index(orient)
        val_idx = ["y", "x"].index(orient)

        # 创建默认字典存储图形对象列表
        patches = defaultdict(list)
        # 遍历分割生成器并处理数据
        for _, data, ax in split_gen():
            # 创建条形图形并添加到对应坐标轴的列表中
            bars, _ = self._make_patches(data, scales, orient)
            patches[ax].extend(bars)

        # 创建图形集合字典
        collections = {}
        # 遍历图形对象字典
        for ax, ax_patches in patches.items():
            # 创建图形集合并匹配原始对象
            col = mpl.collections.PatchCollection(ax_patches, match_original=True)
            # 设置粘性边缘值
            col.sticky_edges[val_idx][:] = (0, np.inf)
            # 添加集合到坐标轴并禁用自动限制
            ax.add_collection(col, autolim=False)
            # 存储集合对象到字典中
            collections[ax] = col

            # 解决 matplotlib 自动缩放的问题
            xys = np.vstack([path.vertices for path in col.get_paths()])
            ax.update_datalim(xys)

        # 如果 'edgewidth' 不在比例尺中且是 Mappable 类型
        if "edgewidth" not in scales and isinstance(self.edgewidth, Mappable):
            # 对每个集合进行自动缩放视图
            for ax in collections:
                ax.autoscale_view()

            # 获取尺寸信息的函数
            def get_dimensions(collection):
                edges, widths = [], []
                # 遍历集合中路径的顶点
                for verts in (path.vertices for path in collection.get_paths()):
                    edges.append(min(verts[:, ori_idx]))
                    widths.append(np.ptp(verts[:, ori_idx]))
                return np.array(edges), np.array(widths)

            # 初始化最小宽度值
            min_width = np.inf
            # 遍历集合和坐标轴
            for ax, col in collections.items():
                edges, widths = get_dimensions(col)
                points = 72 / ax.figure.dpi * abs(
                    ax.transData.transform([edges + widths] * 2)
                    - ax.transData.transform([edges] * 2)
                )
                # 更新最小宽度值
                min_width = min(min_width, min(points[:, ori_idx]))

            # 计算线宽
            linewidth = min(.1 * min_width, mpl.rcParams["patch.linewidth"])
            # 设置集合对象的线宽属性
            for _, col in collections.items():
                col.set_linewidth(linewidth)
```