# `D:\src\scipysrc\seaborn\seaborn\_marks\area.py`

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
    MappableFloat,
    MappableColor,
    MappableStyle,
    resolve_properties,
    resolve_color,
    document_properties,
)


class AreaBase:
    """
    Base class for drawing filled areas on plots.
    """

    def _plot(self, split_gen, scales, orient):
        """
        Plot method for drawing filled areas.

        Parameters:
        - split_gen: a generator function that yields keys, data, and axes
        - scales: scaling information for the plot
        - orient: orientation of the plot (e.g., 'x' or 'y')

        This method iterates over generated splits and adds patches to axes.
        """

        patches = defaultdict(list)

        for keys, data, ax in split_gen():
            kws = {}

            # Standardize coordinate parameters based on orientation
            data = self._standardize_coordinate_parameters(data, orient)

            # Resolve properties using keys and scales
            resolved = resolve_properties(self, keys, scales)

            # Get vertices for the polygon
            verts = self._get_verts(data, orient)

            # Update the data limits of the axes
            ax.update_datalim(verts)

            # Resolve fill color
            fc = resolve_color(self, keys, "", scales)
            if not resolved["fill"]:
                fc = mpl.colors.to_rgba(fc, 0)

            # Set keyword arguments for the Polygon
            kws["facecolor"] = fc
            kws["edgecolor"] = resolve_color(self, keys, "edge", scales)
            kws["linewidth"] = resolved["edgewidth"]
            kws["linestyle"] = resolved["edgestyle"]

            # Append Polygon patch to patches dictionary for the axis
            patches[ax].append(mpl.patches.Polygon(verts, **kws))

        # Add patches to respective axes
        for ax, ax_patches in patches.items():
            for patch in ax_patches:
                # Perform post-processing on the artist (patch)
                self._postprocess_artist(patch, ax, orient)
                ax.add_patch(patch)

    def _standardize_coordinate_parameters(self, data, orient):
        """
        Standardizes coordinate parameters based on the orientation.

        Parameters:
        - data: input data for plotting
        - orient: orientation of the plot ('x' or 'y')

        Returns:
        - standardized data
        """
        return data

    def _postprocess_artist(self, artist, ax, orient):
        """
        Post-processing function for the artist (patch).

        Parameters:
        - artist: the artist object to post-process
        - ax: the axis object associated with the artist
        - orient: orientation of the plot ('x' or 'y')
        """
        pass

    def _get_verts(self, data, orient):
        """
        Retrieves vertices for drawing the polygon.

        Parameters:
        - data: input data for plotting
        - orient: orientation of the plot ('x' or 'y')

        Returns:
        - vertices of the polygon
        """
        dv = {"x": "y", "y": "x"}[orient]
        data = data.sort_values(orient, kind="mergesort")
        verts = np.concatenate([
            data[[orient, f"{dv}min"]].to_numpy(),
            data[[orient, f"{dv}max"]].to_numpy()[::-1],
        ])
        if orient == "y":
            verts = verts[:, ::-1]
        return verts

    def _legend_artist(self, variables, value, scales):
        """
        Generates a legend artist (Patch) for the filled area.

        Parameters:
        - variables: variables associated with the legend
        - value: value associated with the legend
        - scales: scaling information for the plot

        Returns:
        - mpl.patches.Patch object for legend
        """
        keys = {v: value for v in variables}
        resolved = resolve_properties(self, keys, scales)

        # Resolve fill color for legend
        fc = resolve_color(self, keys, "", scales)
        if not resolved["fill"]:
            fc = mpl.colors.to_rgba(fc, 0)

        # Create and return Patch object with resolved properties
        return mpl.patches.Patch(
            facecolor=fc,
            edgecolor=resolve_color(self, keys, "edge", scales),
            linewidth=resolved["edgewidth"],
            linestyle=resolved["edgestyle"],
            **self.artist_kws,  # Additional keyword arguments
        )


@document_properties
@dataclass
class Area(AreaBase, Mark):
    """
    A fill mark drawn from a baseline to data values.

    See also
    --------
    Band : A fill mark representing an interval between values.

    Examples
    --------
    Documentation link to examples for Area.
    """

    color: MappableColor = Mappable("C0", )
    alpha: MappableFloat = Mappable(.2, )
    fill: MappableBool = Mappable(True, )
    edgecolor: MappableColor = Mappable(depend="color")
    # 创建一个可映射的浮点数对象 edgealpha，初始值为 1
    edgealpha: MappableFloat = Mappable(1, )

    # 创建一个可映射的浮点数对象 edgewidth，使用 patch.linewidth 作为默认参数
    edgewidth: MappableFloat = Mappable(rc="patch.linewidth", )

    # 创建一个可映射的样式对象 edgestyle，初始样式为 "-"
    edgestyle: MappableStyle = Mappable("-", )

    # TODO 是否应该设置/可映射化？
    # 创建一个可映射的浮点数对象 baseline，初始值为 0，不进行分组
    baseline: MappableFloat = Mappable(0, grouping=False)

    # 定义一个方法 _standardize_coordinate_parameters 用于标准化坐标参数
    def _standardize_coordinate_parameters(self, data, orient):
        # 根据 orient 参数选择 dv
        dv = {"x": "y", "y": "x"}[orient]
        # 重命名 data DataFrame 的列，将 "baseline" 列重命名为 f"{dv}min"，dv 列重命名为 f"{dv}max"
        return data.rename(columns={"baseline": f"{dv}min", dv: f"{dv}max"})

    # 定义一个方法 _postprocess_artist 用于后处理艺术家对象
    def _postprocess_artist(self, artist, ax, orient):

        # TODO 从 Bar 复制了大量代码，应该将其抽象化
        # 参见那里的注释，我不打算过多重复它们

        # 设置艺术家对象的线宽为当前线宽的两倍
        artist.set_linewidth(artist.get_linewidth() * 2)

        # 获取当前艺术家对象的线型
        linestyle = artist.get_linestyle()
        # 如果线型的第二个元素存在，则将其缩小为原来的一半
        if linestyle[1]:
            linestyle = (linestyle[0], tuple(x / 2 for x in linestyle[1]))
        # 设置艺术家对象的新线型
        artist.set_linestyle(linestyle)

        # 设置艺术家对象的剪切路径，使用艺术家对象的路径和数据坐标变换
        artist.set_clip_path(artist.get_path(), artist.get_transform() + ax.transData)
        # 如果配置中设置为 clip_on（默认为 True），则将剪切框设置为 ax.bbox
        if self.artist_kws.get("clip_on", True):
            artist.set_clip_box(ax.bbox)

        # 根据 orient 参数确定值索引的顺序（"y" 为 0，"x" 为 1），将 sticky_edges 相应位置的值设置为 (0, np.inf)
        val_idx = ["y", "x"].index(orient)
        artist.sticky_edges[val_idx][:] = (0, np.inf)
@document_properties
@dataclass
class Band(AreaBase, Mark):
    """
    A fill mark representing an interval between values.

    See also
    --------
    Area : A fill mark drawn from a baseline to data values.

    Examples
    --------
    .. include:: ../docstrings/objects.Band.rst

    """
    # 带区间标记，表示数值之间的一个填充标记。

    # 默认颜色为蓝色 ("C0")
    color: MappableColor = Mappable("C0", )
    # 默认透明度为 0.2
    alpha: MappableFloat = Mappable(.2, )
    # 默认进行填充
    fill: MappableBool = Mappable(True, )
    # 边缘颜色与主颜色一致
    edgecolor: MappableColor = Mappable(depend="color", )
    # 边缘透明度为 1（不透明）
    edgealpha: MappableFloat = Mappable(1, )
    # 边缘宽度为 0
    edgewidth: MappableFloat = Mappable(0, )
    # 边缘线型为实线 "-"
    edgestyle: MappableFloat = Mappable("-", )

    def _standardize_coordinate_parameters(self, data, orient):
        # 选择相应的坐标轴
        other = {"x": "y", "y": "x"}[orient]
        # 如果数据中缺少对应的 {other}min 和 {other}max 列
        if not set(data.columns) & {f"{other}min", f"{other}max"}:
            # 按照 orient 进行分组汇总，生成缺失的列
            agg = {f"{other}min": (other, "min"), f"{other}max": (other, "max")}
            data = data.groupby(orient).agg(**agg).reset_index()
        return data
```