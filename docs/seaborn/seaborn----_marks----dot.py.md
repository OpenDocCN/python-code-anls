# `D:\src\scipysrc\seaborn\seaborn\_marks\dot.py`

```
# 导入 __future__ 模块中的 annotations 特性，支持类型注解
from __future__ import annotations
# 导入 dataclass 模块，支持创建数据类
from dataclasses import dataclass

# 导入 numpy 库，并使用 np 别名引用
import numpy as np
# 导入 matplotlib 库，并使用 mpl 别名引用
import matplotlib as mpl

# 从 seaborn 库中导入标记相关的基础类和函数
from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableBool,
    MappableFloat,
    MappableString,
    MappableColor,
    MappableStyle,
    resolve_properties,
    resolve_color,
    document_properties,
)

# 导入 TYPE_CHECKING 类型检查模块
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # 导入必要的类型
    from typing import Any
    from matplotlib.artist import Artist
    from seaborn._core.scales import Scale

# DotBase 类，继承自 Mark 类
class DotBase(Mark):

    # 解析路径的内部方法，接受数据参数
    def _resolve_paths(self, data):

        # 路径列表初始化为空
        paths = []
        # 路径缓存字典初始化为空字典
        path_cache = {}
        # 获取标记数据
        marker = data["marker"]

        # 获取变换后的路径函数
        def get_transformed_path(m):
            return m.get_path().transformed(m.get_transform())

        # 如果标记是单一的 matplotlib 标记样式对象
        if isinstance(marker, mpl.markers.MarkerStyle):
            # 直接返回其变换后的路径
            return get_transformed_path(marker)

        # 对于每个标记对象 m
        for m in marker:
            # 如果该标记不在路径缓存中
            if m not in path_cache:
                # 计算其变换后的路径并存储到路径缓存中
                path_cache[m] = get_transformed_path(m)
            # 将路径添加到路径列表中
            paths.append(path_cache[m])
        
        # 返回所有标记的变换后的路径列表
        return paths

    # 解析属性的内部方法，接受数据和比例参数
    def _resolve_properties(self, data, scales):

        # 使用 resolve_properties 函数解析属性
        resolved = resolve_properties(self, data, scales)
        # 调用内部方法解析路径，存储到 resolved 字典中
        resolved["path"] = self._resolve_paths(resolved)
        # 计算点的大小，存储到 resolved 字典中
        resolved["size"] = resolved["pointsize"] ** 2

        # 如果数据是字典类型（单个点）
        if isinstance(data, dict):
            # 判断 marker 是否填充
            filled_marker = resolved["marker"].is_filled()
        else:
            # 对于多个点，判断每个 marker 是否填充，并存储结果到列表中
            filled_marker = [m.is_filled() for m in resolved["marker"]]

        # 计算填充属性，并更新到 resolved 字典中
        resolved["fill"] = resolved["fill"] * filled_marker

        # 返回解析后的属性字典
        return resolved

    # 绘制图形的内部方法，接受拆分生成器、比例和方向参数
    def _plot(self, split_gen, scales, orient):

        # 遍历拆分生成器生成的数据
        for _, data, ax in split_gen():

            # 将 x 和 y 数据合并为偏移量矩阵
            offsets = np.column_stack([data["x"], data["y"]])
            # 解析数据属性
            data = self._resolve_properties(data, scales)

            # 创建路径集合对象 PathCollection
            points = mpl.collections.PathCollection(
                offsets=offsets,
                paths=data["path"],
                sizes=data["size"],
                facecolors=data["facecolor"],
                edgecolors=data["edgecolor"],
                linewidths=data["linewidth"],
                linestyles=data["edgestyle"],
                transOffset=ax.transData,
                transform=mpl.transforms.IdentityTransform(),
                **self.artist_kws,
            )
            # 将路径集合添加到坐标轴 ax 上
            ax.add_collection(points)

    # 图例艺术家方法，接受变量列表、值和比例字典参数
    def _legend_artist(
        self, variables: list[str], value: Any, scales: dict[str, Scale],
    # 定义函数签名，指定返回类型为 Artist 对象
    ) -> Artist:
        # 从 variables 中创建一个键为变量 v，值为 value 的字典
        key = {v: value for v in variables}
        # 使用 _resolve_properties 方法解析属性，并得到结果 res
        res = self._resolve_properties(key, scales)
        
        # 创建一个 PathCollection 对象，用于绘制路径集合
        return mpl.collections.PathCollection(
            # 指定路径数组为 res 中的 "path"
            paths=[res["path"]],
            # 指定大小数组为 res 中的 "size"
            sizes=[res["size"]],
            # 指定填充颜色数组为 res 中的 "facecolor"
            facecolors=[res["facecolor"]],
            # 指定边缘颜色数组为 res 中的 "edgecolor"
            edgecolors=[res["edgecolor"]],
            # 指定线宽数组为 res 中的 "linewidth"
            linewidths=[res["linewidth"]],
            # 指定线型数组为 res 中的 "edgestyle"
            linestyles=[res["edgestyle"]],
            # 使用 IdentityTransform() 进行坐标变换
            transform=mpl.transforms.IdentityTransform(),
            # 使用 self.artist_kws 的其它关键字参数
            **self.artist_kws,
        )
@document_properties
@dataclass
class Dot(DotBase):
    """
    A mark suitable for dot plots or less-dense scatterplots.

    See also
    --------
    Dots : A dot mark defined by strokes to better handle overplotting.

    Examples
    --------
    .. include:: ../docstrings/objects.Dot.rst

    """
    # 定义标记的形状，可以在散点图中使用，不分组
    marker: MappableString = Mappable("o", grouping=False)
    # 点的大小，默认为6，不分组
    pointsize: MappableFloat = Mappable(6, grouping=False)  # TODO rcParam?
    # 点的描边粗细，默认为0.75，不分组
    stroke: MappableFloat = Mappable(.75, grouping=False)  # TODO rcParam?
    # 点的颜色，默认为"C0"，不分组
    color: MappableColor = Mappable("C0", grouping=False)
    # 点的透明度，默认为1，不分组
    alpha: MappableFloat = Mappable(1, grouping=False)
    # 是否填充点的内部，默认为True，不分组
    fill: MappableBool = Mappable(True, grouping=False)
    # 点的边缘颜色，默认与color相同，不分组
    edgecolor: MappableColor = Mappable(depend="color", grouping=False)
    # 点的边缘透明度，默认与alpha相同，不分组
    edgealpha: MappableFloat = Mappable(depend="alpha", grouping=False)
    # 点的边缘宽度，默认为0.5，不分组
    edgewidth: MappableFloat = Mappable(.5, grouping=False)  # TODO rcParam?
    # 点的边缘样式，默认为实线，不分组
    edgestyle: MappableStyle = Mappable("-", grouping=False)

    def _resolve_properties(self, data, scales):
        # 调用父类方法解析属性
        resolved = super()._resolve_properties(data, scales)
        # 获取填充状态
        filled = resolved["fill"]

        # 获取主要描边宽度和边缘描边宽度
        main_stroke = resolved["stroke"]
        edge_stroke = resolved["edgewidth"]
        # 根据填充状态决定线宽
        resolved["linewidth"] = np.where(filled, edge_stroke, main_stroke)

        # 解析主要颜色和边缘颜色
        main_color = resolve_color(self, data, "", scales)
        edge_color = resolve_color(self, data, "edge", scales)

        # 如果filled不是标量，扩展维度以便在np.where中使用rgba数组
        if not np.isscalar(filled):
            filled = filled[:, None]
        # 根据填充状态设置边缘颜色
        resolved["edgecolor"] = np.where(filled, edge_color, main_color)

        # 压缩filled数组
        filled = np.squeeze(filled)
        # 如果main_color是元组，处理其alpha通道
        if isinstance(main_color, tuple):
            main_color = tuple([*main_color[:3], main_color[3] * filled])
        else:
            main_color = np.c_[main_color[:, :3], main_color[:, 3] * filled]
        # 设置填充颜色
        resolved["facecolor"] = main_color

        return resolved


@document_properties
@dataclass
class Dots(DotBase):
    """
    A dot mark defined by strokes to better handle overplotting.

    See also
    --------
    Dot : A mark suitable for dot plots or less-dense scatterplots.

    Examples
    --------
    .. include:: ../docstrings/objects.Dots.rst

    """
    # TODO retype marker as MappableMarker
    # 定义标记的形状，可以在散点图中使用，不分组
    marker: MappableString = Mappable(rc="scatter.marker", grouping=False)
    # 点的大小，默认为4，不分组
    pointsize: MappableFloat = Mappable(4, grouping=False)  # TODO rcParam?
    # 点的描边粗细，默认为0.75，不分组
    stroke: MappableFloat = Mappable(.75, grouping=False)  # TODO rcParam?
    # 点的颜色，默认为"C0"，不分组
    color: MappableColor = Mappable("C0", grouping=False)
    # 点的透明度，默认为1，不分组
    alpha: MappableFloat = Mappable(1, grouping=False)  # TODO auto alpha?
    # 是否填充点的内部，默认为True，不分组
    fill: MappableBool = Mappable(True, grouping=False)
    # 填充的颜色，默认与color相同，不分组
    fillcolor: MappableColor = Mappable(depend="color", grouping=False)
    # 填充的透明度，默认为0.2，不分组
    fillalpha: MappableFloat = Mappable(.2, grouping=False)
    # 调用父类方法，解析属性数据，并将结果保存到 resolved 中
    resolved = super()._resolve_properties(data, scales)
    # 将键名 "stroke" 的值改为 "linewidth"，并从 resolved 中移除 "stroke"
    resolved["linewidth"] = resolved.pop("stroke")
    # 解析填充颜色，将结果保存到 resolved["facecolor"] 中
    resolved["facecolor"] = resolve_color(self, data, "fill", scales)
    # 解析边框颜色，将结果保存到 resolved["edgecolor"] 中
    resolved["edgecolor"] = resolve_color(self, data, "", scales)
    # 如果 "edgestyle" 键不存在于 resolved 中，则设为默认值 (0, None)
    resolved.setdefault("edgestyle", (0, None))

    # 获取 resolved["facecolor"] 的值
    fc = resolved["facecolor"]
    # 如果 facecolor 是元组形式
    if isinstance(fc, tuple):
        # 将 facecolor 的 alpha 通道乘以 resolved["fill"]，并更新 resolved["facecolor"]
        resolved["facecolor"] = fc[0], fc[1], fc[2], fc[3] * resolved["fill"]
    else:
        # 否则，假设 facecolor 是数组形式，将其 alpha 通道乘以 resolved["fill"]，这里存在 inplace 修改
        fc[:, 3] = fc[:, 3] * resolved["fill"]  # TODO Is inplace mod a problem?
        # 更新 resolved["facecolor"]
        resolved["facecolor"] = fc

    # 返回更新后的 resolved 字典
    return resolved
```