# `D:\src\scipysrc\seaborn\seaborn\_marks\text.py`

```
# 导入必要的模块和类
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
from matplotlib.transforms import ScaledTranslation

# 导入 seaborn 库中的相关基础组件
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

# 声明一个类装饰器，用于文档化对象的属性
@document_properties
# 使用 dataclass 装饰器，定义一个数据类 Text，继承自 Mark 类
@dataclass
class Text(Mark):
    """
    A textual mark to annotate or represent data values.

    Examples
    --------
    .. include:: ../docstrings/objects.Text.rst

    """
    # 文本内容，可映射到 MappableString 对象，默认为空字符串
    text: MappableString = Mappable("")
    # 文本颜色，可映射到 MappableColor 对象，默认为黑色
    color: MappableColor = Mappable("k")
    # 透明度，可映射到 MappableFloat 对象，默认为 1（完全不透明）
    alpha: MappableFloat = Mappable(1)
    # 字体大小，可映射到 MappableFloat 对象，从全局的 font.size 属性获取
    fontsize: MappableFloat = Mappable(rc="font.size")
    # 水平对齐方式，可映射到 MappableString 对象，默认为居中对齐
    halign: MappableString = Mappable("center")
    # 垂直对齐方式，可映射到 MappableString 对象，默认为基线居中对齐
    valign: MappableString = Mappable("center_baseline")
    # 偏移量，可映射到 MappableFloat 对象，默认为 4
    offset: MappableFloat = Mappable(4)

    # 定义一个私有方法 _plot，用于绘制图形
    def _plot(self, split_gen, scales, orient):

        # 使用 defaultdict 创建一个空列表的字典 ax_data
        ax_data = defaultdict(list)

        # 迭代 split_gen() 生成器产生的键、数据和坐标轴对象 ax
        for keys, data, ax in split_gen():

            # 解析属性值，获取文本的水平对齐、垂直对齐、字体大小和偏移量
            vals = resolve_properties(self, keys, scales)
            # 解析颜色属性
            color = resolve_color(self, keys, "", scales)

            # 获取水平对齐方式、垂直对齐方式、字体大小和偏移量除以 72（dpi 转换）
            halign = vals["halign"]
            valign = vals["valign"]
            fontsize = vals["fontsize"]
            offset = vals["offset"] / 72

            # 创建一个缩放平移对象 offset_trans，根据对齐方式和坐标轴的 DPI 缩放转换
            offset_trans = ScaledTranslation(
                {"right": -offset, "left": +offset}.get(halign, 0),
                {"top": -offset, "bottom": +offset, "baseline": +offset}.get(valign, 0),
                ax.figure.dpi_scale_trans,
            )

            # 遍历数据的字典记录，为每行创建一个文本对象 artist
            for row in data.to_dict("records"):
                artist = mpl.text.Text(
                    x=row["x"],
                    y=row["y"],
                    text=str(row.get("text", vals["text"])),
                    color=color,
                    fontsize=fontsize,
                    horizontalalignment=halign,
                    verticalalignment=valign,
                    transform=ax.transData + offset_trans,
                    **self.artist_kws,
                )
                # 将文本对象添加到坐标轴 ax 上
                ax.add_artist(artist)
                # 将坐标数据添加到 ax_data[ax] 列表中
                ax_data[ax].append([row["x"], row["y"]])

        # 遍历 ax_data 字典，更新每个坐标轴 ax 的数据限制
        for ax, ax_vals in ax_data.items():
            ax.update_datalim(np.array(ax_vals))
```