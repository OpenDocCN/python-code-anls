# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\donut.py`

```
"""
=============
Mmh Donuts!!!
=============

Draw donuts (miam!) using `~.path.Path`\s and `~.patches.PathPatch`\es.
This example shows the effect of the path's orientations in a compound path.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 库

import matplotlib.patches as mpatches  # 导入 matplotlib 的 patches 模块
import matplotlib.path as mpath  # 导入 matplotlib 的 path 模块


def wise(v):
    # 根据给定的方向值 v 返回相应的方向字符串
    if v == 1:
        return "CCW"  # 逆时针方向
    else:
        return "CW"  # 顺时针方向


def make_circle(r):
    # 创建一个半径为 r 的圆的顶点坐标
    t = np.arange(0, np.pi * 2.0, 0.01)  # 创建角度从 0 到 2π 的数组
    t = t.reshape((len(t), 1))  # 将 t 转换为列向量
    x = r * np.cos(t)  # 计算圆上点的 x 坐标
    y = r * np.sin(t)  # 计算圆上点的 y 坐标
    return np.hstack((x, y))  # 返回 x 和 y 坐标的水平堆叠


Path = mpath.Path  # 将 mpath 中的 Path 类赋值给 Path

fig, ax = plt.subplots()  # 创建图形和子图

inside_vertices = make_circle(0.5)  # 创建内圆的顶点坐标
outside_vertices = make_circle(1.0)  # 创建外圆的顶点坐标
codes = np.ones(
    len(inside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO  # 创建指令数组，默认为 LINETO
codes[0] = mpath.Path.MOVETO  # 将第一个指令设置为 MOVETO，即起始点

for i, (inside, outside) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
    # 合并内外圆的顶点坐标，根据参数顺序调整顶点顺序
    vertices = np.concatenate((outside_vertices[::outside],
                               inside_vertices[::inside]))
    # 平移路径
    vertices[:, 0] += i * 2.5
    # 创建 Path 对象
    all_codes = np.concatenate((codes, codes))  # 创建所有顶点对应的指令数组
    path = mpath.Path(vertices, all_codes)  # 创建路径对象
    # 创建并添加 PathPatch 对象到图形中
    patch = mpatches.PathPatch(path, facecolor='#885500', edgecolor='black')
    ax.add_patch(patch)  # 将 patch 添加到子图 ax 中

    # 添加标注说明
    ax.annotate(f"Outside {wise(outside)},\nInside {wise(inside)}",
                (i * 2.5, -1.5), va="top", ha="center")

ax.set_xlim(-2, 10)  # 设置 x 轴的显示范围
ax.set_ylim(-3, 2)   # 设置 y 轴的显示范围
ax.set_title('Mmm, donuts!')  # 设置图形的标题
ax.set_aspect(1.0)  # 设置纵横比为 1.0
plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.path`
#    - `matplotlib.path.Path`
#    - `matplotlib.patches`
#    - `matplotlib.patches.PathPatch`
#    - `matplotlib.patches.Circle`
#    - `matplotlib.axes.Axes.add_patch`
#    - `matplotlib.axes.Axes.annotate`
#    - `matplotlib.axes.Axes.set_aspect`
#    - `matplotlib.axes.Axes.set_xlim`
#    - `matplotlib.axes.Axes.set_ylim`
#    - `matplotlib.axes.Axes.set_title`
```