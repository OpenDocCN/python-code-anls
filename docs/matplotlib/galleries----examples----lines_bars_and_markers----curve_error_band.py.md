# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\curve_error_band.py`

```py
"""
=====================
Curve with error band
=====================

This example illustrates how to draw an error band around a parametrized curve.

A parametrized curve x(t), y(t) can directly be drawn using `~.Axes.plot`.
"""
# sphinx_gallery_thumbnail_number = 2

# 导入需要的库
import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy

from matplotlib.patches import PathPatch  # 导入 PathPatch 类
from matplotlib.path import Path  # 导入 Path 类

# 设置参数
N = 400  # 曲线上的点数
t = np.linspace(0, 2 * np.pi, N)  # 参数 t 在 [0, 2π] 上均匀分布
r = 0.5 + np.cos(t)  # 构造曲线的半径 r
x, y = r * np.cos(t), r * np.sin(t)  # 计算曲线上每个点的 x 和 y 坐标

# 创建图像和坐标系
fig, ax = plt.subplots()  # 创建图像和坐标系对象
ax.plot(x, y, "k")  # 绘制曲线，颜色为黑色
ax.set(aspect=1)  # 设置坐标系纵横比为1

# %%
# An error band can be used to indicate the uncertainty of the curve.
# In this example we assume that the error can be given as a scalar *err*
# that describes the uncertainty perpendicular to the curve in every point.
#
# We visualize this error as a colored band around the path using a
# `.PathPatch`. The patch is created from two path segments *(xp, yp)*, and
# *(xn, yn)* that are shifted by +/- *err* perpendicular to the curve *(x, y)*.
#
# Note: This method of using a `.PathPatch` is suited to arbitrary curves in
# 2D. If you just have a standard y-vs.-x plot, you can use the simpler
# `~.Axes.fill_between` method (see also
# :doc:`/gallery/lines_bars_and_markers/fill_between_demo`).

# 定义函数：绘制误差带
def draw_error_band(ax, x, y, err, **kwargs):
    # 计算法线向量，通过中心差分法计算（首尾使用前向和后向差分）
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)  # 计算每段的长度
    nx = dy / l  # x 方向上的法线分量
    ny = -dx / l  # y 方向上的法线分量

    # 误差带的两个端点
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err

    # 构建误差带的路径
    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T  # 合并顶点坐标
    codes = np.full(len(vertices), Path.LINETO)  # 所有点的路径代码为 LINETO
    codes[0] = codes[len(xp)] = Path.MOVETO  # 第一个点和中间点的路径代码为 MOVETO
    path = Path(vertices, codes)  # 创建 Path 对象
    ax.add_patch(PathPatch(path, **kwargs))  # 添加误差带路径的 PathPatch 对象到坐标系

# 创建包含两个子图的图像
_, axs = plt.subplots(1, 2, layout='constrained', sharex=True, sharey=True)

# 不同误差条件的配置
errs = [
    (axs[0], "constant error", 0.05),  # 固定误差
    (axs[1], "variable error", 0.05 * np.sin(2 * t) ** 2 + 0.04),  # 变化误差
]

# 遍历每个子图和误差设置
for i, (ax, title, err) in enumerate(errs):
    ax.set(title=title, aspect=1, xticks=[], yticks=[])  # 设置子图标题和纵横比，隐藏刻度
    ax.plot(x, y, "k")  # 绘制原始曲线，颜色为黑色
    draw_error_band(ax, x, y, err=err,  # 绘制误差带
                    facecolor=f"C{i}", edgecolor="none", alpha=.3)

plt.show()  # 显示图像

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches.PathPatch`
#    - `matplotlib.path.Path`
```