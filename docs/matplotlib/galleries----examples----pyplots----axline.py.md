# `D:\src\scipysrc\matplotlib\galleries\examples\pyplots\axline.py`

```py
"""
==============
Infinite lines
==============

`~.axes.Axes.axvline` and `~.axes.Axes.axhline` draw infinite vertical /
horizontal lines, at given *x* / *y* positions. They are usually used to mark
special data values, e.g. in this example the center and limit values of the
sigmoid function.

`~.axes.Axes.axline` draws infinite straight lines in arbitrary directions.
"""

# 导入 matplotlib 库
import matplotlib.pyplot as plt
import numpy as np

# 生成从 -10 到 10 等间隔的 100 个数的数组
t = np.linspace(-10, 10, 100)
# 计算 sigmoid 函数
sig = 1 / (1 + np.exp(-t))

# 绘制 y=0, y=0.5, y=1 的水平虚线
plt.axhline(y=0, color="black", linestyle="--")
plt.axhline(y=0.5, color="black", linestyle=":")
plt.axhline(y=1.0, color="black", linestyle="--")

# 绘制垂直灰色直线（未指定位置，默认在 x=0 处）
plt.axvline(color="grey")

# 绘制斜率为 0.25，通过点 (0, 0.5) 的直线，线型为虚线
plt.axline((0, 0.5), slope=0.25, color="black", linestyle=(0, (5, 5)))

# 绘制 sigmoid 函数曲线，线宽为 2
plt.plot(t, sig, linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")

# 设置 x 轴范围为 -10 到 10
plt.xlim(-10, 10)
# 设置 x 轴标签为 "t"
plt.xlabel("t")
# 添加图例，字体大小为 14
plt.legend(fontsize=14)
# 显示图形
plt.show()

# %%
# `~.axes.Axes.axline` can also be used with a ``transform`` parameter, which
# applies to the point, but not to the slope. This can be useful for drawing
# diagonal grid lines with a fixed slope, which stay in place when the
# plot limits are moved.

# 在转换后的坐标系中绘制斜率为 0.5 的直线，位置从 (-2, 0) 到 (1, 0)，共 10 条
for pos in np.linspace(-2, 1, 10):
    plt.axline((pos, 0), slope=0.5, color='k', transform=plt.gca().transAxes)

# 设置 y 轴范围为 [0, 1]
plt.ylim([0, 1])
# 设置 x 轴范围为 [0, 1]
plt.xlim([0, 1])
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.axhline` / `matplotlib.pyplot.axhline`
#    - `matplotlib.axes.Axes.axvline` / `matplotlib.pyplot.axvline`
#    - `matplotlib.axes.Axes.axline` / `matplotlib.pyplot.axline`
```