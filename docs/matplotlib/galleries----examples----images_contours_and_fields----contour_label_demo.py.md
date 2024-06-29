# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\contour_label_demo.py`

```py
"""
==================
Contour Label Demo
==================

Illustrate some of the more advanced things that one can do with
contour labels.

See also the :doc:`contour demo example
</gallery/images_contours_and_fields/contour_demo>`.
"""

# 导入所需的库
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker

# %%
# Define our surface

# 设置 delta 值
delta = 0.025
# 创建 X 和 Y 轴的数值范围
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
# 创建网格数据
X, Y = np.meshgrid(x, y)
# 定义两个高斯分布函数生成的表面
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# %%
# Make contour labels with custom level formatters

# 定义自定义格式化函数，用于去除小数后的零并添加百分号
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


# 创建基本的等高线图
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
# 标记等高线，并使用自定义格式化函数
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

# %%
# Label contours with arbitrary strings using a dictionary

fig1, ax1 = plt.subplots()

# 创建基本的等高线图
CS1 = ax1.contour(X, Y, Z)

# 定义一个字典，将每个等高线级别映射到自定义字符串
fmt = {}
strs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
for l, s in zip(CS1.levels, strs):
    fmt[l] = s

# 标记每隔一个等高线级别，并使用自定义字符串
ax1.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=10)

# %%
# Use a Formatter

fig2, ax2 = plt.subplots()

# 创建基于对数尺度的等高线图
CS2 = ax2.contour(X, Y, 100**Z, locator=plt.LogLocator())
# 使用对数格式化器
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
ax2.clabel(CS2, CS2.levels, fmt=fmt)
ax2.set_title("$100^Z$")

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.ticker.LogFormatterMathtext`
#    - `matplotlib.ticker.TickHelper.create_dummy_axis`
```