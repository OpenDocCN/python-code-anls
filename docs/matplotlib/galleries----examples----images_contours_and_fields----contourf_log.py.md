# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\contourf_log.py`

```py
"""
============================
Contourf and log color scale
============================

Demonstrate use of a log color scale in contourf
"""

# 导入 matplotlib 库
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

from matplotlib import cm, ticker

# 定义网格的大小
N = 100
# 创建 x 和 y 的均匀分布
x = np.linspace(-3.0, 3.0, N)
y = np.linspace(-2.0, 2.0, N)

# 创建 X 和 Y 矩阵，用于网格化 x 和 y
X, Y = np.meshgrid(x, y)

# 创建一个低矮的隆起和一个有尖峰的表面
# 需要在 z/颜色轴上使用对数尺度，以便同时显示隆起和尖峰
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
z = Z1 + 50 * Z2

# 放入一些负值（左下角），以便在对数尺度下出现问题
z[:5, :5] = -1

# 下面的操作并非绝对必需，但它可以消除警告。取消注释以查看警告。
z = ma.masked_where(z <= 0, z)

# 自动选择级别有效；设置对数定位器告诉 contourf 使用对数尺度
fig, ax = plt.subplots()
cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)

# 或者，您可以手动设置级别和规范化：
# lev_exp = np.arange(np.floor(np.log10(z.min())-1),
#                    np.ceil(np.log10(z.max())+1))
# levs = np.power(10, lev_exp)
# cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())

# 添加色条
cbar = fig.colorbar(cs)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
#    - `matplotlib.ticker.LogLocator`
```