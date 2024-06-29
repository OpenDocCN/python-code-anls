# `D:\src\scipysrc\matplotlib\galleries\plot_types\arrays\pcolormesh.py`

```
"""
===================
pcolormesh(X, Y, Z)
===================
Create a pseudocolor plot with a non-regular rectangular grid.

`~.axes.Axes.pcolormesh` is more flexible than `~.axes.Axes.imshow` in that
the x and y vectors need not be equally spaced (indeed they can be skewed).

"""
# 导入 matplotlib.pyplot 库，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np

# 使用自定义的 matplotlib 样式 '_mpl-gallery-nogrid'
plt.style.use('_mpl-gallery-nogrid')

# 创建一个不均匀采样的 x 值数组
x = [-3, -2, -1.6, -1.2, -.8, -.5, -.2, .1, .3, .5, .8, 1.1, 1.5, 1.9, 2.3, 3]
# 创建一个网格，X 是 x 的网格化版本，Y 是从 -3 到 3 均匀分布的 128 个点组成的数组
X, Y = np.meshgrid(x, np.linspace(-3, 3, 128))
# 根据 X, Y 计算 Z 值，这里是一个复杂的函数
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

# 创建一个图形和坐标轴对象
fig, ax = plt.subplots()

# 使用 ax 对象的 pcolormesh 方法绘制伪彩色图，使用之前计算得到的 X, Y, Z 数据
ax.pcolormesh(X, Y, Z, vmin=-0.5, vmax=1.0)

# 显示图形
plt.show()
```