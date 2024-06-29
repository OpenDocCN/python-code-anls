# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\hexbin.py`

```py
"""
===============
hexbin(x, y, C)
===============
Make a 2D hexagonal binning plot of points x, y.

See `~matplotlib.axes.Axes.hexbin`.
"""
# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 使用指定的样式表 '_mpl-gallery-nogrid'
plt.style.use('_mpl-gallery-nogrid')

# 生成数据：包含相关性和噪声
np.random.seed(1)
# 生成服从标准正态分布的随机数据 x
x = np.random.randn(5000)
# 生成带有一定相关性和噪声的随机数据 y
y = 1.2 * x + np.random.randn(5000) / 3

# 创建图形和子图对象
fig, ax = plt.subplots()

# 绘制二维六边形的二进制图，用于显示点 x, y 的分布情况
ax.hexbin(x, y, gridsize=20)

# 设置坐标轴的范围
ax.set(xlim=(-2, 2), ylim=(-3, 3))

# 显示图形
plt.show()
```