# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\hist2d.py`

```
"""
============
hist2d(x, y)
============
Make a 2D histogram plot.

See `~matplotlib.axes.Axes.hist2d`.
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 使用指定的样式表 '_mpl-gallery-nogrid' 进行绘图设置
plt.style.use('_mpl-gallery-nogrid')

# 生成数据: x 和 y 是相关的数据加上噪声
np.random.seed(1)
x = np.random.randn(5000)
y = 1.2 * x + np.random.randn(5000) / 3

# 创建图形和坐标系对象
fig, ax = plt.subplots()

# 绘制 2D 直方图
ax.hist2d(x, y, bins=(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1)))

# 设置 x 和 y 轴的显示范围
ax.set(xlim=(-2, 2), ylim=(-3, 3))

# 显示图形
plt.show()
```