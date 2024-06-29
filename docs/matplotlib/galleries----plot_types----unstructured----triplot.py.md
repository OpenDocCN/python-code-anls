# `D:\src\scipysrc\matplotlib\galleries\plot_types\unstructured\triplot.py`

```py
"""
=============
triplot(x, y)
=============
Draw an unstructured triangular grid as lines and/or markers.

See `~matplotlib.axes.Axes.triplot`.
"""
# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 使用特定的样式表 '_mpl-gallery-nogrid'
plt.style.use('_mpl-gallery-nogrid')

# 生成数据:
# 设定随机数种子为1，以便结果可重复
np.random.seed(1)
# 生成在区间 [-3, 3] 内均匀分布的 256 个随机数作为 x 坐标
x = np.random.uniform(-3, 3, 256)
# 生成在区间 [-3, 3] 内均匀分布的 256 个随机数作为 y 坐标
y = np.random.uniform(-3, 3, 256)
# 根据 x, y 计算出 z 值，用于显示颜色深浅
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)

# 绘图:
# 创建一个图形窗口和一个坐标轴对象
fig, ax = plt.subplots()

# 在坐标轴上绘制三角形的轮廓图
ax.triplot(x, y)

# 设定 x 轴和 y 轴的显示范围为 [-3, 3]
ax.set(xlim=(-3, 3), ylim=(-3, 3))

# 显示图形
plt.show()
```