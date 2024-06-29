# `D:\src\scipysrc\matplotlib\galleries\plot_types\unstructured\tripcolor.py`

```py
"""
==================
tripcolor(x, y, z)
==================
Create a pseudocolor plot of an unstructured triangular grid.

See `~matplotlib.axes.Axes.tripcolor`.
"""
# 导入 matplotlib.pyplot 库，用于绘图操作
import matplotlib.pyplot as plt
# 导入 numpy 库，用于生成数据
import numpy as np

# 使用特定的绘图风格 '_mpl-gallery-nogrid'
plt.style.use('_mpl-gallery-nogrid')

# 生成数据:
# 设置随机种子，以确保每次运行生成相同的随机数
np.random.seed(1)
# 在区间 [-3, 3) 中生成均匀分布的随机数，生成长度为 256 的一维数组
x = np.random.uniform(-3, 3, 256)
y = np.random.uniform(-3, 3, 256)
# 计算 z 值，根据复杂的函数关系来生成数据
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)

# 绘图:
# 创建一个新的图形和轴对象
fig, ax = plt.subplots()

# 在轴上绘制散点图，表示 (x, y) 数据点
ax.plot(x, y, 'o', markersize=2, color='grey')
# 在轴上绘制三角网格的伪彩色图，使用 (x, y, z) 数据进行绘制
ax.tripcolor(x, y, z)

# 设置 x 轴和 y 轴的限制范围
ax.set(xlim=(-3, 3), ylim=(-3, 3))

# 显示绘制的图形
plt.show()
```