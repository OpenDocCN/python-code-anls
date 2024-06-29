# `D:\src\scipysrc\matplotlib\galleries\plot_types\arrays\streamplot.py`

```py
"""
======================
streamplot(X, Y, U, V)
======================
Draw streamlines of a vector flow.

See `~matplotlib.axes.Axes.streamplot`.
"""
# 导入matplotlib.pyplot库，并将其命名为plt，用于绘图操作
import matplotlib.pyplot as plt
# 导入numpy库，并将其命名为np，用于数值计算和数组操作
import numpy as np

# 使用指定的Matplotlib风格样式
plt.style.use('_mpl-gallery-nogrid')

# 创建二维网格X和Y，范围在[-3, 3]内，步长为256
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
# 创建流函数Z，基于给定的公式
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
# 从流函数Z计算得到U和V，表示流场的水平和垂直分量
V = np.diff(Z[1:, :], axis=1)
U = -np.diff(Z[:, 1:], axis=0)

# 创建图形和轴对象
fig, ax = plt.subplots()

# 使用流函数的网格和U、V数组绘制流线图
ax.streamplot(X[1:, 1:], Y[1:, 1:], U, V)

# 显示绘制的图形
plt.show()
```