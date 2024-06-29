# `D:\src\scipysrc\matplotlib\galleries\plot_types\arrays\contourf.py`

```
"""
=================
contourf(X, Y, Z)
=================
Plot filled contours.

See `~matplotlib.axes.Axes.contourf`.
"""
# 导入 matplotlib.pyplot 库，并用 plt 作为别名
import matplotlib.pyplot as plt
# 导入 numpy 库，并用 np 作为别名
import numpy as np

# 使用 '_mpl-gallery-nogrid' 样式
plt.style.use('_mpl-gallery-nogrid')

# 生成数据
# 创建 X 和 Y 网格，范围从 -3 到 3，共 256 个点
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
# 计算 Z 值，根据给定的公式
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
# 生成等高线的级别，范围从 Z 的最小值到最大值，共 7 个级别
levels = np.linspace(Z.min(), Z.max(), 7)

# 绘图
# 创建图形和坐标轴对象
fig, ax = plt.subplots()
# 绘制填充的等高线图，使用之前生成的 X, Y, Z 数据和指定的 levels
ax.contourf(X, Y, Z, levels=levels)

# 显示图形
plt.show()
```