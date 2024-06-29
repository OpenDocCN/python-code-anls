# `D:\src\scipysrc\matplotlib\galleries\plot_types\arrays\contour.py`

```
"""
================
contour(X, Y, Z)
================
Plot contour lines.

See `~matplotlib.axes.Axes.contour`.
"""
# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 使用指定的样式
plt.style.use('_mpl-gallery-nogrid')

# 生成数据
# 创建 X 和 Y 的网格，范围是从 -3 到 3，256个点
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
# 计算 Z 值，使用给定的数学表达式
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
# 设置等高线的级别，均匀分布在 Z 的最小值和最大值之间，总共 7 个级别
levels = np.linspace(np.min(Z), np.max(Z), 7)

# 绘图
# 创建图形和子图对象
fig, ax = plt.subplots()

# 绘制等高线图，使用之前生成的 X、Y、Z 和 levels
ax.contour(X, Y, Z, levels=levels)

# 显示图形
plt.show()
```