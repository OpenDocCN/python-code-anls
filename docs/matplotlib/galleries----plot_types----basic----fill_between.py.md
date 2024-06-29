# `D:\src\scipysrc\matplotlib\galleries\plot_types\basic\fill_between.py`

```
"""
=======================
fill_between(x, y1, y2)
=======================
Fill the area between two horizontal curves.

See `~matplotlib.axes.Axes.fill_between`.
"""

# 导入 matplotlib.pyplot 库，并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并命名为 np
import numpy as np

# 使用指定的样式 '_mpl-gallery'
plt.style.use('_mpl-gallery')

# 生成数据
np.random.seed(1)
# 生成等间隔的数组 x
x = np.linspace(0, 8, 16)
# 生成带有随机扰动的数组 y1
y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
# 生成带有随机扰动的数组 y2
y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 填充两条水平曲线 y1 和 y2 之间的区域，设置透明度为 0.5，线宽为 0
ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
# 绘制 x 与 (y1 + y2)/2 的均值曲线，设置线宽为 2
ax.plot(x, (y1 + y2)/2, linewidth=2)

# 设置坐标轴的范围和刻度
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```