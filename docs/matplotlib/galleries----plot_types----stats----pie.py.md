# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\pie.py`

```py
"""
======
pie(x)
======
Plot a pie chart.

See `~matplotlib.axes.Axes.pie`.
"""
# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 使用自定义的样式 '_mpl-gallery-nogrid'
plt.style.use('_mpl-gallery-nogrid')

# 创建数据
x = [1, 2, 3, 4]
# 使用蓝色调色板生成颜色数组，根据数据长度进行线性分布
colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))

# 绘图
# 创建图形和轴对象
fig, ax = plt.subplots()
# 绘制饼图，设置颜色、半径、中心位置、楔形属性和边框
ax.pie(x, colors=colors, radius=3, center=(4, 4),
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

# 设置坐标轴的范围和刻度
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```