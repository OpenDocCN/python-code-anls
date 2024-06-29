# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\hist_plot.py`

```py
"""
=======
hist(x)
=======
Compute and plot a histogram.

See `~matplotlib.axes.Axes.hist`.
"""
# 导入 matplotlib.pyplot 模块，并使用 plt 别名
import matplotlib.pyplot as plt
# 导入 numpy 模块，并使用 np 别名
import numpy as np

# 使用 '_mpl-gallery' 风格样式
plt.style.use('_mpl-gallery')

# 生成数据
np.random.seed(1)
# 生成 200 个服从正态分布的随机数，均值为 4，标准差为 1.5
x = 4 + np.random.normal(0, 1.5, 200)

# 绘制直方图:
# 创建图形和轴对象
fig, ax = plt.subplots()

# 绘制直方图，设置分箱数为 8，边框线宽度为 0.5，边框颜色为白色
ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")

# 设置 x 轴的范围为 (0, 8)，设置 x 轴刻度为从 1 到 7 的整数
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       # 设置 y 轴的范围为 (0, 56)，设置 y 轴刻度为从 0 到 56 之间均匀分布的 9 个点
       ylim=(0, 56), yticks=np.linspace(0, 56, 9))

# 显示图形
plt.show()
```