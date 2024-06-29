# `D:\src\scipysrc\matplotlib\galleries\plot_types\basic\scatter_plot.py`

```
"""
=============
scatter(x, y)
=============
A scatter plot of y vs. x with varying marker size and/or color.

See `~matplotlib.axes.Axes.scatter`.
"""
# 导入 matplotlib 库中的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于生成数组和数学运算
import numpy as np

# 使用 '_mpl-gallery' 风格样式
plt.style.use('_mpl-gallery')

# 生成数据
np.random.seed(3)
# 生成 x 数据，服从均值为 4，标准差为 2 的正态分布，共 24 个数据点
x = 4 + np.random.normal(0, 2, 24)
# 生成 y 数据，同样服从均值为 4，标准差为 2 的正态分布，数据点个数与 x 相同
y = 4 + np.random.normal(0, 2, len(x))
# 生成用于指定每个点大小的随机数组，范围在 15 到 80 之间，数据点个数与 x 相同
sizes = np.random.uniform(15, 80, len(x))
# 生成用于指定每个点颜色的随机数组，范围在 15 到 80 之间，数据点个数与 x 相同
colors = np.random.uniform(15, 80, len(x))

# 创建图形和轴对象
fig, ax = plt.subplots()

# 绘制散点图，设置点的大小和颜色，颜色范围从 0 到 100
ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

# 设置 x 轴和 y 轴的范围，并指定刻度位置
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```