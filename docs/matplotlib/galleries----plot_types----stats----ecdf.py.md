# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\ecdf.py`

```py
"""
=======
ecdf(x)
=======
Compute and plot the empirical cumulative distribution function of x.

See `~matplotlib.axes.Axes.ecdf`.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 数学计算库

# 设置绘图风格为 '_mpl-gallery'
plt.style.use('_mpl-gallery')

# 生成数据
np.random.seed(1)  # 设置随机种子以确保可重复性
x = 4 + np.random.normal(0, 1.5, 200)  # 生成均值为 4，标准差为 1.5 的正态分布随机数据，共 200 个数据点

# 绘图:
fig, ax = plt.subplots()  # 创建一个新的图形和坐标系对象
ax.ecdf(x)  # 绘制 x 的经验累积分布函数图
plt.show()  # 显示图形
```