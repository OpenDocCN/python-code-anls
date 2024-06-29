# `D:\src\scipysrc\matplotlib\galleries\examples\misc\coords_report.py`

```py
"""
=============
Coords Report
=============

Override the default reporting of coords as the mouse moves over the Axes
in an interactive backend.
"""

# 导入 matplotlib 的 pyplot 模块并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并命名为 np
import numpy as np


# 定义一个函数 millions，用于格式化数值为百万单位的字符串
def millions(x):
    return '$%1.1fM' % (x * 1e-6)


# 固定随机数生成器的种子，以便可复现性
np.random.seed(19680801)

# 生成长度为 20 的随机数组 x，取值范围在 [0, 1) 之间
x = np.random.rand(20)
# 生成长度为 20 的随机数组 y，取值范围在 [0, 10M) 之间
y = 1e7 * np.random.rand(20)

# 创建一个新的图形和一个子图(ax)，并将其返回
fig, ax = plt.subplots()

# 设置 ax 对象的 fmt_ydata 属性为 millions 函数，用于格式化 y 轴上的数据显示
ax.fmt_ydata = millions
# 在子图 ax 上绘制散点图，数据为 (x, y)，使用 'o' 表示散点样式
plt.plot(x, y, 'o')

# 显示图形
plt.show()
```