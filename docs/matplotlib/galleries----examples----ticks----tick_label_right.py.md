# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\tick_label_right.py`

```
"""
============================================
Set default y-axis tick labels on the right
============================================

We can use :rc:`ytick.labelright`, :rc:`ytick.right`, :rc:`ytick.labelleft`,
and :rc:`ytick.left` to control where on the axes ticks and their labels
appear. These properties can also be set in ``.matplotlib/matplotlibrc``.

"""
# 导入matplotlib库，并重命名为plt
import matplotlib.pyplot as plt
# 导入numpy库，并重命名为np
import numpy as np

# 设置右侧刻度标签显示在y轴上
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
# 设置左侧刻度标签不显示在y轴上
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

# 创建一个包含10个元素的数组
x = np.arange(10)

# 创建一个包含两个子图的图形对象，共享x轴，大小为6x6英寸
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

# 在第一个子图ax0上绘制x数组的折线图
ax0.plot(x)
# 设置第一个子图ax0的y轴刻度标签显示在左侧
ax0.yaxis.tick_left()

# 在第二个子图ax1上绘制x数组的折线图，使用默认的刻度参数，不显式调用tick_right()
ax1.plot(x)

# 显示图形
plt.show()
```