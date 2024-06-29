# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\simple_axisline3.py`

```
"""
================
Simple Axisline3
================

"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 从 mpl_toolkits.axisartist.axislines 模块中导入 Axes 类
from mpl_toolkits.axisartist.axislines import Axes

# 创建一个新的图形对象，指定大小为 3x3 英寸
fig = plt.figure(figsize=(3, 3))

# 在图形上添加一个子图，使用 Axes 类来创建
ax = fig.add_subplot(axes_class=Axes)

# 设置右边的轴线不可见
ax.axis["right"].set_visible(False)

# 设置顶部的轴线不可见
ax.axis["top"].set_visible(False)

# 显示绘制的图形
plt.show()
```