# `D:\src\scipysrc\matplotlib\galleries\examples\color\color_by_yvalue.py`

```py
"""
================
Color by y-value
================

Use masked arrays to plot a line with different colors by y-value.
"""

# 导入 matplotlib.pyplot 库作为 plt 别名
import matplotlib.pyplot as plt
# 导入 numpy 库作为 np 别名
import numpy as np

# 生成一个从 0 到 2（不含）的数组，步长为 0.01
t = np.arange(0.0, 2.0, 0.01)
# 计算 t 对应的正弦值
s = np.sin(2 * np.pi * t)

# 设置上界和下界值
upper = 0.77
lower = -0.77

# 使用 masked arrays 来根据条件屏蔽部分数据，创建不同颜色的线
supper = np.ma.masked_where(s < upper, s)   # 将 s 中小于 upper 的值屏蔽
slower = np.ma.masked_where(s > lower, s)   # 将 s 中大于 lower 的值屏蔽
smiddle = np.ma.masked_where((s < lower) | (s > upper), s)   # 将 s 中不在 lower 和 upper 范围内的值屏蔽

# 创建图形和坐标系
fig, ax = plt.subplots()
# 绘制三条线，每条线的数据和颜色根据上面的 masked arrays 来确定
ax.plot(t, smiddle, t, slower, t, supper)
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
```