# `D:\src\scipysrc\matplotlib\galleries\examples\scales\semilogx_demo.py`

```
"""
========
Log Axis
========

.. redirect-from:: /gallery/scales/log_test

This is an example of assigning a log-scale for the x-axis using
`~.axes.Axes.semilogx`.
"""

# 导入 matplotlib.pyplot 库，通常用 plt 作为别名
import matplotlib.pyplot as plt
# 导入 numpy 库，通常用 np 作为别名
import numpy as np

# 创建一个新的图形和一个子图
fig, ax = plt.subplots()

# 设置时间步长为 0.01
dt = 0.01
# 创建一个从 dt 到 20.0 的时间数组 t
t = np.arange(dt, 20.0, dt)

# 在子图 ax 上使用 semilogx 方法绘制 t 和 np.exp(-t / 5.0) 的图像
ax.semilogx(t, np.exp(-t / 5.0))
# 添加网格线到子图 ax
ax.grid()

# 显示图形
plt.show()
```