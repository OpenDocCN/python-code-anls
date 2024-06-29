# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\simple_plot.py`

```py
"""
===========
Simple Plot
===========

Create a simple plot.
"""

# 导入 matplotlib.pyplot 库并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np

# Data for plotting
# 生成一个从 0 到 2（不包括）的数组，步长为 0.01
t = np.arange(0.0, 2.0, 0.01)
# 计算 s，即 1 + sin(2πt) 的值
s = 1 + np.sin(2 * np.pi * t)

# 创建一个新的图形和轴对象
fig, ax = plt.subplots()
# 在轴对象上绘制 t 对应的 s 的曲线
ax.plot(t, s)

# 设置 x 轴标签为 'time (s)'，y 轴标签为 'voltage (mV)'
# 设置图表标题为 'About as simple as it gets, folks'
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
# 显示网格线
ax.grid()

# 将图形保存为文件 'test.png'
fig.savefig("test.png")
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
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.figure.Figure.savefig`
```