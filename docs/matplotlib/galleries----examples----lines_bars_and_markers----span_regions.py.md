# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\span_regions.py`

```py
"""
==========================================================
Shade regions defined by a logical mask using fill_between
==========================================================
"""

# 导入需要的库
import matplotlib.pyplot as plt
import numpy as np

# 创建时间序列数据
t = np.arange(0.0, 2, 0.01)
# 计算正弦函数值
s = np.sin(2*np.pi*t)

# 创建图形和坐标轴对象
fig, ax = plt.subplots()

# 绘制正弦曲线
ax.plot(t, s, color='black')
# 添加水平线，位置为 y=0
ax.axhline(0, color='black')

# 使用 fill_between 方法填充正弦曲线上方区域（当 s > 0）
ax.fill_between(t, 1, where=s > 0, facecolor='green', alpha=.5)
# 使用 fill_between 方法填充正弦曲线下方区域（当 s < 0）
ax.fill_between(t, -1, where=s < 0, facecolor='red', alpha=.5)

# 显示图形
plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.fill_between`
```