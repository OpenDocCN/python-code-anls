# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\multicursor.py`

```
"""
===========
Multicursor
===========

Showing a cursor on multiple plots simultaneously.

This example generates three Axes split over two different figures.  On
hovering the cursor over data in one subplot, the values of that datapoint are
shown in all Axes.
"""

# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np
# 导入 MultiCursor 类从 matplotlib.widgets 模块
from matplotlib.widgets import MultiCursor

# 生成时间轴数据
t = np.arange(0.0, 2.0, 0.01)
# 生成三种不同频率的正弦波数据
s1 = np.sin(2*np.pi*t)
s2 = np.sin(3*np.pi*t)
s3 = np.sin(4*np.pi*t)

# 创建包含两个子图的图像，共享 X 轴
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# 在第一个子图上绘制 s1 的正弦波曲线
ax1.plot(t, s1)
# 在第二个子图上绘制 s2 的正弦波曲线
ax2.plot(t, s2)

# 创建另一个图像，并在其上创建第三个子图
fig, ax3 = plt.subplots()
# 在第三个子图上绘制 s3 的正弦波曲线
ax3.plot(t, s3)

# 创建一个 MultiCursor 对象，将其绑定到三个子图上，颜色为红色，线宽为 1
multi = MultiCursor(None, (ax1, ax2, ax3), color='r', lw=1)

# 显示图形界面
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.MultiCursor`
```