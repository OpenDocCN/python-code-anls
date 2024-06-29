# `D:\src\scipysrc\matplotlib\galleries\examples\scales\symlog_demo.py`

```py
"""
===========
Symlog Demo
===========

Example use of symlog (symmetric log) axis scaling.
"""
# 导入 matplotlib 的 pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 设置时间步长
dt = 0.01
# 创建 x 数组，从 -50.0 到 50.0，步长为 dt
x = np.arange(-50.0, 50.0, dt)
# 创建 y 数组，从 0 到 100.0，步长为 dt
y = np.arange(0, 100.0, dt)

# 创建包含三个子图的图像窗口
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)

# 在第一个子图上绘制 x vs y
ax0.plot(x, y)
# 设置 x 轴为对称对数坐标轴
ax0.set_xscale('symlog')
# 设置 y 轴标签
ax0.set_ylabel('symlogx')
# 显示网格线
ax0.grid()
# 显示次要刻度的 x 轴网格线
ax0.xaxis.grid(which='minor')

# 在第二个子图上绘制 y vs x
ax1.plot(y, x)
# 设置 y 轴为对称对数坐标轴
ax1.set_yscale('symlog')
# 设置 y 轴标签
ax1.set_ylabel('symlogy')

# 在第三个子图上绘制 x vs sin(x/3.0)
ax2.plot(x, np.sin(x / 3.0))
# 设置 x 和 y 轴都为对称对数坐标轴，指定线性区间的阈值为 0.015
ax2.set_xscale('symlog')
ax2.set_yscale('symlog', linthresh=0.015)
# 显示网格线
ax2.grid()
# 设置 y 轴标签
ax2.set_ylabel('symlog both')

# 调整子图的布局使其紧凑显示
fig.tight_layout()
# 显示图形
plt.show()

# %%
# 应注意，“symlog”坐标变换在其线性和对数区域之间的过渡处具有不连续的梯度。
# ``asinh`` 坐标轴刻度是一种替代技术，可以避免这些不连续性引起的视觉伪影。

# %%
#
# .. admonition:: References
#
#    - `matplotlib.scale.SymmetricalLogScale`
#    - `matplotlib.ticker.SymmetricalLogLocator`
#    - `matplotlib.scale.AsinhScale`
```