# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\invert_axes.py`

```
"""
=============
Inverted axis
=============

This example demonstrates two ways to invert the direction of an axis:

- If you want to set *explicit axis limits* anyway, e.g. via `~.Axes.set_xlim`, you
  can swap the limit values: ``set_xlim(4, 0)`` instead of ``set_xlim(0, 4)``.
- Use `.Axis.set_inverted` if you only want to invert the axis *without modifying
  the limits*, i.e. keep existing limits or existing autoscaling behavior.
"""

# 导入 matplotlib 库
import matplotlib.pyplot as plt
# 导入 numpy 库
import numpy as np

# 创建数据
x = np.arange(0.01, 4.0, 0.01)
y = np.exp(-x)

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4,  4), layout="constrained")
# 设置总标题
fig.suptitle('Inverted axis with ...')

# 在第一个子图中绘制图形
ax1.plot(x, y)
# 设置 x 轴的固定反向限制
ax1.set_xlim(4, 0)   # inverted fixed limits
# 设置子图标题
ax1.set_title('fixed limits: set_xlim(4, 0)')
# 设置 x 轴标签
ax1.set_xlabel('decreasing x ⟶')
# 显示网格
ax1.grid(True)

# 在第二个子图中绘制图形
ax2.plot(x, y)
# 设置 x 轴自动缩放的反向
ax2.xaxis.set_inverted(True)  # inverted axis with autoscaling
# 设置子图标题
ax2.set_title('autoscaling: set_inverted(True)')
# 设置 x 轴标签
ax2.set_xlabel('decreasing x ⟶')
# 显示网格
ax2.grid(True)

# 显示图形
plt.show()
```