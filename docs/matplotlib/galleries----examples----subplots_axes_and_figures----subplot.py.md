# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\subplot.py`

```
"""
=================
Multiple subplots
=================

Simple demo with multiple subplots.

For more options, see :doc:`/gallery/subplots_axes_and_figures/subplots_demo`.

.. redirect-from:: /gallery/subplots_axes_and_figures/subplot_demo
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

# Create some fake data.
x1 = np.linspace(0.0, 5.0)  # 创建从 0 到 5 的等间隔数据
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)  # 计算 y1 的值
x2 = np.linspace(0.0, 2.0)  # 创建从 0 到 2 的等间隔数据
y2 = np.cos(2 * np.pi * x2)  # 计算 y2 的值

# %%
# `~.pyplot.subplots()` is the recommended method to generate simple subplot
# arrangements:

fig, (ax1, ax2) = plt.subplots(2, 1)  # 创建一个包含两个子图的图形对象
fig.suptitle('A tale of 2 subplots')  # 设置图形对象的总标题

ax1.plot(x1, y1, 'o-')  # 在第一个子图上绘制 x1, y1 的散点图
ax1.set_ylabel('Damped oscillation')  # 设置第一个子图的 y 轴标签

ax2.plot(x2, y2, '.-')  # 在第二个子图上绘制 x2, y2 的折线图
ax2.set_xlabel('time (s)')  # 设置第二个子图的 x 轴标签
ax2.set_ylabel('Undamped')  # 设置第二个子图的 y 轴标签

plt.show()  # 显示图形

# %%
# Subplots can also be generated one at a time using `~.pyplot.subplot()`:

plt.subplot(2, 1, 1)  # 创建一个包含两个子图的图形对象的第一个子图
plt.plot(x1, y1, 'o-')  # 在第一个子图上绘制 x1, y1 的散点图
plt.title('A tale of 2 subplots')  # 设置第一个子图的标题
plt.ylabel('Damped oscillation')  # 设置第一个子图的 y 轴标签

plt.subplot(2, 1, 2)  # 创建一个包含两个子图的图形对象的第二个子图
plt.plot(x2, y2, '.-')  # 在第二个子图上绘制 x2, y2 的折线图
plt.xlabel('time (s)')  # 设置第二个子图的 x 轴标签
plt.ylabel('Undamped')  # 设置第二个子图的 y 轴标签

plt.show()  # 显示图形
```