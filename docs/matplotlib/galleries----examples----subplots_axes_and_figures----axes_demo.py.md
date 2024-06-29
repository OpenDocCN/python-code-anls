# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\axes_demo.py`

```py
"""
=========
Axes Demo
=========

Example use of ``fig.add_axes`` to create inset Axes within the main plot Axes.

Please see also the :ref:`axes_grid_examples` section, and the following three
examples:

- :doc:`/gallery/subplots_axes_and_figures/zoom_inset_axes`
- :doc:`/gallery/axes_grid1/inset_locator_demo`
- :doc:`/gallery/axes_grid1/inset_locator_demo2`
"""
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库
import numpy as np  # 导入 numpy 库，并简称为 np

np.random.seed(19680801)  # 设置随机种子以保证随机数可复现性

# 创建用于绘图的数据
dt = 0.001
t = np.arange(0.0, 10.0, dt)
r = np.exp(-t[:1000] / 0.05)  # 冲激响应
x = np.random.randn(len(t))
s = np.convolve(x, r)[:len(x)] * dt  # 彩色噪声

# 创建主图 Axes 对象，并绘制主要曲线
fig, main_ax = plt.subplots()
main_ax.plot(t, s)
main_ax.set_xlim(0, 1)  # 设置 x 轴范围
main_ax.set_ylim(1.1 * np.min(s), 2 * np.max(s))  # 设置 y 轴范围
main_ax.set_xlabel('time (s)')  # 设置 x 轴标签
main_ax.set_ylabel('current (nA)')  # 设置 y 轴标签
main_ax.set_title('Gaussian colored noise')  # 设置图表标题

# 创建第一个嵌入的 Axes 对象，位于主图 Axes 上方右侧
right_inset_ax = fig.add_axes([.65, .6, .2, .2], facecolor='k')
right_inset_ax.hist(s, 400, density=True)  # 绘制直方图
right_inset_ax.set(title='Probability', xticks=[], yticks=[])  # 设置标题和轴标签为空

# 创建第二个嵌入的 Axes 对象，位于主图 Axes 上方左侧
left_inset_ax = fig.add_axes([.2, .6, .2, .2], facecolor='k')
left_inset_ax.plot(t[:len(r)], r)  # 绘制曲线
left_inset_ax.set(title='Impulse response', xlim=(0, .2), xticks=[], yticks=[])  # 设置标题、x 轴范围和轴标签为空

plt.show()  # 显示图形
```