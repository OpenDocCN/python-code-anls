# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\align_labels_demo.py`

```
"""
==========================
Aligning Labels and Titles
==========================

Aligning xlabel, ylabel, and title using `.Figure.align_xlabels`,
`.Figure.align_ylabels`, and `.Figure.align_titles`.

`.Figure.align_labels` wraps the x and y label functions.

Note that the xlabel "XLabel1 1" would normally be much closer to the
x-axis, "YLabel0 0" would be much closer to the y-axis, and title
"Title0 0" would be much closer to the top of their respective axes.
"""
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入numpy数值计算库

# 创建一个2x2子图的图形对象，并使用'constrained'布局
fig, axs = plt.subplots(2, 2, layout='constrained')

# 左上角子图
ax = axs[0][0]
ax.plot(np.arange(0, 1e6, 1000))  # 绘制数据
ax.set_title('Title0 0')  # 设置子图标题
ax.set_ylabel('YLabel0 0')  # 设置y轴标签

# 右上角子图
ax = axs[0][1]
ax.plot(np.arange(1., 0., -0.1) * 2000., np.arange(1., 0., -0.1))  # 绘制数据
ax.set_title('Title0 1')  # 设置子图标题
ax.xaxis.tick_top()  # x轴刻度显示在顶部
ax.tick_params(axis='x', rotation=55)  # 设置x轴刻度旋转角度

# 遍历下方两个子图
for i in range(2):
    ax = axs[1][i]
    ax.plot(np.arange(1., 0., -0.1) * 2000., np.arange(1., 0., -0.1))  # 绘制数据
    ax.set_ylabel('YLabel1 %d' % i)  # 设置y轴标签
    ax.set_xlabel('XLabel1 %d' % i)  # 设置x轴标签
    if i == 0:
        ax.tick_params(axis='x', rotation=55)  # 若为第一个子图，则设置x轴刻度旋转角度

fig.align_labels()  # 对所有子图进行标签对齐操作，相当于fig.align_xlabels(); fig.align_ylabels()
fig.align_titles()  # 对所有子图标题进行对齐操作

plt.show()  # 显示图形
```