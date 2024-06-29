# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\demo_constrained_layout.py`

```py
"""
=====================================
Resizing Axes with constrained layout
=====================================

*Constrained layout* attempts to resize subplots in
a figure so that there are no overlaps between Axes objects and labels
on the Axes.

See :ref:`constrainedlayout_guide` for more details and
:ref:`tight_layout_guide` for an alternative.

"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块


def example_plot(ax):
    ax.plot([1, 2])  # 绘制简单的线图，x轴为[1, 2]
    ax.set_xlabel('x-label', fontsize=12)  # 设置x轴标签为'x-label'，字体大小为12
    ax.set_ylabel('y-label', fontsize=12)  # 设置y轴标签为'y-label'，字体大小为12
    ax.set_title('Title', fontsize=14)  # 设置图表标题为'Title'，字体大小为14


# %%
# If we don't use *constrained layout*, then labels overlap the Axes
# 如果不使用*constrained layout*，标签可能会重叠在图表上

fig, axs = plt.subplots(nrows=2, ncols=2, layout=None)  # 创建2x2子图的图表对象

for ax in axs.flat:  # 遍历所有子图对象
    example_plot(ax)  # 在每个子图对象上绘制示例图

# %%
# adding ``layout='constrained'`` automatically adjusts.
# 添加 ``layout='constrained'`` 可以自动调整子图布局

fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')  # 创建2x2子图的图表对象，并使用constrained布局

for ax in axs.flat:  # 遍历所有子图对象
    example_plot(ax)  # 在每个子图对象上绘制示例图

# %%
# Below is a more complicated example using nested gridspecs.
# 下面是使用嵌套GridSpec的更复杂示例

fig = plt.figure(layout='constrained')  # 创建使用constrained布局的图表对象

import matplotlib.gridspec as gridspec  # 导入matplotlib.gridspec模块

gs0 = gridspec.GridSpec(1, 2, figure=fig)  # 创建包含两个列的GridSpec对象

gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0])  # 创建第一个列的GridSpecFromSubplotSpec对象
for n in range(3):  # 遍历3次
    ax = fig.add_subplot(gs1[n])  # 在第一个列的子图中添加一个子图对象
    example_plot(ax)  # 在子图对象上绘制示例图

gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1])  # 创建第二个列的GridSpecFromSubplotSpec对象
for n in range(2):  # 遍历2次
    ax = fig.add_subplot(gs2[n])  # 在第二个列的子图中添加一个子图对象
    example_plot(ax)  # 在子图对象上绘制示例图

plt.show()  # 显示图表

# %%
#
# .. admonition:: References
#    
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.gridspec.GridSpec`
#    - `matplotlib.gridspec.GridSpecFromSubplotSpec`
```