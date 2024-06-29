# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\gridspec_nested.py`

```py
"""
================
Nested Gridspecs
================

GridSpecs can be nested, so that a subplot from a parent GridSpec can
set the position for a nested grid of subplots.

Note that the same functionality can be achieved more directly with
`~.Figure.subfigures`; see
:doc:`/gallery/subplots_axes_and_figures/subfigures`.

"""
# 导入 matplotlib 的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt

# 导入 matplotlib 的 gridspec 模块，用于管理子图的位置和布局
import matplotlib.gridspec as gridspec


# 定义函数 format_axes，用于设置子图的文本和轴参数
def format_axes(fig):
    # 遍历图形 fig 中的所有子图 ax
    for i, ax in enumerate(fig.axes):
        # 在每个子图中心添加文本，标识子图编号
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        # 设置子图的轴参数，隐藏标签
        ax.tick_params(labelbottom=False, labelleft=False)


# 创建图形对象 fig
fig = plt.figure()

# 创建一个顶级 GridSpec，分为 1 行 2 列，将其与图形对象 fig 关联
gs0 = gridspec.GridSpec(1, 2, figure=fig)

# 在顶级 GridSpec 中的第一个位置创建一个 GridSpecFromSubplotSpec 子网格，3 行 3 列
gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

# 向图形 fig 中添加子图，分别放置在 gs00 的不同位置
ax1 = fig.add_subplot(gs00[:-1, :])    # 占据除最后一行外的所有行，所有列
ax2 = fig.add_subplot(gs00[-1, :-1])    # 占据最后一行除最后一列外的所有列
ax3 = fig.add_subplot(gs00[-1, -1])     # 占据 gs00 中的最后一个位置

# 使用与上述 GridSpecFromSubplotSpec 相同的语法创建一个子网格 gs01
gs01 = gs0[1].subgridspec(3, 3)

# 向图形 fig 中添加子图，放置在 gs01 的不同位置
ax4 = fig.add_subplot(gs01[:, :-1])     # 占据所有行，除最后一列外的所有列
ax5 = fig.add_subplot(gs01[:-1, -1])    # 占据除最后一行外的所有行，最后一列
ax6 = fig.add_subplot(gs01[-1, -1])     # 占据 gs01 中的最后一个位置

# 添加图形的总标题
plt.suptitle("GridSpec Inside GridSpec")

# 调用 format_axes 函数，设置所有子图的文本和轴参数
format_axes(fig)

# 显示图形
plt.show()
```