# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\gridspec_multicolumn.py`

```
"""
=======================================================
Using Gridspec to make multi-column/row subplot layouts
=======================================================

`.GridSpec` is a flexible way to layout
subplot grids.  Here is an example with a 3x3 grid, and
axes spanning all three columns, two columns, and two rows.

"""
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块

from matplotlib.gridspec import GridSpec  # 从matplotlib的gridspec模块导入GridSpec类


def format_axes(fig):
    # 对图形fig中的每个子图ax进行格式化处理
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")  # 在子图中心添加文本，显示子图编号
        ax.tick_params(labelbottom=False, labelleft=False)  # 设置子图的刻度标签不显示


fig = plt.figure(layout="constrained")  # 创建一个带有约束布局的图形对象

gs = GridSpec(3, 3, figure=fig)  # 创建一个3x3的GridSpec对象，并将其绑定到指定的图形fig上
ax1 = fig.add_subplot(gs[0, :])  # 在GridSpec中第一行的所有列创建子图ax1
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax2 = fig.add_subplot(gs[1, :-1])  # 在GridSpec中第二行除最后一列以外的列创建子图ax2
ax3 = fig.add_subplot(gs[1:, -1])  # 在GridSpec中第二行及其后所有行的最后一列创建子图ax3
ax4 = fig.add_subplot(gs[-1, 0])   # 在GridSpec中最后一行的第一列创建子图ax4
ax5 = fig.add_subplot(gs[-1, -2])  # 在GridSpec中最后一行的倒数第二列创建子图ax5

fig.suptitle("GridSpec")  # 设置整个图形的标题为"GridSpec"
format_axes(fig)  # 调用函数对图形中所有子图进行格式化处理

plt.show()  # 显示图形
```