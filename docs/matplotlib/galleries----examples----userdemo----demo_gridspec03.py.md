# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\demo_gridspec03.py`

```
"""
=============
GridSpec demo
=============

This example demonstrates the use of `.GridSpec` to generate subplots,
the control of the relative sizes of subplots with *width_ratios* and
*height_ratios*, and the control of the spacing around and between subplots
using subplot params (*left*, *right*, *bottom*, *top*, *wspace*, and
*hspace*).
"""

# 导入 matplotlib 的 pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt
# 导入 GridSpec 类，用于创建自定义的 subplot 布局
from matplotlib.gridspec import GridSpec


# 定义函数 annotate_axes，用于在每个 subplot 上标注其编号
def annotate_axes(fig):
    # 遍历图形对象的所有子图并标注编号
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        # 设置不显示坐标轴标签
        ax.tick_params(labelbottom=False, labelleft=False)


# 创建一个新的图形对象
fig = plt.figure()
# 设置整个图形的标题
fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

# 使用 GridSpec 创建一个2x2的子图布局，设置宽度和高度的比例
gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[4, 1])
# 在图形对象中添加四个子图，位置由 GridSpec 控制
ax1 = fig.add_subplot(gs[0])   # 第一个子图
ax2 = fig.add_subplot(gs[1])   # 第二个子图
ax3 = fig.add_subplot(gs[2])   # 第三个子图
ax4 = fig.add_subplot(gs[3])   # 第四个子图

# 在图形对象上标注子图编号
annotate_axes(fig)

# %%

# 创建第二个示例图形对象
fig = plt.figure()
# 设置整个图形的标题
fig.suptitle("Controlling spacing around and between subplots")

# 使用 GridSpec 创建一个3x3的子图布局，设置左右间距和子图之间的水平间距
gs1 = GridSpec(3, 3, left=0.05, right=0.48, wspace=0.05)
# 在第一个图形对象中添加三个子图，位置由 GridSpec 控制
ax1 = fig.add_subplot(gs1[:-1, :])   # 第一个子图
ax2 = fig.add_subplot(gs1[-1, :-1])  # 第二个子图
ax3 = fig.add_subplot(gs1[-1, -1])   # 第三个子图

# 使用 GridSpec 创建另一个3x3的子图布局，设置左右间距和子图之间的垂直间距
gs2 = GridSpec(3, 3, left=0.55, right=0.98, hspace=0.05)
# 在第二个图形对象中添加三个子图，位置由 GridSpec 控制
ax4 = fig.add_subplot(gs2[:, :-1])   # 第四个子图
ax5 = fig.add_subplot(gs2[:-1, -1])  # 第五个子图
ax6 = fig.add_subplot(gs2[-1, -1])   # 第六个子图

# 在图形对象上标注子图编号
annotate_axes(fig)

# 显示图形对象
plt.show()
```