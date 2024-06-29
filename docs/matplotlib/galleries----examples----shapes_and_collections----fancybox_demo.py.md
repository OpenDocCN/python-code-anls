# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\fancybox_demo.py`

```py
"""
===================
Drawing fancy boxes
===================

The following examples show how to plot boxes with different visual properties.
"""

import inspect  # 导入inspect模块，用于获取对象信息

import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块

import matplotlib.patches as mpatch  # 导入matplotlib的图形修饰模块
from matplotlib.patches import FancyBboxPatch  # 从matplotlib的图形修饰模块导入FancyBboxPatch类
import matplotlib.transforms as mtransforms  # 导入matplotlib的变换模块

# %%
# 首先展示一些带有fancybox的示例。

styles = mpatch.BoxStyle.get_styles()  # 获取所有可用的box样式
ncol = 2  # 列数
nrow = (len(styles) + 1) // ncol  # 行数
axs = (plt.figure(figsize=(3 * ncol, 1 + nrow))
       .add_gridspec(1 + nrow, ncol, wspace=.5).subplots())  # 创建子图网格
for ax in axs.flat:
    ax.set_axis_off()  # 关闭坐标轴
for ax in axs[0, :]:
    ax.text(.2, .5, "boxstyle",
            transform=ax.transAxes, size="large", color="tab:blue",
            horizontalalignment="right", verticalalignment="center")  # 在图中添加文本说明
    ax.text(.4, .5, "default parameters",
            transform=ax.transAxes,
            horizontalalignment="left", verticalalignment="center")  # 在图中添加文本说明
for ax, (stylename, stylecls) in zip(axs[1:, :].T.flat, styles.items()):
    ax.text(.2, .5, stylename, bbox=dict(boxstyle=stylename, fc="w", ec="k"),
            transform=ax.transAxes, size="large", color="tab:blue",
            horizontalalignment="right", verticalalignment="center")  # 在图中添加文本说明
    ax.text(.4, .5, str(inspect.signature(stylecls))[1:-1].replace(", ", "\n"),
            transform=ax.transAxes,
            horizontalalignment="left", verticalalignment="center")  # 在图中添加文本说明

# %%
# 接下来展示多个fancy box。

def add_fancy_patch_around(ax, bb, **kwargs):
    fancy = FancyBboxPatch(bb.p0, bb.width, bb.height,
                           fc=(1, 0.8, 1, 0.5), ec=(1, 0.5, 1, 0.5),
                           **kwargs)  # 创建一个FancyBboxPatch对象
    ax.add_patch(fancy)  # 将创建的FancyBboxPatch对象添加到子图中
    return fancy

def draw_control_points_for_patches(ax):
    for patch in ax.patches:
        patch.axes.plot(*patch.get_path().vertices.T, ".",
                        c=patch.get_edgecolor())  # 在指定的patches上绘制控制点

fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # 创建包含4个子图的figure

# 创建一个Bbox对象，围绕它将绘制fancy box。
bb = mtransforms.Bbox([[0.3, 0.4], [0.7, 0.6]])

ax = axs[0, 0]
# 一个带有圆角的fancy box，pad=0.1
fancy = add_fancy_patch_around(ax, bb, boxstyle="round,pad=0.1")
ax.set(xlim=(0, 1), ylim=(0, 1), aspect=1,
       title='boxstyle="round,pad=0.1"')

ax = axs[0, 1]
# bbox=round有两个可选参数：pad和rounding_size。
# 它们可以在初始化时设置。
fancy = add_fancy_patch_around(ax, bb, boxstyle="round,pad=0.1")
# boxstyle及其参数可以使用set_boxstyle()稍后修改。
# 注意，即使boxstyle名称相同，旧属性也会被简单地遗忘。
fancy.set_boxstyle("round,pad=0.1,rounding_size=0.2")
# 或者：fancy.set_boxstyle("round", pad=0.1, rounding_size=0.2)
ax.set(xlim=(0, 1), ylim=(0, 1), aspect=1,
       title='boxstyle="round,pad=0.1,rounding_size=0.2"')

ax = axs[1, 0]
# mutation_scale决定变异的整体比例，即pad和rounding_size根据此值缩放。
# 为指定的 Axes 对象添加一个带有特定样式的装饰框，并返回装饰框对象
fancy = add_fancy_patch_around(
    ax, bb, boxstyle="round,pad=0.1", mutation_scale=2)

# 设置 Axes 对象的 x 和 y 轴的界限，使其范围在 0 到 1 之间，并保持等比例，设置标题
ax.set(xlim=(0, 1), ylim=(0, 1), aspect=1,
       title='boxstyle="round,pad=0.1"\n mutation_scale=2')

# 获取第二行第二列的 Axes 对象
ax = axs[1, 1]

# 当 Axes 对象的纵横比不为1时，带有指定样式的装饰框可能不会如预期般显示（绿色）。
fancy = add_fancy_patch_around(ax, bb, boxstyle="round,pad=0.2")
# 设置装饰框的边框颜色为绿色，内部颜色为空
fancy.set(facecolor="none", edgecolor="green")

# 通过设置 mutation_aspect 可以弥补这一问题（粉色）。
fancy = add_fancy_patch_around(
    ax, bb, boxstyle="round,pad=0.3", mutation_aspect=0.5)
# 设置 Axes 对象的 x 和 y 轴的界限，使其范围在 -.5 到 1.5 之间，并保持纵横比为2，设置标题
ax.set(xlim=(-.5, 1.5), ylim=(0, 1), aspect=2,
       title='boxstyle="round,pad=0.3"\nmutation_aspect=.5')

# 遍历所有的 Axes 对象
for ax in axs.flat:
    # 在 Axes 对象上绘制控制点以显示装饰框
    draw_control_points_for_patches(ax)
    # 绘制原始的 bbox（使用 boxstyle=square 和 pad=0）
    fancy = add_fancy_patch_around(ax, bb, boxstyle="square,pad=0")
    # 设置装饰框的边框颜色为黑色，内部颜色为空，置于 zorder=10 层级
    fancy.set(edgecolor="black", facecolor="none", zorder=10)

# 调整图形布局以适应显示
fig.tight_layout()

# 显示图形
plt.show()
```