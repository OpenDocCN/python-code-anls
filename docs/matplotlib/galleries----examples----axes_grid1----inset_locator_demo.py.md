# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\inset_locator_demo.py`

```
# %%
# 导入 matplotlib.pyplot 库，用于绘图操作
import matplotlib.pyplot as plt

# 从 mpl_toolkits.axes_grid1.inset_locator 中导入 inset_axes 函数
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 创建一个包含两个子图的图形窗口，每个子图的大小为 [5.5, 2.8] inches
fig, (ax, ax2) = plt.subplots(1, 2, figsize=[5.5, 2.8])

# 在第一个子图 ax 中创建一个宽度为 1.3 inches，高度为 0.9 inches 的插图，默认放置在右上角
axins = inset_axes(ax, width=1.3, height=0.9)

# 在第一个子图 ax 中创建一个宽度为父 Axes 边界框宽度的 30%，高度为父 Axes 边界框高度的 40% 的插图，放置在左下角 (loc=3)
axins2 = inset_axes(ax, width="30%", height="40%", loc=3)

# 在第二个子图 ax2 中创建一个宽度为父 Axes 边界框宽度的 30%，高度为 1 inch 的插图，放置在左上角 (loc=2)
axins3 = inset_axes(ax2, width="30%", height=1., loc=2)

# 在第二个子图 ax2 中创建一个宽度为父 Axes 边界框宽度的 20%，高度为父 Axes 边界框高度的 20% 的插图，放置在右下角 (loc=4)，设置 borderpad=1
axins4 = inset_axes(ax2, width="20%", height="20%", loc=4, borderpad=1)

# 关闭插图的刻度标签
for axi in [axins, axins2, axins3, axins4]:
    axi.tick_params(labelleft=False, labelbottom=False)

# 显示图形
plt.show()

# %%
# 创建一个新的图形窗口，大小为 [5.5, 2.8] inches
fig = plt.figure(figsize=[5.5, 2.8])

# 在图形窗口中添加一个子图 ax
ax = fig.add_subplot(121)

# 在 ax 子图中创建一个插图，其位置和大小通过 bbox_to_anchor 和 bbox_transform 参数进行精确控制
# bbox_to_anchor 参数在 axes 坐标系下定义边界框，bbox_transform 指定了坐标变换方式
# 这里的边界框范围为 (0.2, 0.4) 到 (0.8, 0.9)，在该边界框内创建一个宽度为边界框宽度的 50%，高度为边界框高度的 75% 的插图，左下角对齐边界框的左下角 (loc=3)
axins = inset_axes(ax, width="50%", height="75%",
                   bbox_to_anchor=(.2, .4, .6, .5),
                   bbox_transform=ax.transAxes, loc=3)

# 为了可视化，通过在 ax 子图中添加一个矩形来标记边界框
ax.add_patch(plt.Rectangle((.2, .4), .6, .5, ls="--", ec="c", fc="none",
                           transform=ax.transAxes))

# 设置 ax 子图的坐标轴限制，避免分散注意力于默认设置
ax.set(xlim=(0, 10), ylim=(0, 10))

# 显示图形
plt.show()
# 创建一个新的子图ax2，位置在2x2网格的第二个位置
ax2 = fig.add_subplot(222)
# 在ax2上创建一个插图，宽度为30%，高度为50%
axins2 = inset_axes(ax2, width="30%", height="50%")

# 创建一个新的子图ax3，位置在2x2网格的第四个位置
ax3 = fig.add_subplot(224)
# 在ax3上创建一个插图，宽度为100%，高度为100%，位置由bbox_to_anchor参数指定
# bbox_to_anchor=(.7, .5, .3, .5)表示相对于ax3的坐标系，左下角位置在(.7, .5)，宽度为.3，高度为.5
axins3 = inset_axes(ax3, width="100%", height="100%",
                    bbox_to_anchor=(.7, .5, .3, .5),
                    bbox_transform=ax3.transAxes)

# 在ax2上添加一个矩形补丁，用于可视化边界框
ax2.add_patch(plt.Rectangle((0, 0), 1, 1, ls="--", lw=2, ec="c", fc="none"))
# 在ax3上添加一个矩形补丁，用于可视化边界框
ax3.add_patch(plt.Rectangle((.7, .5), .3, .5, ls="--", lw=2,
                            ec="c", fc="none"))

# 关闭所有子图的刻度标签
for axi in [axins2, axins3, ax2, ax3]:
    axi.tick_params(labelleft=False, labelbottom=False)

# 显示图形
plt.show()

# %%
# 在上面的代码中，使用了Axes变换以及4元组边界框，主要是为了指定插图相对于其所属的Axes的位置。
# 然而，还有其他使用情况同样适用。下面的例子探讨了其中一些。

# 创建一个新的图形fig，设置大小为[5.5, 2.8]
fig = plt.figure(figsize=[5.5, 2.8])
# 在fig上添加一个子图ax，位置在1x3网格的第一个位置
ax = fig.add_subplot(131)

# 创建一个在Axes之外的插图，位置由bbox_to_anchor参数指定
axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(1.05, .6, .5, .4),
                   bbox_transform=ax.transAxes, loc=2, borderpad=0)
# 关闭插图的左侧刻度和标签，打开右侧刻度和标签
axins.tick_params(left=False, right=True, labelleft=False, labelright=True)

# 创建一个使用2元组边界框的插图。注意，这种情况下创建的边界框没有extent属性，
# 因此仅在以绝对单位（英寸）指定宽度和高度时才有意义。
axins2 = inset_axes(ax, width=0.5, height=0.4,
                    bbox_to_anchor=(0.33, 0.25),
                    bbox_transform=ax.transAxes, loc=3, borderpad=0)

# 在fig上添加一个子图ax2，位置在1x3网格的第三个位置
ax2 = fig.add_subplot(133)
# 设置ax2的x轴为对数刻度，设置x轴和y轴的范围
ax2.set_xscale("log")
ax2.set(xlim=(1e-6, 1e6), ylim=(-2, 6))

# 创建一个在数据坐标系中的插图，位置由bbox_to_anchor参数指定
axins3 = inset_axes(ax2, width="100%", height="100%",
                    bbox_to_anchor=(1e-2, 2, 1e3, 3),
                    bbox_transform=ax2.transData, loc=2, borderpad=0)

# 创建一个在图形坐标系中水平居中的插图，并且垂直与Axes对齐。
from matplotlib.transforms import blended_transform_factory  # noqa

# 创建混合变换，结合fig.transFigure和ax2.transAxes
transform = blended_transform_factory(fig.transFigure, ax2.transAxes)
axins4 = inset_axes(ax2, width="16%", height="34%",
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=transform, loc=8, borderpad=0)

# 显示图形
plt.show()
```