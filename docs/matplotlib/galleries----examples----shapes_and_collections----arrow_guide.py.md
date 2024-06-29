# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\arrow_guide.py`

```
# 导入matplotlib库中的pyplot模块，并用plt作为别名
import matplotlib.pyplot as plt

# 导入matplotlib库中的patches模块，并用mpatches作为别名
import matplotlib.patches as mpatches

# 设置箭头起始点和终止点的坐标
x_tail = 0.1
y_tail = 0.5
x_head = 0.9
y_head = 0.8

# 计算箭头的水平和垂直方向的变化量
dx = x_head - x_tail
dy = y_head - y_tail

# %%
# 头部形状固定在显示空间，锚点固定在数据空间
# -------------------------------------------------------
#
# 如果你正在注释一个图，不希望箭头在平移或缩放图时改变形状或位置，这种情况非常有用。
#
# 在这种情况下，我们使用`.patches.FancyArrowPatch`。
#
# 注意，当轴限制改变时，箭头的形状保持不变，但锚点会移动。

# 创建两个子图
fig, axs = plt.subplots(nrows=2)

# 在第一个子图上添加一个FancyArrowPatch对象，指定起始点和终止点，并设置变异比例为100
arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                 mutation_scale=100)
axs[0].add_patch(arrow)

# 在第二个子图上添加一个FancyArrowPatch对象，指定起始点和终止点，并设置变异比例为100
arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                 mutation_scale=100)
axs[1].add_patch(arrow)

# 设置第二个子图的坐标轴限制
axs[1].set(xlim=(0, 2), ylim=(0, 2))

# %%
# 头部形状和锚点固定在显示空间
# ---------------------------------------------------
#
# 如果你正在注释一个图，不希望箭头在平移或缩放图时改变形状或位置，这种情况非常有用。
#
# 在这种情况下，我们使用`.patches.FancyArrowPatch`，并传递关键字参数
# ``transform=ax.transAxes``，其中``ax``是我们要添加补丁的Axes对象。
#
# 注意，当轴限制改变时，箭头的形状和位置保持不变。

# 创建两个子图
fig, axs = plt.subplots(nrows=2)

# 在第一个子图上添加一个FancyArrowPatch对象，指定起始点和终止点，并设置变异比例为100，
# 并且将变换参数设置为axs[0].transAxes，以固定箭头的形状和位置在显示空间
arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                 mutation_scale=100,
                                 transform=axs[0].transAxes)
axs[0].add_patch(arrow)
# 创建一个 FancyArrowPatch 对象，用于在图表 axs[1] 上绘制箭头，箭头从 (x_tail, y_tail) 到 (x_head, y_head)
arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                 mutation_scale=100,
                                 transform=axs[1].transAxes)
# 将箭头对象添加到 axs[1] 的图形中
axs[1].add_patch(arrow)
# 设置 axs[1] 的 x 和 y 轴限制
axs[1].set(xlim=(0, 2), ylim=(0, 2))


# %%
# 头部形状和锚点在数据空间中固定
# ------------------------------------------------
#
# 在这种情况下，我们使用 `.patches.Arrow` 或 `.patches.FancyArrow`（后者是橙色的）。
#
# 注意，当轴限制改变时，箭头的形状和位置也会改变。
#
# `.FancyArrow` 的 API 相对复杂，特别是需要传递 `length_includes_head=True`，
# 以便箭头的尖端是距离箭头起点 `(dx, dy)` 的位置。它仅在此参考中包含，
# 因为它是由 `.Axes.arrow` 返回的箭头类（绿色）。

fig, axs = plt.subplots(nrows=2)

# 在 axs[0] 上创建一个简单箭头对象，从 (x_tail, y_tail) 开始，朝 (dx, dy) 方向
arrow = mpatches.Arrow(x_tail, y_tail, dx, dy)
axs[0].add_patch(arrow)
# 在 axs[0] 上创建一个 FancyArrow 对象，带有指定的宽度和颜色，长度包括箭头头部
arrow = mpatches.FancyArrow(x_tail, y_tail - .4, dx, dy,
                            width=.1, length_includes_head=True, color="C1")
axs[0].add_patch(arrow)
# 使用 axs[0].arrow 方法创建箭头，指定位置、方向、宽度和颜色
axs[0].arrow(x_tail + 1, y_tail - .4, dx, dy,
             width=.1, length_includes_head=True, color="C2")

# 在 axs[1] 上创建一个简单箭头对象
arrow = mpatches.Arrow(x_tail, y_tail, dx, dy)
axs[1].add_patch(arrow)
# 在 axs[1] 上创建一个 FancyArrow 对象，带有指定的宽度和颜色，长度包括箭头头部
arrow = mpatches.FancyArrow(x_tail, y_tail - .4, dx, dy,
                            width=.1, length_includes_head=True, color="C1")
axs[1].add_patch(arrow)
# 使用 axs[1].arrow 方法创建箭头，指定位置、方向、宽度和颜色
axs[1].arrow(x_tail + 1, y_tail - .4, dx, dy,
             width=.1, length_includes_head=True, color="C2")
# 设置 axs[1] 的 x 和 y 轴限制
axs[1].set(xlim=(0, 2), ylim=(0, 2))

# %%

# 显示绘制的图形
plt.show()
```