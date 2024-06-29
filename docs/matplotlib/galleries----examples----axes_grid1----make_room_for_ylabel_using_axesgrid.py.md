# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\make_room_for_ylabel_using_axesgrid.py`

```py
"""
====================================
Make room for ylabel using axes_grid
====================================
"""

# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt

# 从mpl_toolkits.axes_grid1模块中导入make_axes_locatable和make_axes_area_auto_adjustable函数
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

# 创建一个新的Figure对象
fig = plt.figure()
# 在Figure对象中添加一个Axes对象，位置从(0,0)到(1,1)
ax = fig.add_axes([0, 0, 1, 1])

# 设置y轴刻度为[0.5]，并设置标签为"very long label"
ax.set_yticks([0.5], labels=["very long label"])

# 使用make_axes_area_auto_adjustable函数自动调整Axes对象的区域
make_axes_area_auto_adjustable(ax)

# %%

# 创建另一个新的Figure对象
fig = plt.figure()
# 在Figure对象中添加两个Axes对象，位置分别为(0,0,1,0.5)和(0,0.5,1,0.5)
ax1 = fig.add_axes([0, 0, 1, 0.5])
ax2 = fig.add_axes([0, 0.5, 1, 0.5])

# 设置ax1的y轴刻度为[0.5]，并设置标签为"very long label"
ax1.set_yticks([0.5], labels=["very long label"])
# 设置ax1的y轴标签为"Y label"
ax1.set_ylabel("Y label")

# 设置ax2的标题为"Title"
ax2.set_title("Title")

# 对ax1和ax2使用make_axes_area_auto_adjustable函数自动调整区域，设置pad为0.1
make_axes_area_auto_adjustable(ax1, pad=0.1, use_axes=[ax1, ax2])
make_axes_area_auto_adjustable(ax2, pad=0.1, use_axes=[ax1, ax2])

# %%

# 创建另一个新的Figure对象
fig = plt.figure()
# 在Figure对象中添加一个Axes对象，位置从(0,0)到(1,1)
ax1 = fig.add_axes([0, 0, 1, 1])
# 使用make_axes_locatable函数创建一个分隔对象divider
divider = make_axes_locatable(ax1)

# 在divider中添加一个新的Axes对象，放置在右侧，宽度为"100%"，pad为0.3，与ax1共享y轴
ax2 = divider.append_axes("right", "100%", pad=0.3, sharey=ax1)
# 取消ax2的左侧标签显示
ax2.tick_params(labelleft=False)
# 将ax2添加到Figure对象中

divider.add_auto_adjustable_area(use_axes=[ax1], pad=0.1,
                                 adjust_dirs=["left"])
divider.add_auto_adjustable_area(use_axes=[ax2], pad=0.1,
                                 adjust_dirs=["right"])
divider.add_auto_adjustable_area(use_axes=[ax1, ax2], pad=0.1,
                                 adjust_dirs=["top", "bottom"])

# 设置ax1的y轴刻度为[0.5]，并设置标签为"very long label"
ax1.set_yticks([0.5], labels=["very long label"])

# 设置ax2的标题为"Title"，设置x轴标签为"X - Label"
ax2.set_title("Title")
ax2.set_xlabel("X - Label")

# 显示图形
plt.show()
```