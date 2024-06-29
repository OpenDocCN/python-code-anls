# `D:\src\scipysrc\matplotlib\galleries\examples\pie_and_polar_charts\pie_features.py`

```py
# %%
# Label slices
# ------------
#
# Plot a pie chart of animals and label the slices. To add
# labels, pass a list of labels to the *labels* parameter
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'  # 定义标签列表
sizes = [15, 30, 45, 10]  # 定义每个部分的大小

fig, ax = plt.subplots()  # 创建一个图形和一个子图对象
ax.pie(sizes, labels=labels)  # 绘制饼图，带有指定的标签

# %%
# Each slice of the pie chart is a `.patches.Wedge` object; therefore in
# addition to the customizations shown here, each wedge can be customized using
# the *wedgeprops* argument, as demonstrated in
# :doc:`/gallery/pie_and_polar_charts/nested_pie`.
#
# Auto-label slices
# -----------------
#
# Pass a function or format string to *autopct* to label slices.
fig, ax = plt.subplots()  # 创建一个新的图形和子图对象
ax.pie(sizes, labels=labels, autopct='%1.1f%%')  # 绘制带有自动标签的饼图

# %%
# By default, the label values are obtained from the percent size of the slice.
#
# Color slices
# ------------
#
# Pass a list of colors to *colors* to set the color of each slice.
fig, ax = plt.subplots()  # 创建一个新的图形和子图对象
ax.pie(sizes, labels=labels,
       colors=['olivedrab', 'rosybrown', 'gray', 'saddlebrown'])  # 绘制带有自定义颜色的饼图

# %%
# Hatch slices
# ------------
#
# Pass a list of hatch patterns to *hatch* to set the pattern of each slice.
fig, ax = plt.subplots()  # 创建一个新的图形和子图对象
ax.pie(sizes, labels=labels, hatch=['**O', 'oO', 'O.O', '.||.'])  # 绘制带有指定刻度样式的饼图

# %%
# Swap label and autopct text positions
# -------------------------------------
# Use the *labeldistance* and *pctdistance* parameters to position the *labels*
# and *autopct* text respectively.
fig, ax = plt.subplots()  # 创建一个新的图形和子图对象
ax.pie(sizes, labels=labels, autopct='%1.1f%%',
       pctdistance=1.25, labeldistance=.6)  # 设置标签和百分比文本的位置参数

# %%
# *labeldistance* and *pctdistance* are ratios of the radius; therefore they
# vary between ``0`` for the center of the pie and ``1`` for the edge of the
# pie, and can be set to greater than ``1`` to place text outside the pie.
#
# Explode, shade, and rotate slices
# ---------------------------------
#
# In addition to the basic pie chart, this demo shows a few optional features:
#
# * offsetting a slice using *explode*
# * add a drop-shadow using *shadow*
# * custom start angle using *startangle*
#
# This example orders the slices, separates (explodes) them, and rotates them.
explode = (0, 0.1, 0, 0)  # 只“爆炸”第二个切片（即 'Hogs'）

fig, ax = plt.subplots()  # 创建一个新的图形和子图对象
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)  # 绘制饼图，带有偏移、阴影和起始角度设置
plt.show()
# 创建一个新的图形（Figure）和一个新的轴（Axes），用于绘制饼图
fig, ax = plt.subplots()

# 在轴上绘制饼图，使用给定的数据 sizes 和标签 labels，自动计算百分比显示在每个扇区上
# 设置自动百分比格式为整数形式，设置文本属性字典中的文本大小为较小
# 设置饼图的半径为 0.5
ax.pie(sizes, labels=labels, autopct='%.0f%%',
       textprops={'size': 'smaller'}, radius=0.5)

# 显示绘制的图形
plt.show()

# %%
# 修改阴影效果
# --------------------
#
# *shadow* 参数可以选择性地接受一个字典，包含传递给 `.Shadow` 补丁的参数。
# 这可以用来修改默认的阴影效果。

# 创建一个新的图形（Figure）和一个新的轴（Axes），用于绘制饼图
fig, ax = plt.subplots()

# 在轴上绘制饼图，使用给定的数据 sizes、标签 labels 和分离 explode 数据
# 设置自动百分比格式为一位小数形式，设置起始角度为 90 度
# 设置阴影参数为一个字典，调整阴影的偏移、边缘颜色和透明度
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9}, startangle=90)

# 显示绘制的图形
plt.show()

# %%
# .. admonition:: References
#
#    本示例展示了以下函数、方法、类和模块的使用：
#
#    - `matplotlib.axes.Axes.pie` / `matplotlib.pyplot.pie`
```