# `D:\src\scipysrc\matplotlib\galleries\tutorials\lifecycle.py`

```py
# 导入 matplotlib.pyplot 库，并使用 plt 别名
import matplotlib.pyplot as plt
# 导入 numpy 库，并使用 np 别名
import numpy as np

# 定义包含公司销售数据的字典
data = {'Barton LLC': 109438.50,
        'Frami, Hills and Schmidt': 103569.59,
        'Fritsch, Russel and Anderson': 112214.71,
        'Jerde-Hilpert': 112591.43,
        'Keeling LLC': 100934.30,
        'Koepp Ltd': 103660.54,
        'Kulas Inc': 137351.96,
        'Trantow-Barrows': 123381.38,
        'White-Trantow': 135841.99,
        'Will LLC': 104437.60}

# 提取数据字典中的值作为列表
group_data = list(data.values())
# 提取数据字典中的键作为列表
group_names = list(data.keys())
# 计算销售数据的平均值
group_mean = np.mean(group_data)

# %%
# Getting started
# ===============
#
# This data is naturally visualized as a barplot, with one bar per
# group. To do this with the object-oriented approach, we first generate
# an instance of :class:`figure.Figure` and
# :class:`axes.Axes`. The Figure is like a canvas, and the Axes
# is a part of that canvas on which we will make a particular visualization.
#
# .. note::
#    该数据自然地可视化为条形图，每个条形代表一个组。为了使用面向对象的方法实现这一点，
#    我们首先生成一个 :class:`figure.Figure` 实例和一个 :class:`axes.Axes` 实例。
#    Figure 类似于画布，Axes 是画布上的一部分，我们将在其上制作特定的可视化效果。
# %%
# 创建一个新的图形对象和一个 Axes 对象，这是 Matplotlib 中创建图表的标准方法。
fig, ax = plt.subplots()

# %%
# 现在我们有了一个 Axes 实例，可以在其上绘制图表。
fig, ax = plt.subplots()
ax.barh(group_names, group_data)

# %%
# 控制样式
# =====================
#
# Matplotlib 提供了许多样式，以便根据需要自定义可视化效果。可以使用 :mod:`.style` 查看所有可用样式。
print(plt.style.available)

# %%
# 可以使用以下方法激活一个样式：
plt.style.use('fivethirtyeight')

# %%
# 现在重新绘制上面的图表，看看效果如何：
fig, ax = plt.subplots()
ax.barh(group_names, group_data)

# %%
# 样式控制许多方面，如颜色、线宽、背景等。
#
# 自定义图表
# ====================
#
# 现在我们得到了一个大致符合要求的图表，让我们微调它，使其准备好进行打印。首先让我们旋转 x 轴上的标签，以便更清晰地显示出来。可以使用 :meth:`axes.Axes.get_xticklabels` 方法获取这些标签：
fig, ax = plt.subplots()
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()

# %%
# 如果想要同时设置多个元素的属性，可以使用 :func:`pyplot.setp` 函数。这将接受一个 Matplotlib 对象的列表（或多个列表），并尝试设置每个对象的某些样式元素。
fig, ax = plt.subplots()
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')

# %%
# 看起来这样做会截断底部的一些标签。可以告诉 Matplotlib 自动为创建的图形元素腾出空间。可以通过设置 rcParams 的 ``autolayout`` 值来实现这一点。有关使用 rcParams 控制样式、布局和其他图表特性的详细信息，请参阅 :ref:`customizing`。
plt.rcParams.update({'figure.autolayout': True})

fig, ax = plt.subplots()
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')

# %%
# 接下来，我们为图表添加标签。可以使用 OO 接口的 :meth:`.Artist.set` 方法来设置这个 Axes 对象的属性。
fig, ax = plt.subplots()
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company',
       title='Company Revenue')

# %%
# 还可以使用 :func:`pyplot.subplots` 函数调整图表的大小。可以使用 *figsize* 关键字参数来完成这一点。
#
# .. note::
#    在 NumPy 中的索引形式是 (行, 列)，但 *figsize* 关键字参数的形式是 (宽度, 高度)。这是遵循可视化的传统，不同于数组索引的顺序。
# 创建一个新的图形对象和坐标系对象，设置图形大小为8x4英寸
fig, ax = plt.subplots(figsize=(8, 4))

# 在坐标系上绘制水平条形图，数据为group_data，条形的标签为group_names
ax.barh(group_names, group_data)

# 获取当前坐标系的X轴刻度标签
labels = ax.get_xticklabels()

# 设置X轴刻度标签的旋转角度为45度，水平对齐方式为右对齐
plt.setp(labels, rotation=45, horizontalalignment='right')

# 设置X轴的范围为[-10000, 140000]，X轴标签为'Total Revenue'，Y轴标签为'Company'，图形标题为'Company Revenue'
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company',
       title='Company Revenue')

# %%
# 对于标签，我们可以定义自定义的格式化指南，使用函数的形式。下面我们定义一个函数，接受一个整数作为输入，返回一个字符串作为输出。
# 当与 `.Axis.set_major_formatter` 或 `.Axis.set_minor_formatter` 结合使用时，它们会自动创建和使用 `ticker.FuncFormatter` 类。
#
# 在这个函数中，参数 `x` 是原始刻度标签的值，`pos` 是刻度位置。我们这里只使用 `x`，但两个参数都是需要的。

def currency(x, pos):
    """The two arguments are the value and tick position"""
    # 如果 x 大于等于1百万，格式化为百万美元；否则格式化为千美元
    if x >= 1e6:
        s = f'${x*1e-6:1.1f}M'
    else:
        s = f'${x*1e-3:1.0f}K'
    return s

# %%
# 然后，我们可以将这个函数应用到图形的标签上。为了实现这一点，
# 我们使用图形坐标系的 `xaxis` 属性。这让你可以在图形的特定轴上执行操作。

# 创建一个新的图形对象和坐标系对象，设置图形大小为6x8英寸
fig, ax = plt.subplots(figsize=(6, 8))

# 在坐标系上绘制水平条形图，数据为group_data，条形的标签为group_names
ax.barh(group_names, group_data)

# 获取当前坐标系的X轴刻度标签
labels = ax.get_xticklabels()

# 设置X轴刻度标签的旋转角度为45度，水平对齐方式为右对齐
plt.setp(labels, rotation=45, horizontalalignment='right')

# 设置X轴的范围为[-10000, 140000]，X轴标签为'Total Revenue'，Y轴标签为'Company'，图形标题为'Company Revenue'
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company',
       title='Company Revenue')

# 将X轴的主要刻度标签格式化为货币单位
ax.xaxis.set_major_formatter(currency)

# %%
# 结合多个可视化元素
# ===================
#
# 可以在同一个 :class:`axes.Axes` 实例上绘制多个图形元素。只需在该Axes对象上调用另一个绘图方法即可。

# 创建一个新的图形对象和坐标系对象，设置图形大小为8x8英寸
fig, ax = plt.subplots(figsize=(8, 8))

# 在坐标系上绘制水平条形图，数据为group_data，条形的标签为group_names
ax.barh(group_names, group_data)

# 获取当前坐标系的X轴刻度标签
labels = ax.get_xticklabels()

# 设置X轴刻度标签的旋转角度为45度，水平对齐方式为右对齐
plt.setp(labels, rotation=45, horizontalalignment='right')

# 添加垂直线，这里我们在函数调用中设置了样式
ax.axvline(group_mean, ls='--', color='r')

# 注释新公司
for group in [3, 5, 8]:
    ax.text(145000, group, "New Company", fontsize=10,
            verticalalignment="center")

# 现在我们将标题向上移动一些，因为它有点拥挤
ax.title.set(y=1.05)

# 设置X轴的范围为[-10000, 140000]，X轴标签为'Total Revenue'，Y轴标签为'Company'，图形标题为'Company Revenue'
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company',
       title='Company Revenue')

# 将X轴的主要刻度标签格式化为货币单位
ax.xaxis.set_major_formatter(currency)

# 设置X轴刻度为特定的刻度位置
ax.set_xticks([0, 25e3, 50e3, 75e3, 100e3, 125e3])

# 调整子图的右边距
fig.subplots_adjust(right=.1)

# 显示图形
plt.show()

# %%
# 保存我们的图形
# ===============
#
# 现在我们对图形的结果满意后，我们想要将其保存到磁盘上。在Matplotlib中有很多文件格式可以保存。要查看可用选项的列表，使用:

print(fig.canvas.get_supported_filetypes())

# %%
# 然后，我们可以使用 :meth:`figure.Figure.savefig` 方法将图形保存到磁盘。注意下面我们展示了几个有用的标志:
#
# * ``transparent=True`` 使得保存的图形背景透明
# 如果格式支持的话，可以取消注释以下行来保存图像。
# `dpi=80` 控制输出的分辨率（每英寸点数）。
# `bbox_inches="tight"` 调整图形边界以适应我们的绘图。

# 取消注释以下行来保存图像。
# fig.savefig('sales.png', transparent=False, dpi=80, bbox_inches="tight")
```