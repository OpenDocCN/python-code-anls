# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\boxplot_color.py`

```
"""
=================================
Box plots with custom fill colors
=================================

To color each box of a box plot individually:

1) use the keyword argument ``patch_artist=True`` to create filled boxes.
2) loop through the created boxes and adapt their color.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np  # 导入numpy用于生成随机数据

# 设定随机种子以便结果可重复
np.random.seed(19680801)

# 生成三组随机数据，分别代表不同水果的重量
fruit_weights = [
    np.random.normal(130, 10, size=100),  # 桃子的重量数据
    np.random.normal(125, 20, size=100),  # 橙子的重量数据
    np.random.normal(120, 30, size=100),  # 西红柿的重量数据
]

# 每组数据对应的标签
labels = ['peaches', 'oranges', 'tomatoes']

# 每个箱子填充的颜色
colors = ['peachpuff', 'orange', 'tomato']

# 创建图表和坐标系
fig, ax = plt.subplots()
ax.set_ylabel('fruit weight (g)')  # 设置y轴标签为水果重量（单位：克）

# 创建箱线图，并使用颜色填充
bplot = ax.boxplot(fruit_weights,
                   patch_artist=True,  # 使用填充颜色的形式显示箱子
                   tick_labels=labels)  # 使用labels作为x轴刻度标签

# 循环遍历每个箱子，并设置其填充颜色
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.boxplot` / `matplotlib.pyplot.boxplot`
```