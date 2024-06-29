# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\barchart.py`

```py
"""
=============================
Grouped bar chart with labels
=============================

This example shows a how to create a grouped bar chart and how to annotate
bars with labels.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库用于绘图
import numpy as np  # 导入numpy库用于数值计算

# 定义物种名称和对应的属性平均值
species = ("Adelie", "Chinstrap", "Gentoo")
penguin_means = {
    'Bill Depth': (18.35, 18.43, 14.98),
    'Bill Length': (38.79, 48.83, 47.50),
    'Flipper Length': (189.95, 195.82, 217.19),
}

x = np.arange(len(species))  # 标签的位置，即x轴坐标
width = 0.25  # 条形图的宽度
multiplier = 0  # 初始化一个倍数，用于控制条形图的位置偏移

fig, ax = plt.subplots()  # 创建一个图形和一个轴

# 遍历物种属性和其对应的平均值
for attribute, measurement in penguin_means.items():
    offset = width * multiplier  # 计算每个属性的偏移量，以便将条形图分组显示
    rects = ax.bar(x + offset, measurement, width, label=attribute)  # 绘制条形图
    ax.bar_label(rects, padding=3)  # 添加标签到每个条形图上
    multiplier += 1  # 更新倍数，以便下一个属性使用不同的偏移量

# 添加一些文本标签，标题，自定义x轴刻度标签等
ax.set_ylabel('Length (mm)')  # 设置y轴标签
ax.set_title('Penguin attributes by species')  # 设置图表标题
ax.set_xticks(x + width / 2)  # 设置x轴刻度位置
ax.set_xticklabels(species)  # 设置x轴刻度标签为物种名称
ax.legend(loc='upper left', ncols=3)  # 添加图例，指定位置和列数
ax.set_ylim(0, 250)  # 设置y轴的范围

plt.show()  # 显示图表

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.bar_label` / `matplotlib.pyplot.bar_label`
```