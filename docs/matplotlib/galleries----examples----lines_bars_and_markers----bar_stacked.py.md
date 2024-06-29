# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\bar_stacked.py`

```py
"""
=================
Stacked bar chart
=================

This is an example of creating a stacked bar plot
using `~matplotlib.pyplot.bar`.
"""

# 导入matplotlib.pyplot模块作为plt，用于绘图操作
import matplotlib.pyplot as plt
# 导入numpy模块作为np，用于科学计算和数组操作
import numpy as np

# 定义鸟类物种及其平均体重数据，包含特殊格式化字符串
species = (
    "Adelie\n $\\mu=$3700.66g",
    "Chinstrap\n $\\mu=$3733.09g",
    "Gentoo\n $\\mu=5076.02g$",
)

# 定义体重分类及其对应的数量数据，使用numpy数组
weight_counts = {
    "Below": np.array([70, 31, 58]),
    "Above": np.array([82, 37, 66]),
}

# 设置每个条形图的宽度
width = 0.5

# 创建一个新的图形和坐标轴对象
fig, ax = plt.subplots()

# 初始化堆叠条形图的底部位置数组为全零，长度为3
bottom = np.zeros(3)

# 遍历体重分类字典中的每一项
for boolean, weight_count in weight_counts.items():
    # 绘制堆叠的条形图，并更新底部位置数组
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

# 设置图表的标题
ax.set_title("Number of penguins with above average body mass")

# 在图表中添加图例，并指定其位置为右上角
ax.legend(loc="upper right")

# 显示绘制的图表
plt.show()
```