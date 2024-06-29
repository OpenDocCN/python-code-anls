# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\bar_colors.py`

```
"""
==============
Bar color demo
==============

This is an example showing how to control bar color and legend entries
using the *color* and *label* parameters of `~matplotlib.pyplot.bar`.
Note that labels with a preceding underscore won't show up in the legend.
"""

# 导入 matplotlib 的 pyplot 模块，并简称为 plt
import matplotlib.pyplot as plt

# 创建一个包含图形和轴的图形对象和轴对象
fig, ax = plt.subplots()

# 定义水果种类和数量
fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]

# 定义每个柱形图的标签，用于显示在图例中
bar_labels = ['red', 'blue', '_red', 'orange']

# 定义每个柱形图的颜色，使用了 'tab:' 前缀的颜色字符串
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

# 在轴上绘制柱形图，使用水果名称作为 x 轴，数量作为柱高，标签和颜色作为参数
ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

# 设置 y 轴的标签
ax.set_ylabel('fruit supply')

# 设置图表的标题
ax.set_title('Fruit supply by kind and color')

# 添加图例，指定图例的标题
ax.legend(title='Fruit color')

# 显示图形
plt.show()
```