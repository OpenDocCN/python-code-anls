# `D:\src\scipysrc\matplotlib\galleries\examples\misc\table_demo.py`

```py
"""
==========
Table Demo
==========

Demo of table function to display a table within a plot.
"""
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

data = [[ 66386, 174296,  75131, 577908,  32015],  # 定义二维列表 data，存储数据
        [ 58230, 381139,  78045,  99308, 160454],
        [ 89135,  80552, 152558, 497981, 603535],
        [ 78415,  81858, 150656, 193263,  69638],
        [139361, 331509, 343164, 781380,  52269]]

columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')  # 定义元组 columns，存储列名
rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]  # 定义列表 rows，存储行名

values = np.arange(0, 2500, 500)  # 创建一个 numpy 数组，存储数值
value_increment = 1000  # 定义增量值

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))  # 生成颜色数组，用于图表颜色设置
n_rows = len(data)  # 获取数据的行数

index = np.arange(len(columns)) + 0.3  # 生成索引数组，用于条形图的位置设置
bar_width = 0.4  # 定义条形图的宽度

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))  # 创建全零数组，用于堆叠条形图的垂直偏移设置

# Plot bars and create text labels for the table
cell_text = []  # 创建空列表，用于存储表格的文本内容
for row in range(n_rows):  # 循环处理每一行数据
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])  # 绘制堆叠条形图
    y_offset = y_offset + data[row]  # 更新垂直偏移数组
    cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])  # 将格式化后的数据添加到表格文本列表中
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]  # 反转颜色数组
cell_text.reverse()  # 反转表格文本列表

# Add a table at the bottom of the Axes
the_table = plt.table(cellText=cell_text,  # 创建表格对象，设置表格文本内容
                      rowLabels=rows,  # 设置行标签
                      rowColours=colors,  # 设置行颜色
                      colLabels=columns,  # 设置列标签
                      loc='bottom')  # 将表格放置在图表底部

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)  # 调整子图布局，为表格腾出空间

plt.ylabel(f"Loss in ${value_increment}'s")  # 设置 y 轴标签
plt.yticks(values * value_increment, ['%d' % val for val in values])  # 设置 y 轴刻度
plt.xticks([])  # 隐藏 x 轴刻度
plt.title('Loss by Disaster')  # 设置图表标题

plt.show()  # 显示图表
```