# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\simple_legend02.py`

```
"""
===============
Simple Legend02
===============

"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 创建一个包含单个图形和轴对象的图形窗口
fig, ax = plt.subplots()

# 绘制第一条线，指定标签和线条样式
line1, = ax.plot([1, 2, 3], label="Line 1", linestyle='--')

# 绘制第二条线，指定标签和线条粗细
line2, = ax.plot([3, 2, 1], label="Line 2", linewidth=4)

# 创建第一个图例，包含第一条线，放置在右上角
first_legend = ax.legend(handles=[line1], loc='upper right')

# 将第一个图例手动添加到当前图形轴对象中
ax.add_artist(first_legend)

# 创建第二个图例，包含第二条线，放置在右下角
ax.legend(handles=[line2], loc='lower right')

# 显示图形
plt.show()
```