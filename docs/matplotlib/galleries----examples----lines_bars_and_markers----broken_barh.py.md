# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\broken_barh.py`

```
"""
===========
Broken Barh
===========

Make a "broken" horizontal bar plot, i.e., one with gaps
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()

# 绘制第一个组合的“broken”水平条形图，定义了两个区间和它们的位置
ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue')

# 绘制第二个组合的“broken”水平条形图，定义了三个区间和它们的位置以及颜色
ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),
               facecolors=('tab:orange', 'tab:green', 'tab:red'))

# 设置 y 轴的显示范围
ax.set_ylim(5, 35)

# 设置 x 轴的显示范围
ax.set_xlim(0, 200)

# 设置 x 轴的标签
ax.set_xlabel('seconds since start')

# 修改 y 轴的刻度标签为 ['Bill', 'Jim']
ax.set_yticks([15, 25], labels=['Bill', 'Jim'])

# 显示网格线
ax.grid(True)

# 在图上添加注释 'race interrupted'，指定位置、文本偏移、箭头属性等
ax.annotate('race interrupted', (61, 25),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')

# 显示图形
plt.show()
```