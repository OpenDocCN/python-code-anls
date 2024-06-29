# `D:\src\scipysrc\matplotlib\galleries\examples\pie_and_polar_charts\bar_of_pie.py`

```
"""
==========
Bar of pie
==========

Make a "bar of pie" chart where the first slice of the pie is
"exploded" into a bar chart with a further breakdown of said slice's
characteristics. The example demonstrates using a figure with multiple
sets of Axes and using the Axes patches list to add two ConnectionPatches
to link the subplot charts.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入NumPy数学库

from matplotlib.patches import ConnectionPatch  # 导入连接补丁类

# 创建图形对象和轴对象
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))  # 创建一个包含两个子图的图形对象
fig.subplots_adjust(wspace=0)  # 调整子图之间的水平间距为0

# 饼图参数设定
overall_ratios = [.27, .56, .17]  # 总体比例
labels = ['Approve', 'Disapprove', 'Undecided']  # 标签
explode = [0.1, 0, 0]  # 突出显示第一个部分
angle = -180 * overall_ratios[0]  # 根据第一个部分的比例确定起始角度
wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                     labels=labels, explode=explode)  # 绘制饼图并返回绘制的楔块对象

# 条形图参数设定
age_ratios = [.33, .54, .07, .06]  # 年龄比例
age_labels = ['Under 35', '35-49', '50-65', 'Over 65']  # 年龄段标签
bottom = 1  # 初始底部位置
width = .2  # 条形图宽度

# 从顶部开始添加，与图例匹配
for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                 alpha=0.1 + 0.25 * j)  # 添加条形图并设置颜色、标签和透明度
    ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')  # 在条形图上添加标签

ax2.set_title('Age of approvers')  # 设置子图2的标题
ax2.legend()  # 显示图例
ax2.axis('off')  # 关闭坐标轴显示
ax2.set_xlim(- 2.5 * width, 2.5 * width)  # 设置x轴限制范围

# 使用ConnectionPatch在两个子图之间绘制连接线
theta1, theta2 = wedges[0].theta1, wedges[0].theta2  # 获取第一个楔块的起始和终止角度
center, r = wedges[0].center, wedges[0].r  # 获取第一个楔块的中心点和半径
bar_height = sum(age_ratios)  # 计算条形图的总高度

# 绘制顶部连接线
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)  # 创建连接补丁对象
con.set_color([0, 0, 0])  # 设置连接线颜色
con.set_linewidth(4)  # 设置连接线宽度
ax2.add_artist(con)  # 添加连接线到子图2

# 绘制底部连接线
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)  # 创建连接补丁对象
con.set_color([0, 0, 0])  # 设置连接线颜色
con.set_linewidth(4)  # 设置连接线宽度
ax2.add_artist(con)  # 添加连接线到子图2

plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.pie` / `matplotlib.pyplot.pie`
#    - `matplotlib.patches.ConnectionPatch`
```