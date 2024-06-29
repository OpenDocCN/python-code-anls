# `D:\src\scipysrc\matplotlib\galleries\examples\units\bar_unit_demo.py`

```
"""
=========================
Group barchart with units
=========================

This is the same example as
:doc:`the barchart</gallery/lines_bars_and_markers/barchart>` in
centimeters.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""

# 导入需要的单位定义，包括厘米和英寸
from basic_units import cm, inch

# 导入 matplotlib 库中的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 设置数据组数
N = 5
# 茶的平均高度数据，使用厘米单位
tea_means = [15*cm, 10*cm, 8*cm, 12*cm, 5*cm]
# 茶的高度标准差数据，使用厘米单位
tea_std = [2*cm, 1*cm, 1*cm, 4*cm, 2*cm]

# 创建画布和子图对象
fig, ax = plt.subplots()
# 设置 y 轴单位为英寸
ax.yaxis.set_units(inch)

# 设置 x 轴位置数组，用于组的位置
ind = np.arange(N)    # the x locations for the groups
# 设置条形图的宽度
width = 0.35         # the width of the bars
# 绘制茶的条形图，包括高度数据、宽度、底部、误差条和标签
ax.bar(ind, tea_means, width, bottom=0*cm, yerr=tea_std, label='Tea')

# 咖啡的平均高度数据和标准差数据，使用厘米单位
coffee_means = (14*cm, 19*cm, 7*cm, 5*cm, 10*cm)
coffee_std = (3*cm, 5*cm, 2*cm, 1*cm, 2*cm)
# 绘制咖啡的条形图，包括高度数据、宽度、底部、误差条和标签
ax.bar(ind + width, coffee_means, width, bottom=0*cm, yerr=coffee_std,
       label='Coffee')

# 设置图表标题
ax.set_title('Cup height by group and beverage choice')
# 设置 x 轴刻度位置和标签
ax.set_xticks(ind + width / 2, labels=['G1', 'G2', 'G3', 'G4', 'G5'])

# 添加图例
ax.legend()
# 自动调整视图
ax.autoscale_view()

# 显示图表
plt.show()
```