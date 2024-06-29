# `D:\src\scipysrc\matplotlib\galleries\examples\units\annotate_with_units.py`

```
"""
=====================
Annotation with units
=====================

The example illustrates how to create text and arrow
annotations using a centimeter-scale plot.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""

# 从 basic_units 模块中导入 cm 单位
from basic_units import cm

# 导入 matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt

# 创建一个新的图形和轴对象
fig, ax = plt.subplots()

# 在图上添加一个带注释的点，位置为 [0.5cm, 0.5cm]
ax.annotate("Note 01", [0.5*cm, 0.5*cm])

# 在数据坐标系中添加带箭头的注释，注释位置为 (3cm, 1cm)，文本位置为 (0.8cm, 0.95cm)
ax.annotate('local max', xy=(3*cm, 1*cm), xycoords='data',
            xytext=(0.8*cm, 0.95*cm), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

# 混合使用带单位和不带单位的坐标进行注释
ax.annotate('local max', xy=(3*cm, 1*cm), xycoords='data',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

# 设置 x 和 y 轴的范围为 0 到 4 厘米
ax.set_xlim(0*cm, 4*cm)
ax.set_ylim(0*cm, 4*cm)

# 显示图形
plt.show()
```