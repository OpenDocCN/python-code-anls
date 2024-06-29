# `D:\src\scipysrc\matplotlib\galleries\examples\units\radian_demo.py`

```
"""
============
Radian ticks
============

Plot with radians from the basic_units mockup example package.


This example shows how the unit class can determine the tick locating,
formatting and axis labeling.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""

# 导入需要的函数和类：cos, degrees, radians
from basic_units import cos, degrees, radians

# 导入绘图库
import matplotlib.pyplot as plt
# 导入数值计算库
import numpy as np

# 创建一个包含从0到15（不包括15），步长为0.01的角度值的列表，并将其转换为弧度单位
x = [val*radians for val in np.arange(0, 15, 0.01)]

# 创建一个包含两个子图的图形对象
fig, axs = plt.subplots(2)

# 在第一个子图上绘制余弦函数图像，x轴使用弧度单位
axs[0].plot(x, cos(x), xunits=radians)

# 在第二个子图上绘制余弦函数图像，x轴使用角度单位
axs[1].plot(x, cos(x), xunits=degrees)

# 调整子图布局
fig.tight_layout()

# 显示图形
plt.show()
```