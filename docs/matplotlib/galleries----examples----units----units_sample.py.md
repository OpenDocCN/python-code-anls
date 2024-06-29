# `D:\src\scipysrc\matplotlib\galleries\examples\units\units_sample.py`

```py
"""
======================
Inches and Centimeters
======================

The example illustrates the ability to override default x and y units (ax1) to
inches and centimeters using the *xunits* and *yunits* parameters for the
`~.axes.Axes.plot` function. Note that conversions are applied to get numbers
to correct units.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`

"""
# 导入 cm 和 inch 单位定义
from basic_units import cm, inch

# 导入 matplotlib.pyplot 库，并将其命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并将其命名为 np
import numpy as np

# 创建一个以 constrained 布局的 2x2 子图网格
fig, axs = plt.subplots(2, 2, layout='constrained')

# 在第一个子图(axs[0, 0])上绘制以厘米为单位的数据
axs[0, 0].plot(cms, cms)

# 在第二个子图(axs[0, 1])上绘制以厘米为 x 轴单位和英寸为 y 轴单位的数据
axs[0, 1].plot(cms, cms, xunits=cm, yunits=inch)

# 在第三个子图(axs[1, 0])上绘制以英寸为 x 轴单位和厘米为 y 轴单位的数据，并设置 x 轴范围为 -1 到 4（单位为当前轴的默认单位）
axs[1, 0].plot(cms, cms, xunits=inch, yunits=cm)
axs[1, 0].set_xlim(-1, 4)

# 在第四个子图(axs[1, 1])上绘制以英寸为 x 和 y 轴单位的数据，并设置 x 轴范围为 3 到 6 厘米（单位被转换为英寸）
axs[1, 1].plot(cms, cms, xunits=inch, yunits=inch)
axs[1, 1].set_xlim(3*cm, 6*cm)

# 显示图形
plt.show()
```