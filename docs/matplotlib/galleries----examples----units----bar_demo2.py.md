# `D:\src\scipysrc\matplotlib\galleries\examples\units\bar_demo2.py`

```py
"""
===================
Bar demo with units
===================

A plot using a variety of centimetre and inch conversions. This example shows
how default unit introspection works (ax1), how various keywords can be used to
set the x and y units to override the defaults (ax2, ax3, ax4) and how one can
set the xlimits using scalars (ax3, current units assumed) or units
(conversions applied to get the numbers to current units).

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""

# 导入需要的单位 cm 和 inch，以及 matplotlib.pyplot 和 numpy 库
from basic_units import cm, inch
import matplotlib.pyplot as plt
import numpy as np

# 创建一系列以 cm 为单位的数组
cms = cm * np.arange(0, 10, 2)
bottom = 0 * cm
width = 0.8 * cm

# 创建 2x2 的子图
fig, axs = plt.subplots(2, 2)

# 在 axs[0, 0] 中绘制柱状图，使用 cm 作为默认单位
axs[0, 0].bar(cms, cms, bottom=bottom)

# 在 axs[0, 1] 中绘制柱状图，设置 x 和 y 的单位分别为 cm 和 inch
axs[0, 1].bar(cms, cms, bottom=bottom, width=width, xunits=cm, yunits=inch)

# 在 axs[1, 0] 中绘制柱状图，设置 x 和 y 的单位分别为 inch 和 cm，并设置 x 范围为 2 到 6（假定当前单位）
axs[1, 0].bar(cms, cms, bottom=bottom, width=width, xunits=inch, yunits=cm)
axs[1, 0].set_xlim(2, 6)  # scalars are interpreted in current units

# 在 axs[1, 1] 中绘制柱状图，设置 x 和 y 的单位分别为 inch 和 inch，并设置 x 范围为 2 cm 到 6 cm（转换为 inches）
axs[1, 1].bar(cms, cms, bottom=bottom, width=width, xunits=inch, yunits=inch)
axs[1, 1].set_xlim(2 * cm, 6 * cm)  # cm are converted to inches

# 调整子图的布局，使得各个部分紧凑显示
fig.tight_layout()

# 显示图形
plt.show()
```