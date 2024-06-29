# `D:\src\scipysrc\matplotlib\galleries\examples\units\units_scatter.py`

```py
"""
=============
Unit handling
=============

The example below shows support for unit conversions over masked
arrays.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""

# 从 basic_units 模块导入单位 hertz, minutes, secs
from basic_units import hertz, minutes, secs

# 导入 matplotlib.pyplot 库并使用 plt 别名
import matplotlib.pyplot as plt
# 导入 numpy 库并使用 np 别名
import numpy as np

# 创建一个被掩码数组（masked array）
data = (1, 2, 3, 4, 5, 6, 7, 8)
mask = (1, 0, 1, 0, 0, 0, 1, 0)
# 使用 secs 单位，创建一个被掩码数组 xsecs
xsecs = secs * np.ma.MaskedArray(data, mask, float)

# 创建一个包含三个子图的图形对象，共享 x 轴
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

# 在第一个子图上绘制散点图，使用 xsecs 作为 x 和 y 轴数据，y 轴使用 secs 单位
ax1.scatter(xsecs, xsecs)
ax1.yaxis.set_units(secs)

# 在第二个子图上绘制散点图，使用 xsecs 作为 x 和 y 轴数据，y 轴使用 hertz 单位
ax2.scatter(xsecs, xsecs, yunits=hertz)

# 在第三个子图上绘制散点图，使用 xsecs 作为 x 和 y 轴数据，y 轴使用 minutes 单位
ax3.scatter(xsecs, xsecs, yunits=minutes)

# 调整子图布局，使其更加紧凑
fig.tight_layout()

# 显示图形
plt.show()
```