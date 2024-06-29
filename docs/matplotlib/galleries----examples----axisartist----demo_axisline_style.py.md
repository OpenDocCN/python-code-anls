# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_axisline_style.py`

```
"""
================
Axis line styles
================

This example shows some configurations for axis style.

Note: The `mpl_toolkits.axisartist` Axes classes may be confusing for new
users. If the only aim is to obtain arrow heads at the ends of the axes,
rather check out the :doc:`/gallery/spines/centered_spines_with_arrows`
example.
"""

import matplotlib.pyplot as plt   # 导入matplotlib.pyplot库，用于绘图
import numpy as np   # 导入numpy库，用于数学运算

from mpl_toolkits.axisartist.axislines import AxesZero   # 从mpl_toolkits.axisartist.axislines导入AxesZero类

fig = plt.figure()   # 创建一个新的图形窗口
ax = fig.add_subplot(axes_class=AxesZero)   # 添加一个subplot，并指定使用AxesZero类来绘制坐标轴

for direction in ["xzero", "yzero"]:
    # 在每个坐标轴的末端添加箭头
    ax.axis[direction].set_axisline_style("-|>")
    
    # 显示X和Y轴从原点开始
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    # 隐藏图形的左侧、右侧、底部和顶部边框
    ax.axis[direction].set_visible(False)

x = np.linspace(-0.5, 1., 100)   # 生成从-0.5到1之间100个等间距的数
ax.plot(x, np.sin(x*np.pi))   # 绘制正弦函数曲线

plt.show()   # 显示图形
```