# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\simple_axisline4.py`

```
"""
================
Simple Axisline4
================

"""
# 导入必要的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入NumPy库

from mpl_toolkits.axes_grid1 import host_subplot  # 从matplotlib的axes_grid1模块导入host_subplot函数

# 创建一个主subplot对象
ax = host_subplot(111)
# 创建一个x轴数据数组，范围是0到2*pi，步长为0.01
xx = np.arange(0, 2*np.pi, 0.01)
# 在主subplot对象上绘制sin函数曲线
ax.plot(xx, np.sin(xx))

# 创建一个与主subplot对象关联的次subplot对象，负责"top"轴和"right"轴
ax2 = ax.twin()
# 设置次subplot对象的x轴刻度位置和标签
ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi],
               labels=["$0$", r"$\frac{1}{2}\pi$",
                       r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

# 设置次subplot对象的右边轴的主刻度标签不可见
ax2.axis["right"].major_ticklabels.set_visible(False)
# 设置次subplot对象的顶部轴的主刻度标签可见
ax2.axis["top"].major_ticklabels.set_visible(True)

# 显示绘制的图形
plt.show()
```