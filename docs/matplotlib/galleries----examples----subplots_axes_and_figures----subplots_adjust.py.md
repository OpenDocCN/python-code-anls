# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\subplots_adjust.py`

```
"""
=============================
Subplots spacings and margins
=============================

Adjusting the spacing of margins and subplots using `.pyplot.subplots_adjust`.

.. note::
   There is also a tool window to adjust the margins and spacings of displayed
   figures interactively.  It can be opened via the toolbar or by calling
   `.pyplot.subplot_tool`.

.. redirect-from:: /gallery/subplots_axes_and_figures/subplot_toolbar
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图操作
import numpy as np  # 导入 numpy 库，用于生成随机数据

# 设置随机种子以便结果可复现
np.random.seed(19680801)

# 创建第一个子图，位于总图的上半部分
plt.subplot(211)
# 在子图中显示一个 100x100 的随机数据的图像
plt.imshow(np.random.random((100, 100)))

# 创建第二个子图，位于总图的下半部分
plt.subplot(212)
# 在子图中显示一个 100x100 的随机数据的图像
plt.imshow(np.random.random((100, 100)))

# 调整子图和边缘的间距，设置底部边缘为 0.1，右侧边缘为 0.8，顶部边缘为 0.9
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

# 在特定位置创建一个新的轴用于显示颜色条，左边距离为 0.85，底部边距离为 0.1，宽度为 0.075，高度为 0.8
cax = plt.axes((0.85, 0.1, 0.075, 0.8))
# 在指定的轴上添加颜色条
plt.colorbar(cax=cax)

# 显示图形
plt.show()
```