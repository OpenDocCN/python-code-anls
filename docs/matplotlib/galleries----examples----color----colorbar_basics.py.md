# `D:\src\scipysrc\matplotlib\galleries\examples\color\colorbar_basics.py`

```
"""
========
Colorbar
========

Use `~.Figure.colorbar` by specifying the mappable object (here
the `.AxesImage` returned by `~.axes.Axes.imshow`)
and the Axes to attach the colorbar to.
"""

# 导入 matplotlib.pyplot 库作为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并简称为 np
import numpy as np

# 设置一些通用数据
N = 37
# 创建一个 N x N 的网格矩阵
x, y = np.mgrid[:N, :N]
# 根据 x 和 y 的值计算一个二维数组 Z
Z = (np.cos(x*0.2) + np.sin(y*0.3))

# 将负值和正值分别遮盖掉
Zpos = np.ma.masked_less(Z, 0)
Zneg = np.ma.masked_greater(Z, 0)

# 创建一个包含三个子图的 Figure 对象，并指定子图的大小和列数
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

# 在第一个子图 ax1 上绘制仅包含正值数据的图像，并保存 imshow 返回的颜色映射对象
pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')

# 使用 Figure 对象的 colorbar 方法添加颜色条，指定要添加颜色条的映射对象和对应的 Axes 对象
fig.colorbar(pos, ax=ax1)

# 在第二个子图 ax2 上重复上述步骤，但是绘制包含负值数据的图像，并指定颜色条的位置、锚点和收缩比例
neg = ax2.imshow(Zneg, cmap='Reds_r', interpolation='none')
fig.colorbar(neg, ax=ax2, location='right', anchor=(0, 0.3), shrink=0.7)

# 在第三个子图 ax3 上绘制包含正负值在 +/- 1.2 之间的图像，并设置颜色映射的最小值和最大值，不进行插值
pos_neg_clipped = ax3.imshow(Z, cmap='RdBu', vmin=-1.2, vmax=1.2,
                             interpolation='none')

# 添加颜色条到第三个子图 ax3，并设置颜色条的扩展方式为双向，同时在颜色条上添加次刻度线
cbar = fig.colorbar(pos_neg_clipped, ax=ax3, extend='both')
cbar.minorticks_on()

# 显示绘制的图形
plt.show()
```