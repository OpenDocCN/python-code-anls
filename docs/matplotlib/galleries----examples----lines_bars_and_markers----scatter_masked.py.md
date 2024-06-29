# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\scatter_masked.py`

```
"""
==============
Scatter Masked
==============

Mask some data points and add a line demarking
masked regions.

"""
# 引入matplotlib.pyplot库，并重命名为plt
import matplotlib.pyplot as plt
# 引入numpy库，并重命名为np
import numpy as np

# 设置随机种子以便结果可重复
np.random.seed(19680801)

# 定义数据点数量
N = 100
# 定义半径阈值
r0 = 0.6
# 生成在[0, 0.9)范围内的随机x坐标
x = 0.9 * np.random.rand(N)
# 生成在[0, 0.9)范围内的随机y坐标
y = 0.9 * np.random.rand(N)
# 生成每个数据点的面积，面积在[0, 400)之间
area = (20 * np.random.rand(N))**2  # 0 to 10 point radii
# 计算每个点的颜色值
c = np.sqrt(area)
# 计算每个点到原点的距离
r = np.sqrt(x ** 2 + y ** 2)
# 根据条件对面积进行掩码，小于r0的部分被掩盖
area1 = np.ma.masked_where(r < r0, area)
# 大于等于r0的部分被掩盖
area2 = np.ma.masked_where(r >= r0, area)
# 绘制散点图，使用area1进行掩码，三角形标记，颜色使用c
plt.scatter(x, y, s=area1, marker='^', c=c)
# 绘制散点图，使用area2进行掩码，圆形标记，颜色使用c
plt.scatter(x, y, s=area2, marker='o', c=c)
# 绘制标示掩码区域的边界线
theta = np.arange(0, np.pi / 2, 0.01)
plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

# 显示图形
plt.show()
```