# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\fillunder3d.py`

```
"""
=========================
Fill under 3D line graphs
=========================

Demonstrate how to create polygons which fill the space under a line
graph. In this example polygons are semi-transparent, creating a sort
of 'jagged stained glass' effect.
"""

# 导入所需的库
import math  # 导入数学函数库
import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

# 使用向量化函数来处理 gamma 函数
gamma = np.vectorize(math.gamma)

# 定义参数
N = 31
x = np.linspace(0., 10., N)
lambdas = range(1, 9)

# 创建 3D 子图
ax = plt.figure().add_subplot(projection='3d')

# 生成用于填充颜色的序列
facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(lambdas)))

# 遍历 lambda 参数
for i, l in enumerate(lambdas):
    # 使用 fill_between 方法填充多边形，可以接受长度为 N 的向量或标量作为坐标
    ax.fill_between(x, l, l**x * np.exp(-l) / gamma(x + 1),
                    x, l, 0,
                    facecolors=facecolors[i], alpha=.7)

# 设置坐标轴范围和标签
ax.set(xlim=(0, 10), ylim=(1, 9), zlim=(0, 0.35),
       xlabel='x', ylabel=r'$\lambda$', zlabel='probability')

# 显示图形
plt.show()
```