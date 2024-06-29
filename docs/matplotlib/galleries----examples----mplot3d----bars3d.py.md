# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\bars3d.py`

```py
"""
========================================
Create 2D bar graphs in different planes
========================================

Demonstrates making a 3D plot which has 2D bar graphs projected onto
planes y=0, y=1, etc.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

# 设置随机数种子以便结果可复现
np.random.seed(19680801)

# 创建一个新的图形对象
fig = plt.figure()

# 添加一个带有3D投影的子图
ax = fig.add_subplot(projection='3d')

# 定义颜色列表和y轴刻度列表
colors = ['r', 'g', 'b', 'y']
yticks = [3, 2, 1, 0]

# 遍历颜色和y轴刻度，每个循环生成一个y=k平面的随机数据条形图
for c, k in zip(colors, yticks):
    # 生成y=k平面上的随机数据
    xs = np.arange(20)  # 生成x轴数据（长度为20）
    ys = np.random.rand(20)  # 生成y轴随机数据（长度为20）

    # 指定每个数据条的颜色，将第一个条设置为青色以示例说明
    cs = [c] * len(xs)
    cs[0] = 'c'

    # 在y=k平面上绘制由xs和ys给出的条形图，设置透明度为80%
    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

# 设置图形的x、y、z轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置y轴上的刻度，仅标记我们有数据的离散值
ax.set_yticks(yticks)

# 显示图形
plt.show()
```