# `D:\src\scipysrc\matplotlib\galleries\users_explain\artists\paths.py`

```py
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# 定义顶点列表，表示要绘制的矩形的四个角落
verts = [
   (0., 0.),  # 左下角
   (0., 1.),  # 左上角
   (1., 1.),  # 右上角
   (1., 0.),  # 右下角
   (0., 0.),  # 忽略的顶点，路径闭合时使用
]

# 定义路径命令列表，用于指定绘制路径的操作
codes = [
    Path.MOVETO,    # 移动到第一个顶点
    Path.LINETO,    # 从当前位置画直线到第二个顶点
    Path.LINETO,    # 继续画直线到第三个顶点
    Path.LINETO,    # 继续画直线到第四个顶点
    Path.CLOSEPOLY, # 关闭路径，画一条线段连接到起始点形成闭合多边形
]

# 创建路径对象 Path，将顶点列表和路径命令列表传入
path = Path(verts, codes)

# 创建图形和轴对象
fig, ax = plt.subplots()

# 创建路径补丁对象 PathPatch，设置填充颜色为橙色，线宽为2
patch = patches.PathPatch(path, facecolor='orange', lw=2)

# 将路径补丁对象添加到轴上
ax.add_patch(patch)

# 设置坐标轴范围
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# 显示图形
plt.show()
# 定义四个顶点坐标，用于描述一个包含贝塞尔曲线的路径
verts = [
   (0., 0.),   # P0 贝塞尔曲线起始点
   (0.2, 1.),  # P1 贝塞尔曲线第一个控制点
   (1., 0.8),  # P2 贝塞尔曲线第二个控制点
   (0.8, 0.),  # P3 贝塞尔曲线结束点
]

# 指定路径中各个顶点的操作码，表示每个顶点的类型
codes = [
    Path.MOVETO,    # 将笔移动到下一个坐标点
    Path.CURVE4,    # 用四次贝塞尔曲线连接到下一个点
    Path.CURVE4,    # 用四次贝塞尔曲线连接到下一个点
    Path.CURVE4,    # 用四次贝塞尔曲线连接到下一个点
]

# 使用给定的顶点和操作码创建路径对象
path = Path(verts, codes)

# 创建图形和坐标轴对象
fig, ax = plt.subplots()

# 创建路径对象的图形表示，添加到坐标轴上
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax.add_patch(patch)

# 画出路径上的顶点，用黑色交叉标记表示
xs, ys = zip(*verts)
ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)

# 在图中添加标签显示各顶点的名称
ax.text(-0.05, -0.05, 'P0')
ax.text(0.15, 1.05, 'P1')
ax.text(1.05, 0.85, 'P2')
ax.text(0.85, -0.05, 'P3')

# 设置坐标轴的范围
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

# 显示图形
plt.show()
fig, ax = plt.subplots()
# 创建一个新的图形窗口和一个子图对象

np.random.seed(19680801)
# 设定随机种子以便结果可重复

data = np.random.randn(1000)
# 生成1000个符合标准正态分布的随机数

n, bins = np.histogram(data, 100)
# 用numpy对数据进行直方图统计，返回频数和分箱边界

left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)
# 计算直方图各矩形的左右下上边界及个数

nverts = nrects*(1+3+1)
# 计算所有顶点的数量，每个矩形需要1个MOVETO和3个LINETO，以及每个矩形闭合需要1个CLOSEPOLY

verts = np.zeros((nverts, 2))
# 创建一个形状为(nverts, 2)的全零数组，用来存放矩形的顶点坐标

codes = np.full(nverts, Path.LINETO, dtype=int)
# 创建一个长度为nverts的整数数组，初始值全为LINETO，表示顶点的连接方式

codes[0::5] = Path.MOVETO
# 将每个矩形的第一个顶点标记为MOVETO，即起始点

codes[4::5] = Path.CLOSEPOLY
# 将每个矩形的最后一个顶点标记为CLOSEPOLY，即闭合路径

verts[0::5, 0] = left
verts[0::5, 1] = bottom
verts[1::5, 0] = left
verts[1::5, 1] = top
verts[2::5, 0] = right
verts[2::5, 1] = top
verts[3::5, 0] = right
verts[3::5, 1] = bottom
# 填充顶点数组，分别为左下角、左上角、右上角、右下角的顶点坐标

barpath = Path(verts, codes)
# 创建一个路径对象，用给定的顶点和连接方式数组

patch = patches.PathPatch(barpath, facecolor='green',
                          edgecolor='yellow', alpha=0.5)
# 创建一个PathPatch对象，用于绘制矩形，并设定其填充色、边界色和透明度

ax.add_patch(patch)
# 将矩形添加到子图ax中显示

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())
# 设置子图的坐标轴范围，确保矩形全部显示在图像中

plt.show()
# 显示图形
```