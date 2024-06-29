# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\view_planes_3d.py`

```
"""
======================
Primary 3D view planes
======================

This example generates an "unfolded" 3D plot that shows each of the primary 3D
view planes. The elevation, azimuth, and roll angles required for each view are
labeled. You could print out this image and fold it into a box where each plane
forms a side of the box.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 定义函数 annotate_axes，用于在指定的坐标轴上添加文本注释
def annotate_axes(ax, text, fontsize=18):
    ax.text(x=0.5, y=0.5, z=0.5, s=text,
            va="center", ha="center", fontsize=fontsize, color="black")

# 定义视图列表，包括视图平面名称和对应的 (仰角, 方位角, 翻滚角) 信息
views = [('XY',   (90, -90, 0)),
         ('XZ',    (0, -90, 0)),
         ('YZ',    (0,   0, 0)),
         ('-XY', (-90,  90, 0)),
         ('-XZ',   (0,  90, 0)),
         ('-YZ',   (0, 180, 0))]

# 定义视图的布局结构，以及创建绘图对象和子图布局
layout = [['XY',  '.',   'L',   '.'],
          ['XZ', 'YZ', '-XZ', '-YZ'],
          ['.',   '.', '-XY',   '.']]
fig, axd = plt.subplot_mosaic(layout, subplot_kw={'projection': '3d'},
                              figsize=(12, 8.5))

# 针对每个视图，设置坐标轴标签、投影类型和视角，并添加标签注释
for plane, angles in views:
    axd[plane].set_xlabel('x')
    axd[plane].set_ylabel('y')
    axd[plane].set_zlabel('z')
    axd[plane].set_proj_type('ortho')  # 设置正交投影
    axd[plane].view_init(elev=angles[0], azim=angles[1], roll=angles[2])  # 设置视角
    axd[plane].set_box_aspect(None, zoom=1.25)  # 设置盒子的长宽比

    label = f'{plane}\n{angles}'
    annotate_axes(axd[plane], label, fontsize=14)

# 针对部分视图，调整坐标轴标签和显示方式
for plane in ('XY', '-XY'):
    axd[plane].set_zticklabels([])
    axd[plane].set_zlabel('')
for plane in ('XZ', '-XZ'):
    axd[plane].set_yticklabels([])
    axd[plane].set_ylabel('')
for plane in ('YZ', '-YZ'):
    axd[plane].set_xticklabels([])
    axd[plane].set_xlabel('')

# 添加整体标签注释，并关闭 'L' 视图的坐标轴
label = 'mplot3d primary view planes\n' + 'ax.view_init(elev, azim, roll)'
annotate_axes(axd['L'], label, fontsize=18)
axd['L'].set_axis_off()

# 显示绘图结果
plt.show()
```