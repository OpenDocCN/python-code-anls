# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\demo_annotation_box.py`

```
"""
===================
AnnotationBbox demo
===================

`.AnnotationBbox` creates an annotation using an `.OffsetBox`, and
provides more fine-grained control than `.Axes.annotate`.  This example
demonstrates the use of AnnotationBbox together with three different
OffsetBoxes: `.TextArea`, `.DrawingArea`, and `.OffsetImage`.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from matplotlib.cbook import get_sample_data  # 从matplotlib库中的cbook模块导入get_sample_data函数，用于获取示例数据
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,  # 从matplotlib库中的offsetbox模块导入需要使用的类
                                  TextArea)
from matplotlib.patches import Circle  # 从matplotlib库中的patches模块导入Circle类，用于绘制圆形图形

fig, ax = plt.subplots()  # 创建一个新的图形和一个子图

# Define a 1st position to annotate (display it with a marker)
xy = (0.5, 0.7)  # 设置第一个注释位置的坐标
ax.plot(xy[0], xy[1], ".r")  # 在子图中绘制一个红色的点作为标记位置

# Annotate the 1st position with a text box ('Test 1')
offsetbox = TextArea("Test 1")  # 创建一个文本区域作为注释内容

ab = AnnotationBbox(offsetbox, xy,
                    xybox=(-20, 40),
                    xycoords='data',
                    boxcoords="offset points",
                    arrowprops=dict(arrowstyle="->"),
                    bboxprops=dict(boxstyle="sawtooth"))  # 创建一个基于注释内容和位置的注释框
ax.add_artist(ab)  # 将注释框添加到子图中

# Annotate the 1st position with another text box ('Test')
offsetbox = TextArea("Test")  # 创建另一个文本区域作为注释内容

ab = AnnotationBbox(offsetbox, xy,
                    xybox=(1.02, xy[1]),
                    xycoords='data',
                    boxcoords=("axes fraction", "data"),
                    box_alignment=(0., 0.5),
                    arrowprops=dict(arrowstyle="->"))  # 创建第二个基于不同坐标系的注释框
ax.add_artist(ab)  # 将注释框添加到子图中

# Define a 2nd position to annotate (don't display with a marker this time)
xy = [0.3, 0.55]  # 设置第二个注释位置的坐标

# Annotate the 2nd position with a circle patch
da = DrawingArea(20, 20, 0, 0)  # 创建一个绘制区域，并在其中绘制一个圆形图形
p = Circle((10, 10), 10)  # 创建一个圆形图形

da.add_artist(p)  # 将圆形图形添加到绘制区域中

ab = AnnotationBbox(da, xy,
                    xybox=(1., xy[1]),
                    xycoords='data',
                    boxcoords=("axes fraction", "data"),
                    box_alignment=(0.2, 0.5),
                    arrowprops=dict(arrowstyle="->"),
                    bboxprops=dict(alpha=0.5))  # 创建基于绘制区域的注释框
ax.add_artist(ab)  # 将注释框添加到子图中

# Annotate the 2nd position with an image (a generated array of pixels)
arr = np.arange(100).reshape((10, 10))  # 创建一个10x10的像素数组
im = OffsetImage(arr, zoom=2)  # 创建一个偏移图像对象并设置放大倍数

im.image.axes = ax  # 设置图像所在的坐标轴

ab = AnnotationBbox(im, xy,
                    xybox=(-50., 50.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.3,
                    arrowprops=dict(arrowstyle="->"))  # 创建基于偏移图像的注释框
ax.add_artist(ab)  # 将注释框添加到子图中

# Annotate the 2nd position with another image (a Grace Hopper portrait)
with get_sample_data("grace_hopper.jpg") as file:  # 使用matplotlib提供的样本数据中的Grace Hopper肖像图像
    arr_img = plt.imread(file)  # 读取图像数据

imagebox = OffsetImage(arr_img, zoom=0.2)  # 创建一个偏移图像对象并设置缩放倍数
imagebox.image.axes = ax  # 设置图像所在的坐标轴
# 创建一个注释框对象 `ab`，其包含一个图像 `imagebox`，显示在坐标 `xy` 处，
# 注释框本身位于数据坐标系中，相对于注释点偏移 120 像素横向，-80 像素纵向，
# 使用偏移点坐标系来定位注释框，设置填充为 0.5，
# 注释框的箭头样式为 "->"，连接样式为 "angle,angleA=0,angleB=90,rad=3"
ab = AnnotationBbox(imagebox, xy,
                    xybox=(120., -80.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=3")
                    )

# 将注释框对象 `ab` 添加到图形 `ax` 中
ax.add_artist(ab)

# 设置图形的 X 轴显示范围为 0 到 1
ax.set_xlim(0, 1)
# 设置图形的 Y 轴显示范围为 0 到 1
ax.set_ylim(0, 1)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches.Circle`
#    - `matplotlib.offsetbox.TextArea`
#    - `matplotlib.offsetbox.DrawingArea`
#    - `matplotlib.offsetbox.OffsetImage`
#    - `matplotlib.offsetbox.AnnotationBbox`
#    - `matplotlib.cbook.get_sample_data`
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.pyplot.imread`
```