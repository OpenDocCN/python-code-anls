# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\annotate_text_arrow.py`

```
"""
===================
Annotate Text Arrow
===================

"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

# 设定随机数种子，以便结果可复现
np.random.seed(19680801)

# 创建一个图形和一个坐标轴对象，尺寸为 5x5
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect(1)  # 设置坐标轴纵横比为1，保证图形不会变形

# 生成两组随机数据点
x1 = -1 + np.random.randn(100)
y1 = -1 + np.random.randn(100)
x2 = 1. + np.random.randn(100)
y2 = 1. + np.random.randn(100)

# 在坐标轴上绘制散点图，颜色分别为红色和绿色
ax.scatter(x1, y1, color="r")
ax.scatter(x2, y2, color="g")

# 定义文本框的属性，圆角矩形，白色填充，灰色边框，透明度为0.9
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
# 在指定位置(-2, -2)处添加文本 "Sample A"，居中对齐，字体大小为20，应用上述定义的文本框属性
ax.text(-2, -2, "Sample A", ha="center", va="center", size=20,
        bbox=bbox_props)
# 在指定位置(2, 2)处添加文本 "Sample B"，居中对齐，字体大小为20，应用同样的文本框属性
ax.text(2, 2, "Sample B", ha="center", va="center", size=20,
        bbox=bbox_props)

# 定义箭头形式的文本框属性，向右箭头，浅蓝色填充，蓝色边框，线宽为2
bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="b", lw=2)
# 在坐标轴中心位置添加文本 "Direction"，居中对齐，字体大小为15，旋转角度为45度，应用箭头形式的文本框属性
t = ax.text(0, 0, "Direction", ha="center", va="center", rotation=45,
            size=15,
            bbox=bbox_props)

# 获取文本框对象的路径
bb = t.get_bbox_patch()
# 设置文本框的形状为向右箭头形式，内边距为0.6
bb.set_boxstyle("rarrow", pad=0.6)

# 设置坐标轴的显示范围
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# 显示图形
plt.show()
```