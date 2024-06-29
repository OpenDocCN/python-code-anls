# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\annotation_polar.py`

```py
"""
================
Annotation Polar
================

This example shows how to create an annotation on a polar graph.

For a complete overview of the annotation capabilities, also see the
:ref:`annotations`.

.. redirect-from:: /gallery/pyplots/annotation_polar
"""
# 导入必要的库
import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

# 创建一个新的图形对象
fig = plt.figure()
# 添加极坐标子图
ax = fig.add_subplot(projection='polar')
# 生成极坐标系下的数据
r = np.arange(0, 1, 0.001)
theta = 2 * 2*np.pi * r
# 绘制极坐标系下的曲线
line, = ax.plot(theta, r, color='#ee8d18', lw=3)

# 选择数据中的一个点
ind = 800
thisr, thistheta = r[ind], theta[ind]
# 在图上标记选定的点
ax.plot([thistheta], [thisr], 'o')
# 添加注释到图上的选定点
ax.annotate('a polar annotation',
            xy=(thistheta, thisr),     # 注释的位置，极坐标下的角度和半径
            xytext=(0.05, 0.05),       # 文字的位置，相对于图形的左下角
            textcoords='figure fraction',  # 文字位置的坐标系
            arrowprops=dict(facecolor='black', shrink=0.05),  # 箭头的属性设置
            horizontalalignment='left',    # 文字的水平对齐方式
            verticalalignment='bottom',    # 文字的垂直对齐方式
            )
# 显示绘制的图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.projections.polar`
#    - `matplotlib.axes.Axes.annotate` / `matplotlib.pyplot.annotate`
```