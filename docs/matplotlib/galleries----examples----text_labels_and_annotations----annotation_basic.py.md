# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\annotation_basic.py`

```
"""
=================
Annotating a plot
=================

This example shows how to annotate a plot with an arrow pointing to provided
coordinates. We modify the defaults of the arrow, to "shrink" it.

For a complete overview of the annotation capabilities, also see the
:ref:`annotation tutorial<annotations>`.

.. redirect-from:: /gallery/pyplots/annotation_basic
"""

# 导入 matplotlib 的 pyplot 模块，简称 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，简称 np
import numpy as np

# 创建一个新的图形和轴对象
fig, ax = plt.subplots()

# 生成一些数据
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)

# 绘制数据曲线，并保存线条对象
line, = ax.plot(t, s, lw=2)

# 在图中添加注释，指向坐标 (2, 1)，注释文本为 'local max'，注释箭头属性设置为缩小尺寸且黑色
ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

# 设置 y 轴的显示范围
ax.set_ylim(-2, 2)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.annotate` / `matplotlib.pyplot.annotate`
```