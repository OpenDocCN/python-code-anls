# `D:\src\scipysrc\matplotlib\galleries\examples\color\set_alpha.py`

```py
"""
=================================
设置颜色的透明度值的几种方法
=================================

比较通过 *alpha* 关键字参数和通过 Matplotlib 颜色格式设置透明度的方法。通常，*alpha* 关键字是添加颜色透明度的唯一工具。在某些情况下，*(matplotlib_color, alpha)* 颜色格式提供了一种调整图表外观的简便方法。

"""

import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以便可重现性
np.random.seed(19680801)

# 创建一个包含两个子图的图形对象
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

# 创建随机的 x 和 y 值
x_values = [n for n in range(20)]
y_values = np.random.randn(20)

# 根据 y 值的正负设置柱状图的颜色
facecolors = ['green' if y > 0 else 'red' for y in y_values]
edgecolors = facecolors

# 在第一个子图上绘制柱状图，设置颜色、边缘颜色和透明度
ax1.bar(x_values, y_values, color=facecolors, edgecolor=edgecolors, alpha=0.5)
ax1.set_title("明确指定 'alpha' 关键字的数值\n应用于所有柱体和边缘")

# 将 y 值归一化，以获得不同的柱体透明度值
abs_y = [abs(y) for y in y_values]
face_alphas = [n / max(abs_y) for n in abs_y]
edge_alphas = [1 - alpha for alpha in face_alphas]

# 将颜色和透明度值进行配对
colors_with_alphas = list(zip(facecolors, face_alphas))
edgecolors_with_alphas = list(zip(edgecolors, edge_alphas))

# 在第二个子图上绘制柱状图，设置颜色和边缘颜色，并应用归一化的透明度
ax2.bar(x_values, y_values, color=colors_with_alphas,
        edgecolor=edgecolors_with_alphas)
ax2.set_title('每个柱体和边缘的归一化透明度')

plt.show()

# %%
#
# .. admonition:: 参考文献
#
#    本示例展示了以下函数、方法、类和模块的使用：
#
#    - `matplotlib.axes.Axes.bar`
#    - `matplotlib.pyplot.subplots`
```