# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\scatter_star_poly.py`

```
"""
===============
Marker examples
===============

Example with different ways to specify markers.

See also the `matplotlib.markers` documentation for a list of all markers and
:doc:`/gallery/lines_bars_and_markers/marker_reference` for more information
on configuring markers.

.. redirect-from:: /gallery/lines_bars_and_markers/scatter_custom_symbol
.. redirect-from:: /gallery/lines_bars_and_markers/scatter_symbol
.. redirect-from:: /gallery/lines_bars_and_markers/scatter_piecharts
"""
# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 设置随机种子以便结果可复现
np.random.seed(19680801)

# 生成随机数据
x = np.random.rand(10)
y = np.random.rand(10)
z = np.sqrt(x**2 + y**2)

# 创建包含 2x3 子图的图形对象
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, layout="constrained")

# 在第一行第一列的子图中绘制散点图，使用 '>' 作为标记符号
axs[0, 0].scatter(x, y, s=80, c=z, marker=">")
axs[0, 0].set_title("marker='>'")

# 在第一行第二列的子图中绘制散点图，使用 '$\clubsuit$' 作为 LaTeX 符号
axs[0, 1].scatter(x, y, s=80, c=z, marker=r"$\clubsuit$")
axs[0, 1].set_title(r"marker=r'\$\clubsuit\$'")

# 在第一行第三列的子图中绘制散点图，使用自定义路径作为标记，verts 是一个顶点列表
verts = [[-1, -1], [1, -1], [1, 1], [-1, -1]]
axs[0, 2].scatter(x, y, s=80, c=z, marker=verts)
axs[0, 2].set_title("marker=verts")

# 在第二行第一列的子图中绘制散点图，使用正五边形标记
axs[1, 0].scatter(x, y, s=80, c=z, marker=(5, 0))
axs[1, 0].set_title("marker=(5, 0)")

# 在第二行第二列的子图中绘制散点图，使用正五角星标记
axs[1, 1].scatter(x, y, s=80, c=z, marker=(5, 1))
axs[1, 1].set_title("marker=(5, 1)")

# 在第二行第三列的子图中绘制散点图，使用正五角星（带空心）标记
axs[1, 2].scatter(x, y, s=80, c=z, marker=(5, 2))
axs[1, 2].set_title("marker=(5, 2)")

# 显示图形
plt.show()
```