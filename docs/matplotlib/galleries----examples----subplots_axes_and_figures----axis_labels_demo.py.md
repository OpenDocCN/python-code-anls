# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\axis_labels_demo.py`

```
"""
===================
Axis Label Position
===================

Choose axis label position when calling `~.Axes.set_xlabel` and
`~.Axes.set_ylabel` as well as for colorbar.

"""
# 导入 matplotlib.pyplot 库，用于绘图操作
import matplotlib.pyplot as plt

# 创建一个包含单个子图的图像和对应的轴对象
fig, ax = plt.subplots()

# 在轴上创建一个散点图，并传入 x, y 坐标和颜色数据
sc = ax.scatter([1, 2], [1, 2], c=[1, 2])

# 设置 Y 轴的标签文本为 'YLabel'，并指定其位置在轴的顶部
ax.set_ylabel('YLabel', loc='top')

# 设置 X 轴的标签文本为 'XLabel'，并指定其位置在轴的左侧
ax.set_xlabel('XLabel', loc='left')

# 在图像上创建一个颜色条，并将其与之前创建的散点图关联
cbar = fig.colorbar(sc)

# 设置颜色条的标签文本为 'ZLabel'，并指定其位置在颜色条的顶部
cbar.set_label("ZLabel", loc='top')

# 展示绘制的图像
plt.show()
```