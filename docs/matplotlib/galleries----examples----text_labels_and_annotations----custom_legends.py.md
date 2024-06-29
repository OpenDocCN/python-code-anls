# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\custom_legends.py`

```
"""
========================
Composing Custom Legends
========================

Composing custom legends piece-by-piece.

.. note::

   For more information on creating and customizing legends, see the following
   pages:

   * :ref:`legend_guide`
   * :doc:`/gallery/text_labels_and_annotations/legend_demo`

Sometimes you don't want a legend that is explicitly tied to data that
you have plotted. For example, say you have plotted 10 lines, but don't
want a legend item to show up for each one. If you simply plot the lines
and call ``ax.legend()``, you will get the following:
"""
# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# sphinx_gallery_thumbnail_number = 2
# 导入 matplotlib 的整体配置，包括绘图风格等
import matplotlib as mpl
# 从 matplotlib 中的 cycler 模块导入 cycler 对象
from matplotlib import cycler

# 设置随机种子以便复现结果
np.random.seed(19680801)

# %%
# 定义数据集大小 N
N = 10
# 生成具有随机噪声的数据集
data = (np.geomspace(1, 10, 100) + np.random.randn(N, 100)).T
# 使用 coolwarm 颜色映射创建颜色映射对象
cmap = plt.cm.coolwarm
# 将颜色映射设置为默认绘图风格的轮换（循环）
mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))

# 创建图形和坐标系
fig, ax = plt.subplots()
# 绘制数据并将返回的线条对象存储在 lines 中
lines = ax.plot(data)

# %%
# 由于数据没有标签，因此需要我们定义图例的图标和标签。
# 在这种情况下，可以使用不直接与绘制的数据相关联的 Matplotlib 对象来组合图例。
# 例如：

from matplotlib.lines import Line2D

# 创建自定义的线条对象列表
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

# 创建新的图形和坐标系
fig, ax = plt.subplots()
# 再次绘制数据
lines = ax.plot(data)
# 添加图例，使用自定义线条和对应的标签
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot'])


# %%
# 还有许多其他的 Matplotlib 对象可以用来创建类似的图例。在下面的代码中，我们列出了一些常见的用法。

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 定义图例元素列表，包含线条和色块
legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
                   Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='g', markersize=15),
                   Patch(facecolor='orange', edgecolor='r',
                         label='Color Patch')]

# 创建新的图形和坐标系
fig, ax = plt.subplots()
# 添加图例，使用预定义的图例元素和位置参数 'center'
ax.legend(handles=legend_elements, loc='center')

# 显示图形
plt.show()
```