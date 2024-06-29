# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\placing_text_boxes.py`

```
"""
Placing text boxes
==================

When decorating Axes with text boxes, two useful tricks are to place the text
in axes coordinates (see :ref:`transforms_tutorial`),
so the text doesn't move around with changes in x or y limits.  You
can also use the ``bbox`` property of text to surround the text with a
`~matplotlib.patches.Patch` instance -- the ``bbox`` keyword argument takes a
dictionary with keys that are Patch properties.
"""

# 导入 matplotlib.pyplot 库并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np

# 设定随机数种子，以便结果可复现
np.random.seed(19680801)

# 创建一个新的 Figure 和一个 Axes 对象
fig, ax = plt.subplots()
# 生成服从标准正态分布的随机数数组
x = 30*np.random.randn(10000)
# 计算随机数数组的均值、中位数和标准差
mu = x.mean()
median = np.median(x)
sigma = x.std()
# 创建包含统计信息的文本字符串
textstr = '\n'.join((
    r'$\mu=%.2f$' % (mu, ),
    r'$\mathrm{median}=%.2f$' % (median, ),
    r'$\sigma=%.2f$' % (sigma, )))

# 在 Axes 对象上绘制直方图，分成50个柱子
ax.hist(x, 50)
# 定义文本框的样式属性，圆角矩形、浅黄色背景、半透明
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# 在 Axes 坐标系中的指定位置（左上角）放置一个文本框，显示统计信息
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

# 显示图形
plt.show()
```