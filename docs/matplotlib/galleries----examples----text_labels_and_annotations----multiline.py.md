# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\multiline.py`

```py
"""
=========
Multiline
=========

"""
# 导入 matplotlib.pyplot 库，简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，简写为 np
import numpy as np

# 创建一个包含两个子图的图形对象，指定列数为2，尺寸为7x4
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7, 4))

# 设置 ax0 子图的纵横比为1
ax0.set_aspect(1)
# 在 ax0 子图上绘制简单的折线图，横坐标为0到9
ax0.plot(np.arange(10))
# 设置 ax0 子图的横坐标标签，包含换行
ax0.set_xlabel('this is a xlabel\n(with newlines!)')
# 设置 ax0 子图的纵坐标标签，包含换行和多行对齐
ax0.set_ylabel('this is vertical\ntest', multialignment='center')
# 在 ax0 子图上添加文本，包含换行和旋转
ax0.text(2, 7, 'this is\nyet another test',
         rotation=45,
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center')

# 在 ax0 子图上显示网格
ax0.grid()

# 在 ax1 子图上添加文本，每行文字分别显示，指定位置和大小
ax1.text(0.29, 0.4, "Mat\nTTp\n123", size=18,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

ax1.text(0.34, 0.4, "Mag\nTTT\n123", size=18,
         va="baseline", ha="left", multialignment="left",
         bbox=dict(fc="none"))

ax1.text(0.95, 0.4, "Mag\nTTT$^{A^A}$\n123", size=18,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

# 设置 ax1 子图的横坐标刻度位置和标签，包含换行
ax1.set_xticks([0.2, 0.4, 0.6, 0.8, 1.],
               labels=["Jan\n2009", "Feb\n2009", "Mar\n2009", "Apr\n2009",
                       "May\n2009"])

# 在 ax1 子图上添加水平线
ax1.axhline(0.4)
# 设置 ax1 子图的标题
ax1.set_title("test line spacing for multiline text")

# 调整图形对象的底部和顶部边距
fig.subplots_adjust(bottom=0.25, top=0.75)
# 显示图形对象
plt.show()
```