# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\autowrap.py`

```
"""
==================
Auto-wrapping text
==================

Matplotlib can wrap text automatically, but if it's too long, the text will be
displayed slightly outside of the boundaries of the axis anyways.

Note: Auto-wrapping does not work together with
``savefig(..., bbox_inches='tight')``. The 'tight' setting rescales the canvas
to accommodate all content and happens before wrapping. This affects
``%matplotlib inline`` in IPython and Jupyter notebooks where the inline
setting uses ``bbox_inches='tight'`` by default when saving the image to
embed.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 创建一个新的图形对象
fig = plt.figure()

# 设置坐标轴范围
plt.axis((0, 10, 0, 10))

# 定义一个非常长的字符串，用于测试自动换行功能
t = ("This is a really long string that I'd rather have wrapped so that it "
     "doesn't go outside of the figure, but if it's long enough it will go "
     "off the top or bottom!")

# 在指定位置绘制文本，左对齐，带有15度的旋转，允许自动换行
plt.text(4, 1, t, ha='left', rotation=15, wrap=True)

# 在指定位置绘制文本，左对齐，带有15度的旋转，允许自动换行
plt.text(6, 5, t, ha='left', rotation=15, wrap=True)

# 在指定位置绘制文本，右对齐，带有-15度的旋转，允许自动换行
plt.text(5, 5, t, ha='right', rotation=-15, wrap=True)

# 在指定位置绘制文本，居中对齐，带有18号字体，斜体，允许自动换行
plt.text(5, 10, t, fontsize=18, style='oblique', ha='center',
         va='top', wrap=True)

# 在指定位置绘制文本，右对齐，使用衬线字体，斜体，允许自动换行
plt.text(3, 4, t, family='serif', style='italic', ha='right', wrap=True)

# 在指定位置绘制文本，左对齐，带有-15度的旋转，允许自动换行
plt.text(-1, 0, t, ha='left', rotation=-15, wrap=True)

# 显示绘制的图形
plt.show()
```