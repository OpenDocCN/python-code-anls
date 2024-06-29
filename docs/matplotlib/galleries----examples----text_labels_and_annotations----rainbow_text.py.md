# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\rainbow_text.py`

```py
"""
====================================================
Concatenating text objects with different properties
====================================================

The example strings together several Text objects with different properties
(e.g., color or font), positioning each one after the other. The first Text
is created directly using `~.Axes.text`; all subsequent ones are created with
`~.Axes.annotate`, which allows positioning the Text's lower left corner at the
lower right corner (``xy=(1, 0)``) of the previous one (``xycoords=text``).
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

plt.rcParams["font.size"] = 20  # 设置全局字体大小为 20

ax = plt.figure().add_subplot(xticks=[], yticks=[])  # 创建一个图表并添加子图，无 x 和 y 轴刻度

# The first word, created with text().
# 使用 text() 创建第一个文本对象，放置在坐标 (0.1, 0.5)，文本内容为 "Matplotlib"，红色字体
text = ax.text(.1, .5, "Matplotlib", color="red")

# Subsequent words, positioned with annotate(), relative to the preceding one.
# 使用 annotate() 创建后续文本对象，相对于前一个对象进行定位
text = ax.annotate(
    " says,", xycoords=text, xy=(1, 0), verticalalignment="bottom",
    color="gold", weight="bold")  # 自定义属性：金色加粗字体，位于前一个文本对象右下角

text = ax.annotate(
    " hello", xycoords=text, xy=(1, 0), verticalalignment="bottom",
    color="green", style="italic")  # 自定义属性：绿色斜体字体，位于前一个文本对象右下角

text = ax.annotate(
    " world!", xycoords=text, xy=(1, 0), verticalalignment="bottom",
    color="blue", family="serif")  # 自定义属性：蓝色衬线字体，位于前一个文本对象右下角

plt.show()  # 显示绘制的图形
```