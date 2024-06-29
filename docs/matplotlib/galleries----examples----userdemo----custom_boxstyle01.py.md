# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\custom_boxstyle01.py`

```py
r"""
=================
Custom box styles
=================

This example demonstrates the implementation of a custom `.BoxStyle`.
Custom `.ConnectionStyle`\s and `.ArrowStyle`\s can be similarly defined.
"""

import matplotlib.pyplot as plt

from matplotlib.patches import BoxStyle
from matplotlib.path import Path

# %%
# Custom box styles can be implemented as a function that takes arguments
# specifying both a rectangular box and the amount of "mutation", and
# returns the "mutated" path.  The specific signature is the one of
# ``custom_box_style`` below.
#
# Here, we return a new path which adds an "arrow" shape on the left of the
# box.
#
# The custom box style can then be used by passing
# ``bbox=dict(boxstyle=custom_box_style, ...)`` to `.Axes.text`.

def custom_box_style(x0, y0, width, height, mutation_size):
    """
    Given the location and size of the box, return the path of the box around
    it.

    Rotation is automatically taken care of.

    Parameters
    ----------
    x0, y0, width, height : float
        Box location and size.
    mutation_size : float
        Mutation reference scale, typically the text font size.
    """
    # padding
    mypad = 0.3
    pad = mutation_size * mypad
    # width and height with padding added.
    width = width + 2 * pad
    height = height + 2 * pad
    # boundary of the padded box
    x0, y0 = x0 - pad, y0 - pad
    x1, y1 = x0 + width, y0 + height
    # return the new path
    return Path([(x0, y0),
                 (x1, y0), (x1, y1), (x0, y1),
                 (x0-pad, (y0+y1)/2), (x0, y0),
                 (x0, y0)],
                closed=True)


fig, ax = plt.subplots(figsize=(3, 3))
# 在坐标 (0.5, 0.5) 处放置文本 "Test"，使用自定义的盒子样式 custom_box_style
ax.text(0.5, 0.5, "Test", size=30, va="center", ha="center", rotation=30,
        bbox=dict(boxstyle=custom_box_style, alpha=0.2))


# %%
# Likewise, custom box styles can be implemented as classes that implement
# ``__call__``.
#
# The classes can then be registered into the ``BoxStyle._style_list`` dict,
# which allows specifying the box style as a string,
# ``bbox=dict(boxstyle="registered_name,param=value,...", ...)``.
# Note that this registration relies on internal APIs and is therefore not
# officially supported.

class MyStyle:
    """A simple box."""

    def __init__(self, pad=0.3):
        """
        The arguments must be floats and have default values.

        Parameters
        ----------
        pad : float
            amount of padding
        """
        self.pad = pad
        super().__init__()
    def __call__(self, x0, y0, width, height, mutation_size):
        """
        给定框的位置和大小，返回围绕它的路径。

        自动处理旋转。

        Parameters
        ----------
        x0, y0, width, height : float
            框的位置和大小。
        mutation_size : float
            变异的参考尺度，通常是文本字体大小。
        """
        # 计算填充量
        pad = mutation_size * self.pad
        # 加上填充后的宽度和高度
        width = width + 2.*pad
        height = height + 2.*pad
        # 计算填充后框的边界
        x0, y0 = x0 - pad, y0 - pad
        x1, y1 = x0 + width, y0 + height
        # 返回新路径
        return Path([(x0, y0),
                     (x1, y0), (x1, y1), (x0, y1),
                     (x0-pad, (y0+y1)/2.), (x0, y0),
                     (x0, y0)],
                    closed=True)
# 注册自定义样式到 BoxStyle._style_list 中的 "angled" 键
BoxStyle._style_list["angled"] = MyStyle  # Register the custom style.

# 创建一个新的图形对象和坐标轴对象
fig, ax = plt.subplots(figsize=(3, 3))

# 在坐标轴上添加文本 "Test"
# 设置文本的大小为30，垂直和水平对齐方式为中心
# 设置文本的旋转角度为30度
# 设置文本周围的边框样式为 "angled"，并设置边框的填充为0.5
# 设置边框的透明度为0.2
ax.text(0.5, 0.5, "Test", size=30, va="center", ha="center", rotation=30,
        bbox=dict(boxstyle="angled,pad=0.5", alpha=0.2))

# 从 BoxStyle._style_list 中删除 "angled" 键对应的样式
del BoxStyle._style_list["angled"]  # Unregister it.

# 显示图形
plt.show()
```