# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\auto_subplots_adjust.py`

```
"""
===============================================
Programmatically controlling subplot adjustment
===============================================

.. note::

    This example is primarily intended to show some advanced concepts in
    Matplotlib.

    If you are only looking for having enough space for your labels, it is
    almost always simpler and good enough to either set the subplot parameters
    manually using `.Figure.subplots_adjust`, or use one of the automatic
    layout mechanisms
    (:ref:`constrainedlayout_guide` or
    :ref:`tight_layout_guide`).

This example describes a user-defined way to read out Artist sizes and
set the subplot parameters accordingly. Its main purpose is to illustrate
some advanced concepts like reading out text positions, working with
bounding boxes and transforms and using
:ref:`events <event-handling>`. But it can also serve as a starting
point if you want to automate the layouting and need more flexibility than
tight layout and constrained layout.

Below, we collect the bounding boxes of all y-labels and move the left border
of the subplot to the right so that it leaves enough room for the union of all
the bounding boxes.

There's one catch with calculating text bounding boxes:
Querying the text bounding boxes (`.Text.get_window_extent`) needs a
renderer (`.RendererBase` instance), to calculate the text size. This renderer
is only available after the figure has been drawn (`.Figure.draw`).

A solution to this is putting the adjustment logic in a draw callback.
This function is executed after the figure has been drawn. It can now check
if the subplot leaves enough room for the text. If not, the subplot parameters
are updated and second draw is triggered.

.. redirect-from:: /gallery/pyplots/auto_subplots_adjust
"""

import matplotlib.pyplot as plt  # 导入 Matplotlib 绘图库

import matplotlib.transforms as mtransforms  # 导入 Matplotlib 的 transforms 模块

# 创建一个包含单个子图的 Figure 对象和对应的 Axes 对象
fig, ax = plt.subplots()
# 在 Axes 对象上绘制简单的折线图
ax.plot(range(10))
# 设置 y 轴刻度及其标签
ax.set_yticks([2, 5, 7], labels=['really, really, really', 'long', 'labels'])

# 定义一个绘图事件的回调函数，用于处理自动调整 subplot 的逻辑
def on_draw(event):
    bboxes = []
    # 遍历 y 轴刻度标签，并获取每个标签的像素级边界框
    for label in ax.get_yticklabels():
        # 获取标签的像素级边界框
        bbox_px = label.get_window_extent()
        # 将像素级边界框转换为相对于图形的坐标系（通过图形转换对象 fig.transFigure 的反向变换）
        bbox_fig = bbox_px.transformed(fig.transFigure.inverted())
        bboxes.append(bbox_fig)
    # 计算所有边界框的并集，同样以相对图形坐标表示
    bbox = mtransforms.Bbox.union(bboxes)
    # 如果 subplot 的左边缘不足以容纳所有标签的边界框宽度
    if fig.subplotpars.left < bbox.width:
        # 将 subplot 的左边缘向右移动，以容纳边界框的宽度（在左侧留出一些空间）
        fig.subplots_adjust(left=1.1 * bbox.width)  # 稍微增加一点空间
        # 重新绘制画布
        fig.canvas.draw()

# 将绘图事件的回调函数绑定到图形的绘制事件上
fig.canvas.mpl_connect('draw_event', on_draw)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.artist.Artist.get_window_extent`
#    - `matplotlib.transforms.Bbox`
#    - `matplotlib.transforms.BboxBase.transformed`
# 导入 `matplotlib.transforms` 模块中的 `BboxBase.union` 函数
# 用于合并两个边界框对象（Bbox），返回包含它们的最小边界框

# 导入 `matplotlib.transforms` 模块中的 `Transform.inverted` 方法
# 用于获取当前变换的反变换（逆变换），将目标坐标系映射回原始坐标系

# 导入 `matplotlib.figure` 模块中的 `Figure.subplots_adjust` 方法
# 用于调整图形中子图的布局，包括间距和外边距的调整

# 导入 `matplotlib.gridspec` 模块中的 `SubplotParams` 类
# 用于配置子图的参数，例如间距、高度比例等

# 导入 `matplotlib.backend_bases` 模块中的 `FigureCanvasBase.mpl_connect` 方法
# 用于建立图形画布与事件的连接，例如绑定特定事件处理函数到特定事件的响应上
```