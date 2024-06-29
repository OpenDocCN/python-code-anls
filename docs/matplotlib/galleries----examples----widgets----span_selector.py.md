# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\span_selector.py`

```
"""
=============
Span Selector
=============

The `.SpanSelector` is a mouse widget that enables selecting a range on an
axis.

Here, an x-range can be selected on the upper axis; a detailed view of the
selected range is then plotted on the lower axis.

.. note::

    If the SpanSelector object is garbage collected you will lose the
    interactivity.  You must keep a hard reference to it to prevent this.
"""

# 导入 matplotlib 的 pyplot 模块作为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块并重命名为 np
import numpy as np
# 从 matplotlib.widgets 模块导入 SpanSelector 类
from matplotlib.widgets import SpanSelector

# 固定随机数种子以便复现结果
np.random.seed(19680801)

# 创建一个包含两个子图的 Figure 对象，大小为 8x6 英寸
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

# 生成一组 x 轴数据
x = np.arange(0.0, 5.0, 0.01)
# 生成对应的 y 轴数据，y = sin(2πx) + 0.5 * 随机噪声
y = np.sin(2 * np.pi * x) + 0.5 * np.random.randn(len(x))

# 在第一个子图 ax1 上绘制 x 和 y 数据的折线图
ax1.plot(x, y)
# 设置第一个子图的 y 轴范围为 -2 到 2
ax1.set_ylim(-2, 2)
# 设置第一个子图的标题
ax1.set_title('Press left mouse button and drag '
              'to select a region in the top graph')

# 在第二个子图 ax2 上绘制空的折线图
line2, = ax2.plot([], [])


# 定义一个函数 onselect，当选取区域发生变化时调用
def onselect(xmin, xmax):
    # 使用 np.searchsorted 在 x 数据中找到 xmin 和 xmax 的索引范围
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    # 确保 indmax 不超过数据长度减 1
    indmax = min(len(x) - 1, indmax)

    # 提取选取区域的 x 和 y 数据
    region_x = x[indmin:indmax]
    region_y = y[indmin:indmax]

    # 如果选取区域包含至少两个点，则更新第二个子图的折线图和坐标轴范围
    if len(region_x) >= 2:
        line2.set_data(region_x, region_y)
        ax2.set_xlim(region_x[0], region_x[-1])
        ax2.set_ylim(region_y.min(), region_y.max())
        # 绘图区域对象发生变化，需要重新绘制图形
        fig.canvas.draw_idle()


# 创建一个 SpanSelector 对象 span，绑定到第一个子图 ax1 上
span = SpanSelector(
    ax1,
    onselect,
    "horizontal",
    useblit=True,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
)
# 在大多数后端设置 useblit=True 可以提升性能。

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.SpanSelector`
```