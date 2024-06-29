# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\pick_event_demo.py`

```py
"""
===============
Pick event demo
===============

You can enable picking by setting the "picker" property of an artist
(for example, a Matplotlib Line2D, Text, Patch, Polygon, AxesImage,
etc.)

There are a variety of meanings of the picker property:

* *None* - picking is disabled for this artist (default)

* bool - if *True* then picking will be enabled and the artist will fire a pick
  event if the mouse event is over the artist.

  Setting ``pickradius`` will add an epsilon tolerance in points and the artist
  will fire off an event if its data is within epsilon of the mouse event.  For
  some artists like lines and patch collections, the artist may provide
  additional data to the pick event that is generated, for example, the indices
  of the data within epsilon of the pick event

* function - if picker is callable, it is a user supplied function which
  determines whether the artist is hit by the mouse event. ::

     hit, props = picker(artist, mouseevent)

  to determine the hit test.  If the mouse event is over the artist, return
  hit=True and props is a dictionary of properties you want added to the
  PickEvent attributes.

After you have enabled an artist for picking by setting the "picker"
property, you need to connect to the figure canvas pick_event to get
pick callbacks on mouse press events.  For example, ::

  def pick_handler(event):
      mouseevent = event.mouseevent
      artist = event.artist
      # now do something with this...


The pick event (matplotlib.backend_bases.PickEvent) which is passed to
your callback is always fired with two attributes:

mouseevent
  the mouse event that generate the pick event.

  The mouse event in turn has attributes like x and y (the coordinates in
  display space, such as pixels from left, bottom) and xdata, ydata (the
  coords in data space).  Additionally, you can get information about
  which buttons were pressed, which keys were pressed, which Axes
  the mouse is over, etc.  See matplotlib.backend_bases.MouseEvent
  for details.

artist
  the matplotlib.artist that generated the pick event.

Additionally, certain artists like Line2D and PatchCollection may
attach additional metadata like the indices into the data that meet
the picker criteria (for example, all the points in the line that are within
the specified epsilon tolerance)

The examples below illustrate each of these methods.

.. note::
    These examples exercises the interactive capabilities of Matplotlib, and
    this will not appear in the static documentation. Please run this code on
    your machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块，并命名为 plt
import numpy as np  # 导入 NumPy 库，并命名为 np
from numpy.random import rand  # 从 NumPy 的随机模块中导入 rand 函数

from matplotlib.image import AxesImage  # 从 Matplotlib 的图像模块中导入 AxesImage 类
from matplotlib.lines import Line2D  # 从 Matplotlib 的线条模块中导入 Line2D 类
from matplotlib.patches import Rectangle  # 从 Matplotlib 的补丁模块中导入 Rectangle 类
from matplotlib.text import Text  # 从 Matplotlib 的文本模块中导入 Text 类
# 设置随机种子以便重现随机状态
np.random.seed(19680801)

# %%
# 简单的图形元素选择：线条、矩形和文本
# ------------------------------------------

# 创建包含两个子图的图形窗口
fig, (ax1, ax2) = plt.subplots(2, 1)
# 设置第一个子图标题，并使其可被选择
ax1.set_title('click on points, rectangles or text', picker=True)
# 设置第一个子图的y轴标签，并使其可被选择，在一个红色的边框内
ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
# 在第一个子图上绘制散点图，使其可被选择，选择半径为5个像素
line, = ax1.plot(np.random.rand(100), 'o', picker=True, pickradius=5)

# 在第二个子图上绘制柱状图，使其可被选择
ax2.bar(range(10), np.random.rand(10), picker=True)
# 使第二个子图的x轴刻度标签可被选择
for label in ax2.get_xticklabels():
    label.set_picker(True)


def onpick1(event):
    # 判断事件的艺术家类型是否为Line2D（线条）
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        print('onpick1 line:', np.column_stack([xdata[ind], ydata[ind]]))
    # 判断事件的艺术家类型是否为Rectangle（矩形）
    elif isinstance(event.artist, Rectangle):
        patch = event.artist
        print('onpick1 patch:', patch.get_path())
    # 判断事件的艺术家类型是否为Text（文本）
    elif isinstance(event.artist, Text):
        text = event.artist
        print('onpick1 text:', text.get_text())


# 将onpick1函数连接到图形窗口的pick_event事件
fig.canvas.mpl_connect('pick_event', onpick1)

# %%
# 使用自定义命中测试函数进行选择
# ---------------------------------------
# 您可以通过将picker设置为可调用函数来定义自定义选择器。函数的签名如下::
#
#  hit, props = func(artist, mouseevent)
#
# 用于确定命中测试。如果鼠标事件位于艺术家上，则返回``hit=True``和``props``字典，
# 其中包含要添加到`.PickEvent`属性的属性。

def line_picker(line, mouseevent):
    """
    在数据坐标中查找距离鼠标点击点一定距离内的点，并附加一些额外属性，如pickx和picky，表示选中的数据点。
    """
    if mouseevent.xdata is None:
        return False, dict()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    maxd = 0.05
    d = np.sqrt(
        (xdata - mouseevent.xdata)**2 + (ydata - mouseevent.ydata)**2)

    ind, = np.nonzero(d <= maxd)
    if len(ind):
        pickx = xdata[ind]
        picky = ydata[ind]
        props = dict(ind=ind, pickx=pickx, picky=picky)
        return True, props
    else:
        return False, dict()


def onpick2(event):
    print('onpick2 line:', event.pickx, event.picky)


# 创建一个新的图形窗口
fig, ax = plt.subplots()
ax.set_title('custom picker for line data')
# 在图中绘制散点图，并使用line_picker作为选择器
line, = ax.plot(np.random.rand(100), np.random.rand(100), 'o', picker=line_picker)
# 将onpick2函数连接到图形窗口的pick_event事件
fig.canvas.mpl_connect('pick_event', onpick2)

# %%
# 在散点图上进行选择
# -------------------------
# 散点图由 `~matplotlib.collections.PathCollection` 支持。

x, y, c, s = np.random.rand(4, 100)


def onpick3(event):
    ind = event.ind
    print('onpick3 scatter:', ind, x[ind], y[ind])


# 创建一个新的图形窗口
fig, ax = plt.subplots()
# 在图中绘制散点图，并使其可被选择
ax.scatter(x, y, 100*s, c, picker=True)
# 将onpick3函数连接到图形窗口的pick_event事件
fig.canvas.mpl_connect('pick_event', onpick3)

# %%
# 在图像上进行选择
# --------------
# 使用 `.Axes.imshow` 绘制的图像是 `~matplotlib.image.AxesImage` 对象。
# 创建一个新的图形窗口，并返回包含图形和轴对象的元组
fig, ax = plt.subplots()

# 在轴对象上显示一个随机数矩阵的图像，设置其在坐标轴上的位置和触发拾取事件
ax.imshow(rand(10, 5), extent=(1, 2, 1, 2), picker=True)

# 在同一轴上显示另一个随机数矩阵的图像，设置其在坐标轴上的位置和触发拾取事件
ax.imshow(rand(5, 10), extent=(3, 4, 1, 2), picker=True)

# 再次在轴上显示另一个随机数矩阵的图像，设置其在坐标轴上的位置和触发拾取事件
ax.imshow(rand(20, 25), extent=(1, 2, 3, 4), picker=True)

# 在同一轴上显示另一个随机数矩阵的图像，设置其在坐标轴上的位置和触发拾取事件
ax.imshow(rand(30, 12), extent=(3, 4, 3, 4), picker=True)

# 设置坐标轴的范围为 (0, 5) 和 (0, 5)
ax.set(xlim=(0, 5), ylim=(0, 5))


# 定义一个事件处理函数，当拾取事件发生时调用
def onpick4(event):
    # 获取触发事件的艺术家对象
    artist = event.artist
    # 检查是否是 AxesImage 对象
    if isinstance(artist, AxesImage):
        # 如果是图像对象，则获取其数据数组
        im = artist
        A = im.get_array()
        # 打印图像数据数组的形状信息
        print('onpick4 image', A.shape)


# 将绘图窗口的事件连接到定义的事件处理函数上
fig.canvas.mpl_connect('pick_event', onpick4)

# 显示绘图窗口
plt.show()
```