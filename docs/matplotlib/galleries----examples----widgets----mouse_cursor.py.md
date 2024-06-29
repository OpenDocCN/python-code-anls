# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\mouse_cursor.py`

```py
"""
============
Mouse Cursor
============

This example sets an alternative cursor on a figure canvas.

Note, this is an interactive example, and must be run to see the effect.
"""

# 导入 matplotlib.pyplot 库，并引入 Cursors 工具
import matplotlib.pyplot as plt
from matplotlib.backend_tools import Cursors

# 创建一个包含多个子图的图形对象 fig，并设置子图数量与大小
fig, axs = plt.subplots(len(Cursors), figsize=(6, len(Cursors) + 0.5),
                        gridspec_kw={'hspace': 0})

# 设置整个图形的标题
fig.suptitle('Hover over an Axes to see alternate Cursors')

# 遍历 Cursors 列表和对应的子图 axs
for cursor, ax in zip(Cursors, axs):
    # 设置当前子图的鼠标指针样式
    ax.cursor_to_use = cursor
    # 在子图中心位置添加文本，显示当前鼠标指针的名称
    ax.text(0.5, 0.5, cursor.name,
            horizontalalignment='center', verticalalignment='center')
    # 设置子图的刻度为空列表，不显示刻度
    ax.set(xticks=[], yticks=[])


# 定义一个事件处理函数 hover，响应鼠标移动事件
def hover(event):
    # 如果图形的小部件被锁定（如启用了缩放/平移工具），则不执行任何操作
    if fig.canvas.widgetlock.locked():
        return

    # 根据事件发生的子图设置鼠标指针样式
    fig.canvas.set_cursor(
        event.inaxes.cursor_to_use if event.inaxes else Cursors.POINTER)


# 将 hover 函数与图形对象的鼠标移动事件绑定
fig.canvas.mpl_connect('motion_notify_event', hover)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.backend_bases.FigureCanvasBase.set_cursor`
```