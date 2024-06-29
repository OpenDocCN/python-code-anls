# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\figure_axes_enter_leave.py`

```
"""
==================================
Figure/Axes enter and leave events
==================================

Illustrate the figure and Axes enter and leave events by changing the
frame colors on enter and leave.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入 matplotlib.pyplot 库
import matplotlib.pyplot as plt

# 定义当鼠标进入 Axes 区域时触发的函数
def on_enter_axes(event):
    # 打印进入 Axes 的消息和对应的 Axes 对象
    print('enter_axes', event.inaxes)
    # 设置当前 Axes 区域的背景颜色为黄色
    event.inaxes.patch.set_facecolor('yellow')
    # 重新绘制 Canvas，以显示更新后的效果
    event.canvas.draw()

# 定义当鼠标离开 Axes 区域时触发的函数
def on_leave_axes(event):
    # 打印离开 Axes 的消息和对应的 Axes 对象
    print('leave_axes', event.inaxes)
    # 设置当前 Axes 区域的背景颜色为白色
    event.inaxes.patch.set_facecolor('white')
    # 重新绘制 Canvas，以显示更新后的效果
    event.canvas.draw()

# 定义当鼠标进入 Figure 区域时触发的函数
def on_enter_figure(event):
    # 打印进入 Figure 的消息和对应的 Figure 对象
    print('enter_figure', event.canvas.figure)
    # 设置当前 Figure 的背景颜色为红色
    event.canvas.figure.patch.set_facecolor('red')
    # 重新绘制 Canvas，以显示更新后的效果
    event.canvas.draw()

# 定义当鼠标离开 Figure 区域时触发的函数
def on_leave_figure(event):
    # 打印离开 Figure 的消息和对应的 Figure 对象
    print('leave_figure', event.canvas.figure)
    # 设置当前 Figure 的背景颜色为灰色
    event.canvas.figure.patch.set_facecolor('grey')
    # 重新绘制 Canvas，以显示更新后的效果
    event.canvas.draw()

# 创建一个包含两个子图的 Figure 对象
fig, axs = plt.subplots(2, 1)
# 设置总标题
fig.suptitle('mouse hover over figure or Axes to trigger events')

# 将四个事件连接到相应的处理函数
fig.canvas.mpl_connect('figure_enter_event', on_enter_figure)
fig.canvas.mpl_connect('figure_leave_event', on_leave_figure)
fig.canvas.mpl_connect('axes_enter_event', on_enter_axes)
fig.canvas.mpl_connect('axes_leave_event', on_leave_axes)

# 显示绘制的图形
plt.show()
```