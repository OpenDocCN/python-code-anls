# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\close_event.py`

```
"""
===========
Close Event
===========

Example to show connecting events that occur when the figure closes.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""
# 导入 Matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt

# 定义一个事件处理函数，当图形关闭时被调用
def on_close(event):
    # 打印消息到控制台
    print('Closed Figure!')

# 创建一个新的图形对象
fig = plt.figure()
# 将事件处理函数与图形对象的关闭事件连接起来
fig.canvas.mpl_connect('close_event', on_close)

# 在图形上添加文本，提示用户关闭图形
plt.text(0.35, 0.5, 'Close Me!', dict(size=30))
# 显示图形界面，交互式操作可以看到图形关闭时的效果
plt.show()
```