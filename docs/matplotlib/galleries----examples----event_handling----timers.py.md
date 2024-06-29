# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\timers.py`

```py
"""
======
Timers
======

Simple example of using general timer objects. This is used to update
the time placed in the title of the figure.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 导入 matplotlib.pyplot 模块并重命名为 plt
import matplotlib.pyplot as plt

# 导入 numpy 模块并重命名为 np
import numpy as np

# 定义一个函数 update_title，用于更新图表的标题为当前时间
def update_title(axes):
    # 设置图表 axes 的标题为当前时间
    axes.set_title(datetime.now())
    # 重新绘制图表
    axes.figure.canvas.draw()

# 创建一个图表对象和一个坐标轴对象
fig, ax = plt.subplots()

# 生成一组 x 值数组，范围从 -3 到 3
x = np.linspace(-3, 3)
# 在坐标轴 ax 上绘制 x^2 的图像
ax.plot(x, x ** 2)

# 创建一个新的定时器对象。设置定时器的间隔为 100 毫秒（默认是 1000 毫秒），
# 并告诉定时器在触发时调用 update_title 函数。
timer = fig.canvas.new_timer(interval=100)
timer.add_callback(update_title, ax)
# 启动定时器
timer.start()

# 或者在第一次绘制图表时启动定时器：
# def start_timer(event):
#     timer.start()
#     fig.canvas.mpl_disconnect(drawid)
# drawid = fig.canvas.mpl_connect('draw_event', start_timer)

# 显示图表
plt.show()
```