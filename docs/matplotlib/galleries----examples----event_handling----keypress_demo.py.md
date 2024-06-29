# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\keypress_demo.py`

```py
"""
==============
Keypress event
==============

Show how to connect to keypress events.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入必要的库
import sys

import matplotlib.pyplot as plt  # 导入 Matplotlib 库的 pyplot 模块
import numpy as np  # 导入 NumPy 库，并使用 np 作为别名


def on_press(event):
    # 当按键按下时触发的函数
    print('press', event.key)  # 打印出按下的按键
    sys.stdout.flush()  # 刷新标准输出流，确保打印及时显示

    if event.key == 'x':
        visible = xl.get_visible()  # 获取 x 轴标签是否可见
        xl.set_visible(not visible)  # 切换 x 轴标签的可见性
        fig.canvas.draw()  # 重新绘制图形


# 设置随机种子以便结果可重现
np.random.seed(19680801)

# 创建一个图形窗口和子图
fig, ax = plt.subplots()

# 将按键事件 'key_press_event' 与 on_press 函数绑定
fig.canvas.mpl_connect('key_press_event', on_press)

# 在子图上绘制随机数据点
ax.plot(np.random.rand(12), np.random.rand(12), 'go')

# 设置 x 轴标签
xl = ax.set_xlabel('easy come, easy go')

# 设置图形标题
ax.set_title('Press a key')

# 显示图形界面
plt.show()
```