# `D:\src\scipysrc\matplotlib\galleries\examples\animation\pause_resume.py`

```
"""
=================================
Pausing and Resuming an Animation
=================================

This example showcases:

- using the Animation.pause() method to pause an animation.
- using the Animation.resume() method to resume an animation.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation


class PauseAnimation:
    def __init__(self):
        # 创建图形和轴
        fig, ax = plt.subplots()
        ax.set_title('Click to pause/resume the animation')  # 设置图表标题
        x = np.linspace(-0.1, 0.1, 1000)

        # 初始状态为正态分布
        self.n0 = (1.0 / ((4 * np.pi * 2e-4 * 0.1) ** 0.5)
                   * np.exp(-x ** 2 / (4 * 2e-4 * 0.1)))
        self.p, = ax.plot(x, self.n0)  # 绘制初始曲线并获取曲线对象

        # 创建动画对象
        self.animation = animation.FuncAnimation(
            fig, self.update, frames=200, interval=50, blit=True)
        self.paused = False  # 初始状态为未暂停

        # 连接按钮点击事件到 toggle_pause 方法
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        # 切换暂停状态并调用对应的动画方法
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def update(self, i):
        # 更新动画的每一帧数据
        self.n0 += i / 100 % 5
        self.p.set_ydata(self.n0 % 20)  # 更新曲线数据
        return (self.p,)


pa = PauseAnimation()  # 创建 PauseAnimation 类的实例
plt.show()  # 显示图形界面
```